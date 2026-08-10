"""Qt-free display policy for anatomical slice views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ephys_alignment_gui.core.image_levels import brain_percentile_levels

_METADATA_KEYS = frozenset({"ccf", "label", "annotation_ids", "scale", "offset"})


class SliceImageKind(Enum):
    """How a slice image should be rendered."""

    LABEL = "label"
    RGB = "rgb"
    SCALAR = "scalar"


@dataclass(frozen=True)
class SliceSelection:
    """A selectable slice image source."""

    data_attr: str
    key: str

    def to_payload(self) -> tuple[str, str]:
        """Return the Qt action payload shape used by the legacy UI."""
        return self.data_attr, self.key

    @classmethod
    def from_payload(cls, payload: Any) -> SliceSelection | None:
        """Parse a Qt action payload into a slice selection."""
        if isinstance(payload, cls):
            return payload
        if (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[0], str)
            and isinstance(payload[1], str)
        ):
            return cls(payload[0], payload[1])
        return None


@dataclass(frozen=True)
class SliceMenuItem:
    """One item in the Slice Plots menu."""

    label: str
    selection: SliceSelection


@dataclass(frozen=True)
class SliceSelectionDecision:
    """Selected slice source after applying fallback policy."""

    selection: SliceSelection
    used_previous: bool


@dataclass(frozen=True)
class SliceRenderDecision:
    """Display classification and scalar initialization for one slice image."""

    kind: SliceImageKind
    scalar_channel: str | None = None
    initial_levels: tuple[float, float] | None = None


class SliceDisplayPolicy:
    """Slice menu, fallback, and image-classification policy."""

    ccf_selection = SliceSelection("slice_data", "ccf")
    annotation_selection = SliceSelection("slice_data", "label")
    fp_annotation_selection = SliceSelection("fp_slice_data", "label")
    hist_cb_selection = SliceSelection("slice_data", "hist_cb")
    preferred_scalar_key = "histology_registration"

    def menu_items(
        self,
        *,
        slice_data: Mapping[str, Any],
        fp_slice_data: Mapping[str, Any] | None,
        offline: bool,
    ) -> list[SliceMenuItem]:
        """Return Slice Plots menu items."""
        items = [
            SliceMenuItem("CCF", self.ccf_selection),
            SliceMenuItem("Annotation", self.annotation_selection),
        ]
        if fp_slice_data is not None:
            items.append(SliceMenuItem("Annotation FP", self.fp_annotation_selection))
        if not offline:
            items.append(
                SliceMenuItem("Histology cerebellar example", self.hist_cb_selection)
            )

        for key in self.plottable_channels(slice_data):
            items.append(SliceMenuItem(key, SliceSelection("slice_data", key)))

        return self._dedupe_items(items)

    def default_selection(self, slice_data: Mapping[str, Any]) -> SliceSelection:
        """Prefer registered histology when present; otherwise use CCF."""
        if self.preferred_scalar_key in slice_data:
            return SliceSelection("slice_data", self.preferred_scalar_key)
        return self.ccf_selection

    def plottable_channels(self, slice_data: Mapping[str, Any]) -> list[str]:
        """Return non-metadata slice keys that should be menu items."""
        return [key for key in slice_data.keys() if key not in _METADATA_KEYS]

    def choose_selection(
        self,
        *,
        previous: SliceSelection | None,
        default: SliceSelection,
        data_by_attr: Mapping[str, Mapping[str, Any] | None],
    ) -> SliceSelectionDecision:
        """Restore previous selection when available, otherwise use default."""
        if previous is not None and self.selection_available(data_by_attr, previous):
            return SliceSelectionDecision(previous, used_previous=True)
        return SliceSelectionDecision(default, used_previous=False)

    def selection_available(
        self,
        data_by_attr: Mapping[str, Mapping[str, Any] | None],
        selection: SliceSelection,
    ) -> bool:
        """Whether a selection exists in the current slice data."""
        data = data_by_attr.get(selection.data_attr)
        return data is not None and selection.key in data

    def scalar_channel_for_selection(
        self,
        data_by_attr: Mapping[str, Mapping[str, Any] | None],
        selection: SliceSelection,
    ) -> str | None:
        """Return the scalar channel for a selection, if it is scalar."""
        data = data_by_attr.get(selection.data_attr)
        if data is None or selection.key not in data:
            return None
        decision = self.render_decision(data, selection.key)
        return decision.scalar_channel

    def render_decision(
        self,
        data: Mapping[str, Any],
        img_type: str,
    ) -> SliceRenderDecision:
        """Classify a slice image and compute scalar initial levels."""
        image = data[img_type]
        kind = self.classify_image(img_type, image)
        if kind is not SliceImageKind.SCALAR:
            return SliceRenderDecision(kind)

        return SliceRenderDecision(
            kind=SliceImageKind.SCALAR,
            scalar_channel=img_type,
            initial_levels=brain_percentile_levels(
                image,
                data.get("annotation_ids"),
            ),
        )

    def classify_image(self, img_type: str, image: Any) -> SliceImageKind:
        """Classify the image payload independent of rendering backend."""
        if img_type == "label":
            return SliceImageKind.LABEL
        if np.asarray(image).ndim == 3:
            return SliceImageKind.RGB
        return SliceImageKind.SCALAR

    @staticmethod
    def selection_from_payload(payload: Any) -> SliceSelection | None:
        """Parse a Qt action payload."""
        return SliceSelection.from_payload(payload)

    @staticmethod
    def _dedupe_items(items: list[SliceMenuItem]) -> list[SliceMenuItem]:
        deduped: list[SliceMenuItem] = []
        seen: set[SliceSelection] = set()
        for item in items:
            if item.selection in seen:
                continue
            seen.add(item.selection)
            deduped.append(item)
        return deduped
