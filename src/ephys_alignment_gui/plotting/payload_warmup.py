"""Qt-free warmup job for ephys plot payload caches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.core.workflow import Failed
from ephys_alignment_gui.plotting.menu_state import build_plot_menu_state
from ephys_alignment_gui.plotting.payload_cache import EphysPlotPayloadCache
from ephys_alignment_gui.plotting.payload_cache_factory import (
    EphysPlotPayloadCacheFactory,
)
from ephys_alignment_gui.plotting.registry import plot_spec, resolve_plot_payload
from ephys_alignment_gui.runtime.ephys_stream import StreamKey
from ephys_alignment_gui.services.ephys_data import EphysStreamData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlotPayloadWarmupRequest:
    """Inputs for warming one inactive stream/shank plot payload cache."""

    stream_key: StreamKey
    stream: EphysStreamData
    shank_idx: int
    unit_filter: str
    spec_keys: tuple[str, ...] | None = None
    raster_request: Any | None = None


@dataclass
class PlotPayloadWarmupCancelToken:
    """Cooperative cancellation flag for plot payload warmup jobs."""

    reason: str | None = None

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self.reason is not None

    def cancel(self, reason: str = "cancelled") -> None:
        """Request cancellation at the next warmup checkpoint."""
        self.reason = reason


@dataclass(frozen=True)
class PlotPayloadWarmupCancelled:
    """Plot payload warmup was cancelled at a cooperative checkpoint."""

    stream_key: StreamKey
    shank_idx: int
    reason: str


@dataclass(frozen=True)
class PlotPayloadCacheWarmed:
    """Warmed plot payload cache ready to attach to an inactive runtime."""

    stream_key: StreamKey
    stream: EphysStreamData
    shank_idx: int
    unit_filter: str
    payload_cache: EphysPlotPayloadCache
    warmed_spec_keys: tuple[str, ...]


class PlotPayloadWarmupJob:
    """Build and warm ephys plot payload caches without depending on Qt."""

    def __init__(
        self,
        plot_payload_cache_factory: EphysPlotPayloadCacheFactory,
    ) -> None:
        self.plot_payload_cache_factory = plot_payload_cache_factory

    def run(
        self,
        request: PlotPayloadWarmupRequest,
        *,
        cancel_token: PlotPayloadWarmupCancelToken | None = None,
    ) -> PlotPayloadCacheWarmed | PlotPayloadWarmupCancelled | Failed:
        """Warm one stream/shank plot cache for later activation."""
        cancel_token = cancel_token or PlotPayloadWarmupCancelToken()
        cancelled = _cancelled(request, cancel_token)
        if cancelled is not None:
            return cancelled

        try:
            payload_cache = self.plot_payload_cache_factory.build_for_stream(
                request.stream,
                request.shank_idx,
            )
            cancelled = _cancelled(request, cancel_token)
            if cancelled is not None:
                return cancelled
            payload_cache.filter_units(request.unit_filter)
            cancelled = _cancelled(request, cancel_token)
            if cancelled is not None:
                return cancelled
            menu_state = build_plot_menu_state(payload_cache)
            warmed_spec_keys = self._warm_spec_payloads(
                request,
                payload_cache,
                request.spec_keys or _selected_menu_spec_keys(menu_state),
                cancel_token=cancel_token,
                raster_request=request.raster_request,
            )
            if isinstance(warmed_spec_keys, PlotPayloadWarmupCancelled):
                return warmed_spec_keys
        except Exception as exc:
            logger.warning(
                "Plot payload warmup failed for %s shank %s",
                request.stream_key,
                request.shank_idx,
                exc_info=True,
            )
            return Failed(f"Plot payload warmup failed: {exc}")

        return PlotPayloadCacheWarmed(
            stream_key=request.stream_key,
            stream=request.stream,
            shank_idx=request.shank_idx,
            unit_filter=request.unit_filter,
            payload_cache=payload_cache,
            warmed_spec_keys=warmed_spec_keys,
        )

    @staticmethod
    def _warm_spec_payloads(
        request: PlotPayloadWarmupRequest,
        payload_cache: EphysPlotPayloadCache,
        spec_keys: tuple[str, ...],
        *,
        cancel_token: PlotPayloadWarmupCancelToken,
        raster_request: Any | None = None,
    ) -> tuple[str, ...] | PlotPayloadWarmupCancelled:
        warmed: list[str] = []
        for spec_key in spec_keys:
            cancelled = _cancelled(request, cancel_token)
            if cancelled is not None:
                return cancelled
            spec = plot_spec(spec_key)
            if spec.available is not None and not spec.available(payload_cache):
                continue
            if (
                resolve_plot_payload(
                    payload_cache,
                    spec,
                    raster_request=raster_request,
                )
                is not None
            ):
                warmed.append(spec.key)
            cancelled = _cancelled(request, cancel_token)
            if cancelled is not None:
                return cancelled
        return tuple(warmed)


def _selected_menu_spec_keys(menu_state: Any) -> tuple[str, ...]:
    """Return the per-menu plot keys that activation will render by default."""
    return tuple(
        group.selected_key
        for group in menu_state.groups.values()
        if group.selected_key is not None
    )


def _cancelled(
    request: PlotPayloadWarmupRequest,
    cancel_token: PlotPayloadWarmupCancelToken,
) -> PlotPayloadWarmupCancelled | None:
    if not cancel_token.cancelled:
        return None
    return PlotPayloadWarmupCancelled(
        stream_key=request.stream_key,
        shank_idx=request.shank_idx,
        reason=cancel_token.reason or "cancelled",
    )
