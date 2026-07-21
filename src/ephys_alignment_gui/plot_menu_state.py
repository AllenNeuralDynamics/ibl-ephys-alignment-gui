"""GUI-agnostic state for ephys plot menus."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.plot_registry import (
    PlotMenu,
    PlotSpec,
    available_plot_specs_for_menu,
    mapping_plot_specs,
)

EPHYS_PLOT_MENUS: tuple[PlotMenu, ...] = ("image", "line", "probe")


@dataclass(frozen=True)
class PlotMenuGroupState:
    """Available plot entries and selected key for one plot menu."""

    menu: PlotMenu
    specs: tuple[PlotSpec, ...]
    selected_key: str | None

    @property
    def enabled(self) -> bool:
        return bool(self.specs)

    @property
    def selected_spec(self) -> PlotSpec | None:
        if self.selected_key is None:
            return None
        return next(
            (spec for spec in self.specs if spec.key == self.selected_key),
            None,
        )


@dataclass(frozen=True)
class PlotMenuState:
    """Available ephys plot menu state for the active shank."""

    groups: dict[PlotMenu, PlotMenuGroupState]

    def group(self, menu: PlotMenu) -> PlotMenuGroupState:
        return self.groups[menu]

    @property
    def specs(self) -> tuple[PlotSpec, ...]:
        return tuple(
            spec
            for menu in EPHYS_PLOT_MENUS
            for spec in self.groups[menu].specs
        )


def choose_plot_key(
    specs: tuple[PlotSpec, ...],
    previous_key: str | None = None,
) -> str | None:
    """Choose a menu selection from available specs and a previous key."""
    if not specs:
        return None
    if previous_key is not None and any(spec.key == previous_key for spec in specs):
        return previous_key
    for spec in specs:
        if spec.default:
            return spec.key
    return specs[0].key


def build_plot_menu_state(
    plotdata: Any,
    *,
    previous_selected_keys: Mapping[PlotMenu, str | None] | None = None,
    raw_image_payloads: Mapping[Any, Any] | None = None,
) -> PlotMenuState:
    """Build current-shank plot menu availability and selection state."""
    previous_selected_keys = previous_selected_keys or {}
    raw_image_payloads = raw_image_payloads or {}

    groups: dict[PlotMenu, PlotMenuGroupState] = {}
    for menu in EPHYS_PLOT_MENUS:
        specs = list(available_plot_specs_for_menu(plotdata, menu))
        if menu == "image":
            specs.extend(
                mapping_plot_specs(
                    parent_key="image.raw",
                    menu="image",
                    renderer="image",
                    payloads=raw_image_payloads,
                )
            )
        available_specs = tuple(specs)
        groups[menu] = PlotMenuGroupState(
            menu=menu,
            specs=available_specs,
            selected_key=choose_plot_key(
                available_specs,
                previous_selected_keys.get(menu),
            ),
        )
    return PlotMenuState(groups)
