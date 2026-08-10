"""Desktop presenter for ephys plot menus and plot-spec dispatch."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.core.alignment_read_models import ActiveShankScreenState
from ephys_alignment_gui.plotting.menu_state import (
    EPHYS_PLOT_MENUS,
    PlotMenuGroupState,
    PlotMenuState,
)
from ephys_alignment_gui.plotting.registry import PlotMenu, PlotSpec, plot_spec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EphysPlotRenderCallbacks:
    """Desktop callbacks for rendering resolved ephys plot payloads."""

    raw_image_payloads: Callable[[], Mapping[Any, Any]]
    render_image: Callable[[Any], None]
    render_scatter: Callable[[Any], None]
    render_line: Callable[[Any], None]
    render_probe: Callable[[Any, Any], None]


@dataclass
class _PlotMenuHandles:
    menu: Any
    action_group: Any
    selected_action: Any = None
    initial_action: Any = None


@dataclass
class DesktopEphysPlotPresenter:
    """Own ephys plot QAction state and dispatch selected plots to renderers."""

    app: Any
    callbacks: EphysPlotRenderCallbacks
    action_factory: Callable[..., Any] = QtWidgets.QAction
    action_group_factory: Callable[..., Any] = QtWidgets.QActionGroup
    _groups: dict[PlotMenu, _PlotMenuHandles] = field(default_factory=dict)
    _plot_specs_by_key: dict[str, PlotSpec] = field(default_factory=dict)
    _unit_filter_actions_by_subset: dict[str, Any] = field(default_factory=dict)

    def attach_plot_menus(self, menu_bar: Any) -> None:
        """Create the ephys plot menus on a desktop menu bar."""
        self._groups = {
            "image": self._new_plot_menu_group(
                menu_bar.addMenu("Image Plots"),
            ),
            "line": self._new_plot_menu_group(
                menu_bar.addMenu("Line Plots"),
            ),
            "probe": self._new_plot_menu_group(
                menu_bar.addMenu("Probe Plots"),
            ),
        }

    def attach_unit_filter_menu(self, menu_bar: Any, parent: Any) -> None:
        """Create the ephys unit-filter menu on a desktop menu bar."""
        unit_filter_options = menu_bar.addMenu("Filter Units")
        unit_filter_group = self.action_group_factory(unit_filter_options)
        unit_filter_group.setExclusive(True)

        actions: dict[str, Any] = {}
        for subset, label, checked in (
            ("all", "All", True),
            ("KS good", "KS good", False),
            ("KS mua", "KS mua", False),
            ("IBL good", "IBL good", False),
            ("aind_qc", "aind_qc", False),
            ("unitrefine_sua", "unitrefine_sua", False),
            ("unitrefine_neural", "unitrefine_neural", False),
        ):
            action = self.action_factory(
                label,
                parent,
                checkable=True,
                checked=checked,
            )
            action.setData(subset)
            action.triggered.connect(
                lambda _checked=False, value=subset: self.filter_unit_pressed(value)
            )
            unit_filter_options.addAction(action)
            unit_filter_group.addAction(action)
            actions[subset] = action

        self._unit_filter_actions_by_subset = actions

    def has_plot_menus(self) -> bool:
        """Return whether the desktop plot menus have been attached."""
        return all(menu in self._groups for menu in EPHYS_PLOT_MENUS)

    def render_menus(self, plot_menu_state: PlotMenuState) -> None:
        """Render ephys plot menus from a Qt-free plot menu read model."""
        self._plot_specs_by_key = {}
        for menu in EPHYS_PLOT_MENUS:
            handles = self._groups.get(menu)
            if handles is None:
                logger.warning("Cannot render %s plots before menus exist", menu)
                continue
            self._render_plot_menu_group(
                handles=handles,
                state=plot_menu_state.group(menu),
            )

    def current_plot_keys(self) -> dict[PlotMenu, str | None]:
        """Return selected plot-spec keys for each ephys plot menu."""
        return {
            menu: self._plot_spec_key_from_action(self.checked_action(menu))
            for menu in EPHYS_PLOT_MENUS
        }

    def checked_action(self, menu: PlotMenu) -> Any:
        """Return the currently checked QAction for a plot menu, if any."""
        handles = self._groups.get(menu)
        if handles is None:
            return None
        checked_action = handles.action_group.checkedAction()
        return checked_action if checked_action is not None else handles.selected_action

    def toggle_plot(self, menu: PlotMenu, *, reverse: bool = False) -> None:
        """Toggle to the next available plot in one ephys plot menu."""
        handles = self._groups.get(menu)
        if handles is None:
            logger.warning("No available %s plot actions to toggle", menu)
            return
        current_action = handles.action_group.checkedAction()
        actions = handles.action_group.actions()
        if not actions:
            logger.warning("No available %s plot actions to toggle", menu)
            return
        if current_action is None:
            self._trigger_action(actions[0])
            return
        try:
            current_idx = actions.index(current_action)
        except ValueError:
            self._trigger_action(actions[0])
            return
        next_idx = (current_idx + (-1 if reverse else 1)) % len(actions)
        self._trigger_action(actions[next_idx])

    def render_shank_ephys_plots(self, state: ActiveShankScreenState) -> None:
        """Render the ephys plot selections after a shank-screen refresh."""
        logger.info("Rendering ephys plots...")
        if state.preserve_plot_selection:
            self.set_unit_filter_action_checked(state.unit_filter)
            for menu in EPHYS_PLOT_MENUS:
                action = self.checked_action(menu)
                if action is not None:
                    action.setChecked(True)
            return

        self.set_initial_actions_checked()
        self.set_unit_filter_action_checked("all")
        self.plot_default_spec("image")
        self.plot_default_spec("probe")
        self.plot_default_spec("line")

    def set_initial_actions_checked(self) -> None:
        """Check the initial ephys plot actions without triggering redraw."""
        for handles in self._groups.values():
            if handles.initial_action is not None:
                handles.initial_action.setChecked(True)

    def set_unit_filter_action_checked(self, unit_filter: str) -> None:
        """Reflect selected unit filter in the desktop menu."""
        action = self._unit_filter_actions_by_subset.get(unit_filter)
        if action is not None:
            action.setChecked(True)

    def filter_unit_pressed(self, unit_filter: str) -> None:
        """Apply a unit filter and redraw the currently selected ephys plots."""
        self.app.commands.edit.set_unit_filter(unit_filter)
        self.set_initial_actions_checked()
        self.set_unit_filter_action_checked(unit_filter)
        self.update_plot()

    def update_plot(self) -> None:
        """Re-run the plotting function for the current menu selections."""
        for menu in EPHYS_PLOT_MENUS:
            action = self.checked_action(menu)
            if action is not None:
                action.trigger()

    def plot_from_spec(self, spec_key: str) -> None:
        """Render a registered plot payload with the configured render callbacks."""
        spec = self.registered_plot_spec(spec_key)
        if spec is None:
            return
        data = self.plot_payload_for_spec(spec.key)
        if spec.renderer == "image":
            self.callbacks.render_image(data)
        elif spec.renderer == "scatter":
            self.callbacks.render_scatter(data)
        elif spec.renderer == "line":
            self.callbacks.render_line(data)
        elif spec.renderer == "probe":
            self.callbacks.render_probe(data, self.plot_bounds_for_spec(spec.key))
        else:
            raise ValueError(f"Unsupported plot renderer: {spec.renderer!r}")

    def plot_default_spec(self, menu: PlotMenu) -> None:
        """Render the available default plot for a menu group, if present."""
        specs = [spec for spec in self._plot_specs_by_key.values() if spec.menu == menu]
        if not specs:
            logger.warning("No available %s plot entries", menu)
            return
        for spec in specs:
            if spec.default:
                self.plot_from_spec(spec.key)
                return
        self.plot_from_spec(specs[0].key)

    def registered_plot_spec(self, spec_key: str) -> PlotSpec | None:
        """Return a dynamic menu spec if present, otherwise a static registry spec."""
        spec = self._plot_specs_by_key.get(spec_key)
        if spec is not None:
            return spec
        try:
            return plot_spec(spec_key)
        except KeyError:
            logger.warning("Ignoring unavailable plot spec %s", spec_key)
            return None

    def plot_payload_for_spec(self, spec_key: str) -> Any:
        """Resolve a registered plot payload for the active shank."""
        spec = self.registered_plot_spec(spec_key)
        if spec is None:
            return None
        return self.app.queries.ephys.active_plot_payload(
            spec.key,
            raw_image_payloads=self.callbacks.raw_image_payloads(),
        )

    def plot_bounds_for_spec(self, spec_key: str) -> Any:
        """Resolve optional plot bounds for the active shank."""
        spec = self.registered_plot_spec(spec_key)
        if spec is None:
            return None
        return self.app.queries.ephys.active_plot_bounds(
            spec.key,
            raw_image_payloads=self.callbacks.raw_image_payloads(),
        )

    def _new_plot_menu_group(self, menu: Any) -> _PlotMenuHandles:
        group = self.action_group_factory(menu)
        group.setExclusive(True)
        return _PlotMenuHandles(menu=menu, action_group=group)

    def _render_plot_menu_group(
        self,
        *,
        handles: _PlotMenuHandles,
        state: PlotMenuGroupState,
    ) -> None:
        handles.menu.clear()
        group = self.action_group_factory(handles.menu)
        group.setExclusive(True)
        handles.action_group = group
        group.triggered.connect(
            lambda selected_action, plot_menu=state.menu: self._set_selected_action(
                plot_menu,
                selected_action,
            )
        )

        selected_action = None
        for spec in state.specs:
            action = self._add_plot_spec_action(
                handles.menu,
                group,
                spec,
                checked=spec.key == state.selected_key,
            )
            if spec.key == state.selected_key:
                selected_action = action

        handles.menu.setEnabled(state.enabled)
        handles.initial_action = selected_action
        handles.selected_action = selected_action
        if selected_action is not None:
            selected_action.setChecked(True)

    def _add_plot_spec_action(
        self,
        menu: Any,
        group: Any,
        spec: PlotSpec,
        *,
        checked: bool = False,
    ) -> Any:
        self._plot_specs_by_key[spec.key] = spec
        action = self.action_factory(
            spec.label,
            menu,
            checkable=True,
            checked=checked,
        )
        action.setData({"plot_spec": spec.key})
        action.triggered.connect(
            lambda _checked=False, key=spec.key: self.plot_from_spec(key)
        )
        menu.addAction(action)
        group.addAction(action)
        return action

    def _set_selected_action(self, menu: PlotMenu, action: Any) -> None:
        handles = self._groups.get(menu)
        if handles is not None:
            handles.selected_action = action

    @staticmethod
    def _plot_spec_key_from_action(action: Any) -> str | None:
        if action is None:
            return None
        data = action.data()
        if not isinstance(data, dict):
            return None
        key = data.get("plot_spec")
        return key if isinstance(key, str) else None

    @staticmethod
    def _trigger_action(action: Any) -> None:
        action.setChecked(True)
        action.trigger()
