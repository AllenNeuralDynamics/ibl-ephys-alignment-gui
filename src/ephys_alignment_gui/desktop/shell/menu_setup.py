"""Desktop menu construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt5 import QtWidgets


def build_menu_bar(window: Any) -> None:
    """Create the desktop menu bar and attach plot/action menu groups."""
    menu_bar = QtWidgets.QMenuBar(window)
    menu_bar.setNativeMenuBar(False)
    window.setMenuBar(menu_bar)

    _attach_plot_menus(window, menu_bar)
    _add_fit_options_menu(window, menu_bar)
    _add_display_options_menu(window, menu_bar)
    _add_session_information_menu(window, menu_bar)


def _attach_plot_menus(window: Any, menu_bar: QtWidgets.QMenuBar) -> None:
    render_cluster = window.desktop_workbench.render_cluster
    render_cluster.ephys_plot_presenter.attach_plot_menus(menu_bar)
    render_cluster.slice_menu_coordinator.attach_menu(
        menu_bar,
        parent=window,
        offline=window.offline,
    )
    render_cluster.ephys_plot_presenter.attach_unit_filter_menu(menu_bar, window)


def _add_fit_options_menu(window: Any, menu_bar: QtWidgets.QMenuBar) -> None:
    actions = window.shell_actions
    fit_options = menu_bar.addMenu("Fit Options")
    _add_actions(
        fit_options,
        [
            _action(window, "Fit", "Return", actions.fit_button_pressed),
            _action(window, "Offset", "O", actions.offset_button_pressed),
            _action(window, "Offset + 50um", "Shift+Up", actions.moveup_button_pressed),
            _action(
                window,
                "Offset - 50um",
                "Shift+Down",
                actions.movedown_button_pressed,
            ),
            _action(
                window,
                "Delete Line",
                "Shift+D",
                actions.delete_line_button_pressed,
            ),
            _action(window, "Next", "Right", actions.next_button_pressed),
            _action(window, "Previous", "Left", actions.prev_button_pressed),
            _action(window, "Reset", "Ctrl+R", actions.reset_button_pressed),
            _save_action(window),
        ],
    )


def _add_display_options_menu(window: Any, menu_bar: QtWidgets.QMenuBar) -> None:
    actions = window.shell_actions
    display_options = menu_bar.addMenu("Display Options")
    _add_actions(
        display_options,
        [
            _action(
                window,
                "Toggle Image Plots",
                "Alt+1",
                lambda: window.desktop_workbench.render_cluster.ephys_plot_presenter.toggle_plot(
                    "image"
                ),
            ),
            _action(
                window,
                "Toggle Line Plots",
                "Alt+2",
                lambda: window.desktop_workbench.render_cluster.ephys_plot_presenter.toggle_plot(
                    "line"
                ),
            ),
            _action(
                window,
                "Toggle Probe Plots",
                "Alt+3",
                lambda: window.desktop_workbench.render_cluster.ephys_plot_presenter.toggle_plot(
                    "probe"
                ),
            ),
            _action(
                window,
                "Toggle Slice Plots",
                "Alt+4",
                lambda: window.desktop_workbench.render_cluster.slice_menu_coordinator.toggle_plot(),
            ),
            _action(
                window,
                "Toggle Previous Image Plots",
                "Alt+Ctrl+1",
                lambda: window.desktop_workbench.render_cluster.ephys_plot_presenter.toggle_plot(
                    "image",
                    reverse=True,
                ),
            ),
            _action(
                window,
                "Toggle Previous Line Plots",
                "Alt+Ctrl+2",
                lambda: window.desktop_workbench.render_cluster.ephys_plot_presenter.toggle_plot(
                    "line",
                    reverse=True,
                ),
            ),
            _action(
                window,
                "Toggle Previous Probe Plots",
                "Alt+Ctrl+3",
                lambda: window.desktop_workbench.render_cluster.ephys_plot_presenter.toggle_plot(
                    "probe",
                    reverse=True,
                ),
            ),
            _action(
                window,
                "Toggle Previous Slice Plots",
                "Alt+Ctrl+4",
                lambda: window.desktop_workbench.render_cluster.slice_menu_coordinator.toggle_plot(
                    reverse=True
                ),
            ),
            _action(
                window,
                "View 1",
                "Shift+1",
                lambda: window.shank_screen_view.set_view(view=1),
            ),
            _action(
                window,
                "View 2",
                "Shift+2",
                lambda: window.shank_screen_view.set_view(view=2),
            ),
            _action(
                window,
                "View 3",
                "Shift+3",
                lambda: window.shank_screen_view.set_view(view=3),
            ),
            _action(window, "Reset Axis", "Shift+A", actions.reset_axis_button_pressed),
            _action(
                window,
                "Hide/Show Labels",
                "Shift+L",
                actions.toggle_labels_button_pressed,
            ),
            _action(
                window,
                "Hide/Show Lines",
                "Shift+H",
                actions.toggle_line_button_pressed,
            ),
            _action(
                window,
                "Hide/Show Channels",
                "Shift+C",
                actions.toggle_channel_button_pressed,
            ),
            _action(
                window,
                "Hide/Show Nearby Boundaries",
                "Shift+N",
                actions.toggle_histology_button_pressed,
            ),
            _action(
                window,
                "Change Histology Map",
                "Shift+M",
                actions.toggle_histology_map_button_pressed,
            ),
            _action(
                window,
                "Minimise/Show Cluster Popup",
                "Alt+M",
                actions.minimise_popups,
            ),
            _action(window, "Close Cluster Popup", "Alt+X", actions.close_popups),
            _action(window, "Save Plots", "Ctrl+Shift+S", actions.save_plots),
        ],
    )


def _add_session_information_menu(window: Any, menu_bar: QtWidgets.QMenuBar) -> None:
    actions = window.shell_actions
    info_options = menu_bar.addMenu("Session Information")
    _add_actions(
        info_options,
        [
            _action(window, "Session Notes", None, actions.display_session_notes),
            _action(window, "Region Info", "Shift+I", actions.describe_labels_pressed),
        ],
    )

    if not window.offline:
        info_options.addAction(
            _action(window, "Nearby Sessions", None, actions.display_nearby_sessions)
        )


def _save_action(window: Any) -> QtWidgets.QAction:
    actions = window.shell_actions
    callback = (
        actions.display_qc_options
        if not window.offline
        else actions.complete_button_pressed_offline
    )
    return _action(window, "Save", "Ctrl+S", callback)


def _action(
    window: Any,
    text: str,
    shortcut: str | None,
    callback: Callable[..., Any],
) -> QtWidgets.QAction:
    action = QtWidgets.QAction(text, window)
    if shortcut is not None:
        action.setShortcut(shortcut)
    action.triggered.connect(callback)
    return action


def _add_actions(menu: QtWidgets.QMenu, actions: list[QtWidgets.QAction]) -> None:
    for action in actions:
        menu.addAction(action)
