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
    window.displays.ephys.attach_plot_menus(menu_bar)
    window.displays.slice.attach_slice_menu(
        menu_bar,
        parent=window,
        offline=window.offline,
    )
    window.displays.ephys.attach_unit_filter_menu(menu_bar, window)


def _add_fit_options_menu(window: Any, menu_bar: QtWidgets.QMenuBar) -> None:
    fit_options = menu_bar.addMenu("Fit Options")
    _add_actions(
        fit_options,
        [
            _action(window, "Fit", "Return", window.fit_button_pressed),
            _action(window, "Offset", "O", window.offset_button_pressed),
            _action(window, "Offset + 50um", "Shift+Up", window.moveup_button_pressed),
            _action(
                window,
                "Offset - 50um",
                "Shift+Down",
                window.movedown_button_pressed,
            ),
            _action(
                window,
                "Delete Line",
                "Shift+D",
                window.delete_line_button_pressed,
            ),
            _action(window, "Next", "Right", window.next_button_pressed),
            _action(window, "Previous", "Left", window.prev_button_pressed),
            _action(window, "Reset", "Ctrl+R", window.reset_button_pressed),
            _save_action(window),
        ],
    )


def _add_display_options_menu(window: Any, menu_bar: QtWidgets.QMenuBar) -> None:
    display_options = menu_bar.addMenu("Display Options")
    _add_actions(
        display_options,
        [
            _action(
                window,
                "Toggle Image Plots",
                "Alt+1",
                lambda: window.displays.ephys.toggle_plot("image"),
            ),
            _action(
                window,
                "Toggle Line Plots",
                "Alt+2",
                lambda: window.displays.ephys.toggle_plot("line"),
            ),
            _action(
                window,
                "Toggle Probe Plots",
                "Alt+3",
                lambda: window.displays.ephys.toggle_plot("probe"),
            ),
            _action(
                window,
                "Toggle Slice Plots",
                "Alt+4",
                lambda: window.displays.slice.toggle_slice_plot(),
            ),
            _action(
                window,
                "Toggle Previous Image Plots",
                "Alt+Ctrl+1",
                lambda: window.displays.ephys.toggle_plot("image", reverse=True),
            ),
            _action(
                window,
                "Toggle Previous Line Plots",
                "Alt+Ctrl+2",
                lambda: window.displays.ephys.toggle_plot("line", reverse=True),
            ),
            _action(
                window,
                "Toggle Previous Probe Plots",
                "Alt+Ctrl+3",
                lambda: window.displays.ephys.toggle_plot("probe", reverse=True),
            ),
            _action(
                window,
                "Toggle Previous Slice Plots",
                "Alt+Ctrl+4",
                lambda: window.displays.slice.toggle_slice_plot(reverse=True),
            ),
            _action(window, "View 1", "Shift+1", lambda: window.set_view(view=1)),
            _action(window, "View 2", "Shift+2", lambda: window.set_view(view=2)),
            _action(window, "View 3", "Shift+3", lambda: window.set_view(view=3)),
            _action(window, "Reset Axis", "Shift+A", window.reset_axis_button_pressed),
            _action(
                window,
                "Hide/Show Labels",
                "Shift+L",
                window.toggle_labels_button_pressed,
            ),
            _action(
                window,
                "Hide/Show Lines",
                "Shift+H",
                window.toggle_line_button_pressed,
            ),
            _action(
                window,
                "Hide/Show Channels",
                "Shift+C",
                window.toggle_channel_button_pressed,
            ),
            _action(
                window,
                "Hide/Show Nearby Boundaries",
                "Shift+N",
                window.toggle_histology_button_pressed,
            ),
            _action(
                window,
                "Change Histology Map",
                "Shift+M",
                window.toggle_histology_map_button_pressed,
            ),
            _action(
                window,
                "Minimise/Show Cluster Popup",
                "Alt+M",
                window.minimise_popups,
            ),
            _action(window, "Close Cluster Popup", "Alt+X", window.close_popups),
            _action(window, "Save Plots", "Ctrl+Shift+S", window.save_plots),
        ],
    )


def _add_session_information_menu(window: Any, menu_bar: QtWidgets.QMenuBar) -> None:
    info_options = menu_bar.addMenu("Session Information")
    _add_actions(
        info_options,
        [
            _action(window, "Session Notes", None, window.display_session_notes),
            _action(window, "Region Info", "Shift+I", window.describe_labels_pressed),
        ],
    )

    if not window.offline:
        info_options.addAction(
            _action(window, "Nearby Sessions", None, window.display_nearby_sessions)
        )


def _save_action(window: Any) -> QtWidgets.QAction:
    callback = (
        window.display_qc_options
        if not window.offline
        else window.complete_button_pressed_offline
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
