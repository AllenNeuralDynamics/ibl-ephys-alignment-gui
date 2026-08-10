"""Port DTOs used to compose the desktop Workbench."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop_export_view import DesktopExportView


@dataclass(frozen=True)
class DesktopAlignmentRenderPorts:
    """Desktop operations needed to render alignment edits."""

    capture_depth_plot_y_ranges: Callable[[], Any]
    restore_depth_plot_y_ranges: Callable[[Any], None]


@dataclass(frozen=True)
class DesktopAlignmentEditActionPorts:
    """Desktop state needed to start alignment edit commands."""

    histology_available: Callable[[], bool]
    tip_position_um: Callable[[], float | None]


@dataclass(frozen=True)
class DesktopShankRenderPorts:
    """Desktop operations needed to render an active shank."""

    capture_plot_selection: Callable[[bool], Any]
    render_alignment_choices: Callable[[list[str]], None]
    apply_plot_data_state: Callable[[Any], None]
    raw_image_payloads: Callable[[], Any]
    render_plot_menus: Callable[[Any], None]
    configure_view: Callable[[bool], None]
    offline: Callable[[], bool]


@dataclass(frozen=True)
class DesktopRenderPorts:
    """MainWindow render ports consumed by focused desktop presenters."""

    alignment: DesktopAlignmentRenderPorts
    shank: DesktopShankRenderPorts


@dataclass(frozen=True)
class DesktopBusyPorts:
    """Desktop busy-state operations shared by command presenters."""

    busy_context: Callable[..., AbstractContextManager[Any]]


@dataclass(frozen=True)
class DesktopLoadDataPorts:
    """Desktop operations needed by heavy data load presentation."""

    clear_empty_state: Callable[[], None]


@dataclass(frozen=True)
class DesktopSavePorts:
    """Desktop operations needed by save and QC presentation."""

    use_docdb: Callable[[], bool]
    render_alignment_choices: Callable[[list[str]], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    complete_button: Callable[[], Any]
    histology_available: Callable[[], bool]
    open_qc_dialog: Callable[[], None]
    ephys_qc: Callable[[], str]
    selected_qc_descriptions: Callable[[], list[str]]
    warning: Callable[[str, str], Any]


@dataclass(frozen=True)
class DesktopPreviousAlignmentLoadPorts:
    """Desktop operations needed by previous-alignment loading."""

    use_docdb: Callable[[], bool]
    set_reload_folder_text: Callable[[str], None]
    render_alignment_choices: Callable[[list[str]], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    reload_button: Callable[[], Any]


@dataclass(frozen=True)
class DesktopInteractionPorts:
    """Desktop operations and handles needed by interaction presentation."""

    popup_manager: Any
    struct_list: Callable[[], Any]
    struct_view: Callable[[], Any]
    struct_description: Callable[[], Any]
    scale_plot: Any
    histology_plot: Any
    histology_reference_plot: Any
    scale_axis: Any
    bar_colour: Any
    line_pen: Any
    histology_available: Callable[[], bool]
    activate_window: Callable[[], None]
    set_axis: Callable[..., Any]


@dataclass(frozen=True)
class DesktopLifecyclePorts:
    """Desktop-only operations for stream/session lifecycle presentation."""

    close_popups: Callable[[], None]
    reset_raw_image_payloads: Callable[[], None]
    show_empty_state: Callable[[], None]
    collect_garbage: Callable[[], None]


@dataclass(frozen=True)
class DesktopWorkbenchPorts:
    """MainWindow ports consumed by Workbench presenter composition."""

    render: DesktopRenderPorts
    alignment_edit_actions: DesktopAlignmentEditActionPorts
    busy: DesktopBusyPorts
    load_data: DesktopLoadDataPorts
    lifecycle: DesktopLifecyclePorts
    save: DesktopSavePorts
    previous_alignment_load: DesktopPreviousAlignmentLoadPorts
    export: DesktopExportView
    interaction: DesktopInteractionPorts
