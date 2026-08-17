"""Build desktop Workbench ports from explicit shell handles."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.core.document import AlignmentKey
from ephys_alignment_gui.desktop.displays.axis_style import set_axis
from ephys_alignment_gui.desktop.shell.busy_context import BusyContext
from ephys_alignment_gui.desktop.views.save_progress_dialog import (
    DesktopSaveProgressDialog,
)
from ephys_alignment_gui.desktop.workbench.port_types import (
    DesktopAlignmentEditActionPorts,
    DesktopAlignmentRenderPorts,
    DesktopBusyPorts,
    DesktopInteractionPorts,
    DesktopLifecyclePorts,
    DesktopLoadDataPorts,
    DesktopPreviousAlignmentLoadPorts,
    DesktopRenderPorts,
    DesktopSavePorts,
    DesktopShankRenderPorts,
    DesktopWorkbenchPorts,
)


@dataclass(frozen=True)
class DesktopWorkbenchPortHandles:
    """Shell handles needed to build Workbench port DTOs."""

    app: Any
    parent: Any
    displays: Any
    views: Any
    popup_manager: Any
    shell_actions: Any
    use_docdb_checkbox: Any
    complete_button: Any
    reload_folder_line: Any
    reload_folder_button: Any
    export_view: Any
    offline: Callable[[], bool]
    qc_dialog: Callable[[], Any | None]
    ephys_qc: Callable[[], Any | None]
    struct_list: Callable[[], Any]
    struct_view: Callable[[], Any]
    struct_description: Callable[[], Any]
    activate_window: Callable[[], None]
    bar_colour: Any
    solid_pen: Any


def desktop_workbench_ports_from_handles(
    handles: DesktopWorkbenchPortHandles,
) -> DesktopWorkbenchPorts:
    """Adapt explicit shell handles to Workbench ports."""
    app = handles.app
    displays = handles.displays
    views = handles.views
    parent = handles.parent

    def busy_context(*args: Any, **kwargs: Any) -> BusyContext:
        return BusyContext(parent, *args, **kwargs)

    def open_qc_dialog() -> None:
        if qc_dialog := handles.qc_dialog():
            qc_dialog.open()

    def ephys_qc() -> str:
        if qc_widget := handles.ephys_qc():
            return qc_widget.currentText()
        return "Pass"

    def histology_available() -> bool:
        return app.queries.workspace.histology_data_loaded()

    def use_docdb() -> bool:
        return handles.use_docdb_checkbox.isChecked()

    def unvisited_alignment_targets() -> tuple[AlignmentKey, ...]:
        return getattr(
            app.queries.workspace,
            "unvisited_alignment_targets",
            lambda: (),
        )()

    def confirm_incomplete_alignment_save(
        unvisited_targets: tuple[AlignmentKey, ...],
    ) -> bool:
        box = QtWidgets.QMessageBox(parent)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Incomplete Alignment Set")
        box.setText("Some probe/shank targets have not been visited.")
        box.setInformativeText(
            "Save only the visited shanks? Choose OK only if you intentionally "
            "want an incomplete annotation package.\n\n"
            f"{_format_unvisited_alignment_targets(unvisited_targets, limit=12)}"
        )
        box.setDetailedText(_format_unvisited_alignment_targets(unvisited_targets))
        box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        return box.exec_() == QtWidgets.QMessageBox.Ok

    return DesktopWorkbenchPorts(
        alignment_edit_actions=DesktopAlignmentEditActionPorts(
            histology_available=histology_available,
            tip_position_um=displays.histology.tip_position_um,
        ),
        busy=DesktopBusyPorts(busy_context=busy_context),
        load_data=DesktopLoadDataPorts(
            clear_empty_state=displays.ephys.clear_empty_state,
        ),
        lifecycle=DesktopLifecyclePorts(
            close_popups=handles.popup_manager.close_all,
            reset_raw_image_payloads=views.shank_screen.reset_raw_image_payloads,
            show_empty_state=displays.ephys.show_empty_state,
            collect_garbage=gc.collect,
        ),
        render=DesktopRenderPorts(
            alignment=DesktopAlignmentRenderPorts(
                capture_depth_plot_y_ranges=(
                    views.alignment_screen.capture_depth_plot_y_ranges
                ),
                restore_depth_plot_y_ranges=(
                    views.alignment_screen.restore_depth_plot_y_ranges
                ),
            ),
            shank=DesktopShankRenderPorts(
                capture_plot_selection=lambda preserve, ephys_plot_presenter, slice_menu_coordinator: (
                    views.shank_screen.capture_plot_selection(
                        preserve,
                        ephys_plot_presenter=ephys_plot_presenter,
                        slice_menu_coordinator=slice_menu_coordinator,
                    )
                ),
                render_alignment_choices=(
                    views.alignment_screen.render_alignment_choices
                ),
                apply_plot_data_state=views.shank_screen.apply_plot_data_state,
                raw_image_payloads=views.shank_screen.raw_image_payload_mapping,
                render_plot_menus=lambda state, ephys_plot_presenter: (
                    views.shank_screen.render_plot_menus(
                        state,
                        ephys_plot_presenter=ephys_plot_presenter,
                    )
                ),
                configure_view=(views.shank_screen.configure_view_after_render),
                offline=handles.offline,
            ),
        ),
        save=DesktopSavePorts(
            use_docdb=use_docdb,
            render_alignment_choices=(views.alignment_screen.render_alignment_choices),
            busy_context=busy_context,
            complete_button=lambda: handles.complete_button,
            save_progress_dialog=lambda: DesktopSaveProgressDialog(parent),
            histology_available=histology_available,
            open_qc_dialog=open_qc_dialog,
            ephys_qc=ephys_qc,
            selected_qc_descriptions=handles.shell_actions.selected_qc_descriptions,
            warning=lambda title, message: QtWidgets.QMessageBox.warning(
                parent,
                title,
                message,
            ),
            unvisited_alignment_targets=unvisited_alignment_targets,
            confirm_incomplete_alignment_save=confirm_incomplete_alignment_save,
            save_blocking_widgets=lambda: (
                [central_widget]
                if (central_widget := parent.centralWidget()) is not None
                else []
            ),
        ),
        previous_alignment_load=DesktopPreviousAlignmentLoadPorts(
            use_docdb=use_docdb,
            set_reload_folder_text=handles.reload_folder_line.setText,
            render_alignment_choices=(views.alignment_screen.render_alignment_choices),
            busy_context=busy_context,
            reload_button=lambda: handles.reload_folder_button,
        ),
        export=handles.export_view,
        interaction=DesktopInteractionPorts(
            popup_manager=handles.popup_manager,
            struct_list=handles.struct_list,
            struct_view=handles.struct_view,
            struct_description=handles.struct_description,
            scale_plot=displays.histology.scale_plot,
            histology_plot=displays.histology.aligned_plot,
            histology_reference_plot=displays.histology.reference_plot,
            scale_axis=displays.histology.scale_axis,
            bar_colour=handles.bar_colour,
            line_pen=handles.solid_pen,
            histology_available=histology_available,
            activate_window=handles.activate_window,
            set_axis=set_axis,
        ),
    )


def _format_unvisited_alignment_targets(
    targets: tuple[AlignmentKey, ...],
    *,
    limit: int | None = None,
) -> str:
    """Return a human-readable list of unvisited alignment targets."""
    visible = targets if limit is None else targets[:limit]
    lines = [
        f"- {key.recording_id} / {key.ephys_collection} / shank {key.shank_idx + 1}"
        for key in visible
    ]
    if limit is not None and len(targets) > limit:
        lines.append(f"- ... and {len(targets) - limit} more")
    return "\n".join(lines)
