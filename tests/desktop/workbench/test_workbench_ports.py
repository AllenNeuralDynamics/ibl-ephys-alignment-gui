"""Tests for desktop shell handle-to-Workbench port adaptation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from ephys_alignment_gui.desktop.workbench.ports import (
    DesktopWorkbenchPortHandles,
    desktop_workbench_ports_from_handles,
)


class FakeCheckbox:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


def _fake_handles_without_qc_widgets() -> DesktopWorkbenchPortHandles:
    """Return enough offline shell handles for port construction."""
    region_handles = SimpleNamespace(
        struct_list=None,
        struct_view=None,
        struct_description=None,
    )
    displays = SimpleNamespace(
        ephys=SimpleNamespace(
            clear_empty_state=lambda: None,
            show_empty_state=lambda: None,
        ),
        histology=SimpleNamespace(
            tip_position_um=lambda: 42.0,
            scale_plot=object(),
            aligned_plot=object(),
            reference_plot=object(),
            scale_axis=object(),
        ),
    )
    views = SimpleNamespace(
        shank_screen=SimpleNamespace(
            reset_raw_image_payloads=lambda: None,
            capture_plot_selection=lambda _preserve, **_kwargs: None,
            apply_plot_data_state=lambda _state: None,
            raw_image_payload_mapping=lambda: {},
            render_plot_menus=lambda _state, **_kwargs: None,
            configure_view_after_render=lambda _preserve: None,
        ),
        alignment_screen=SimpleNamespace(
            capture_depth_plot_y_ranges=lambda: None,
            restore_depth_plot_y_ranges=lambda _ranges: None,
            render_alignment_choices=lambda _choices: None,
        ),
    )
    return DesktopWorkbenchPortHandles(
        app=SimpleNamespace(
            queries=SimpleNamespace(
                workspace=SimpleNamespace(histology_data_loaded=lambda: True)
            )
        ),
        parent=object(),
        displays=displays,
        views=views,
        popup_manager=SimpleNamespace(close_all=lambda: None),
        shell_actions=SimpleNamespace(selected_qc_descriptions=lambda: []),
        use_docdb_checkbox=FakeCheckbox(True),
        complete_button=object(),
        reload_folder_line=SimpleNamespace(setText=lambda _text: None),
        reload_folder_button=object(),
        export_view=object(),
        offline=lambda: True,
        qc_dialog=lambda: None,
        ephys_qc=lambda: None,
        struct_list=lambda: region_handles.struct_list,
        struct_view=lambda: region_handles.struct_view,
        struct_description=lambda: region_handles.struct_description,
        activate_window=lambda: None,
        bar_colour=object(),
        solid_pen=object(),
    )


def test_workbench_ports_do_not_require_online_qc_widgets() -> None:
    handles = _fake_handles_without_qc_widgets()

    ports = desktop_workbench_ports_from_handles(handles)

    assert ports.save.use_docdb()
    assert ports.save.histology_available()
    assert ports.save.ephys_qc() == "Pass"
    ports.save.open_qc_dialog()


def test_interaction_region_widgets_are_late_bound() -> None:
    region_handles = SimpleNamespace(
        struct_list=None,
        struct_view=None,
        struct_description=None,
    )
    handles = _fake_handles_without_qc_widgets()
    handles = replace(
        handles,
        struct_list=lambda: region_handles.struct_list,
        struct_view=lambda: region_handles.struct_view,
        struct_description=lambda: region_handles.struct_description,
    )

    ports = desktop_workbench_ports_from_handles(handles)
    region_handles.struct_list = object()
    region_handles.struct_view = object()
    region_handles.struct_description = object()

    assert ports.interaction.struct_list() is region_handles.struct_list
    assert ports.interaction.struct_view() is region_handles.struct_view
    assert ports.interaction.struct_description() is region_handles.struct_description
