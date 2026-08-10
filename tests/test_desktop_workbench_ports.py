"""Tests for MainWindow-to-Workbench port adaptation."""

from __future__ import annotations

from types import SimpleNamespace

from ephys_alignment_gui.desktop_workbench_ports import (
    desktop_workbench_ports_from_main_window,
)


class FakeCheckbox:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


def _fake_window_without_qc_widgets() -> SimpleNamespace:
    """Return enough of the offline MainWindow shape for port construction."""
    return SimpleNamespace(
        app=SimpleNamespace(
            queries=SimpleNamespace(
                workspace=SimpleNamespace(histology_data_loaded=lambda: True)
            )
        ),
        displays=SimpleNamespace(
            histology=SimpleNamespace(tip_position_um=lambda: 42.0)
        ),
        popup_manager=SimpleNamespace(close_all=lambda: None),
        shank_screen_view=SimpleNamespace(
            reset_raw_image_payloads=lambda: None,
            capture_plot_selection=lambda _preserve, displays: None,
            apply_plot_data_state=lambda _state: None,
            raw_image_payload_mapping=lambda: {},
            render_plot_menus=lambda _state, displays: None,
            configure_view_after_render=lambda _preserve: None,
        ),
        alignment_screen_view=SimpleNamespace(
            capture_depth_plot_y_ranges=lambda: None,
            restore_depth_plot_y_ranges=lambda _ranges: None,
        ),
        use_docdb_checkbox=FakeCheckbox(True),
        align_list=object(),
        align_combobox=object(),
        populate_lists=lambda _choices, _list, _combobox: None,
        _clear_empty_state=lambda: None,
        _show_empty_state=lambda: None,
        offline=True,
        complete_button=object(),
        _selected_qc_descriptions=lambda: [],
        reload_folder_line=SimpleNamespace(setText=lambda _text: None),
        reload_folder_button=object(),
        export_view=object(),
        fig_scale=object(),
        fig_hist=object(),
        fig_hist_ref=object(),
        fig_scale_ax=object(),
        bar_colour=object(),
        kpen_solid=object(),
        activateWindow=lambda: None,
        set_axis=lambda *_args, **_kwargs: None,
    )


def test_workbench_ports_do_not_require_online_qc_widgets() -> None:
    window = _fake_window_without_qc_widgets()

    ports = desktop_workbench_ports_from_main_window(window)

    assert ports.save.use_docdb()
    assert ports.save.histology_available()
    assert ports.save.ephys_qc() == "Pass"
    ports.save.open_qc_dialog()


def test_interaction_region_widgets_are_late_bound() -> None:
    window = _fake_window_without_qc_widgets()

    ports = desktop_workbench_ports_from_main_window(window)
    window.struct_list = object()
    window.struct_view = object()
    window.struct_description = object()

    assert ports.interaction.struct_list() is window.struct_list
    assert ports.interaction.struct_view() is window.struct_view
    assert ports.interaction.struct_description() is window.struct_description
