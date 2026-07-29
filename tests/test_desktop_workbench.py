"""Tests for desktop workbench presenter composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop_displays import DesktopDisplays
from ephys_alignment_gui.desktop_shank_presenter import DesktopShankSelectionState
from ephys_alignment_gui.desktop_workbench import (
    DesktopAlignmentRenderPorts,
    DesktopExportPorts,
    DesktopInteractionPorts,
    DesktopLifecyclePorts,
    DesktopPreviousAlignmentLoadPorts,
    DesktopRenderPorts,
    DesktopSaveWorkflowPorts,
    DesktopSelectionWorkflowCallbacks,
    DesktopShankRenderPorts,
    DesktopWorkbench,
    DesktopWorkbenchPorts,
)
from ephys_alignment_gui.event_bus import EventBus
from ephys_alignment_gui.workflow import Ok


class FakeSubscription:
    def __init__(self) -> None:
        self.disconnect_count = 0

    def disconnect(self) -> None:
        self.disconnect_count += 1


class FakeAlignmentPresenter:
    def __init__(self, subscriptions: list[FakeSubscription]) -> None:
        self.subscriptions = subscriptions
        self.connect_count = 0

    def connect_alignment_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions


class FakeShankPresenter:
    def __init__(self, subscriptions: list[FakeSubscription]) -> None:
        self.subscriptions = subscriptions
        self.connect_count = 0
        self.render_calls: list[tuple[int, bool | None]] = []

    def connect_shank_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions

    def render_loaded_shank(
        self,
        *,
        shank_idx: int,
        preserve_plot_selection: bool | None = None,
    ) -> None:
        self.render_calls.append((shank_idx, preserve_plot_selection))


class FakeHistologyDisplay:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def render_active_aligned(
        self,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> bool:
        self.calls.append(("aligned", fig, movable))
        return True

    def render_active_reference(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        self.calls.append(("reference", fig, movable))
        return True

    def render_active_scale_factor(self) -> bool:
        self.calls.append("scale")
        return True

    def render_active_fit(self) -> bool:
        self.calls.append("fit")
        return True

    def render_active_panels(self) -> bool:
        self.calls.append("panels")
        return True

    def render_alignment_edit(self, render_state: Any) -> bool:
        self.calls.append(("edit", render_state))
        return True


class FakeLoadDataPresenter:
    def __init__(self) -> None:
        self.load_count = 0

    def load_heavy_data(self) -> bool:
        self.load_count += 1
        return True


class FakeMouseRootPresenter:
    def __init__(self) -> None:
        self.set_roots: list[Any] = []
        self.edited_count = 0

    def set_mouse_root(self, mouse_root: Any) -> bool:
        self.set_roots.append(mouse_root)
        return True

    def mouse_root_edited(self) -> bool:
        self.edited_count += 1
        return True


class FakeSessionSelectionPresenter:
    def __init__(self) -> None:
        self.selected_count = 0

    def session_selected(self) -> bool:
        self.selected_count += 1
        return True


class FakeProbeSelectionPresenter:
    def __init__(self) -> None:
        self.selected_count = 0

    def probe_selected(self) -> bool:
        self.selected_count += 1
        return True


class FakeOutputPathPresenter:
    def __init__(self) -> None:
        self.save_roots: list[Any] = []
        self.edited_count = 0

    def set_save_root(self, save_root: Any) -> bool:
        self.save_roots.append(save_root)
        return True

    def output_folder_edited(self) -> bool:
        self.edited_count += 1
        return True


class FakePathDialogPresenter:
    def __init__(self) -> None:
        self.mouse_root_count = 0
        self.output_root_count = 0

    def select_mouse_root(self) -> bool:
        self.mouse_root_count += 1
        return True

    def select_output_root(self) -> bool:
        self.output_root_count += 1
        return True


class FakeLoadWorkflowPresenter:
    def __init__(self) -> None:
        self.load_count = 0
        self.logged: list[Any] = []

    def load_data_button_pressed(self) -> bool:
        self.load_count += 1
        return True

    def log_requirement(self, requirement: Any) -> None:
        self.logged.append(requirement)


class FakeOutputFolderPrompt:
    def __init__(self) -> None:
        self.requirements: list[Any] = []

    def ensure_for_save(self, requirement: Any | None = None) -> bool:
        self.requirements.append(requirement)
        return True


class FakeFolderDialog:
    def __init__(self) -> None:
        self.titles: list[str] = []

    def select_existing_directory_text(self, title: str) -> str:
        self.titles.append(title)
        return "/selected"


class FakeSaveWorkflowPresenter:
    def __init__(self) -> None:
        self.saved_count = 0
        self.qc_display_count = 0
        self.qc_clicked_count = 0

    def save_alignment_outputs(self) -> bool:
        self.saved_count += 1
        return True

    def display_qc_options(self) -> bool:
        self.qc_display_count += 1
        return True

    def qc_button_clicked(self) -> bool:
        self.qc_clicked_count += 1
        return True


class FakePreviousAlignmentLoadPresenter:
    def __init__(self) -> None:
        self.load_count = 0

    def load_existing_alignments(self) -> bool:
        self.load_count += 1
        return True


class FakePlotExporter:
    def __init__(self) -> None:
        self.exports: list[tuple[Any, str]] = []

    def export(self, output_dir: Any, *, sess_info: str = "") -> None:
        self.exports.append((output_dir, sess_info))


class FakeEphysDisplay:
    def __init__(self) -> None:
        self.panel = object()
        self.plot_presenter = object()
        self.rendered_states: list[Any] = []

    def render_shank_ephys_plots(self, state: Any) -> None:
        self.rendered_states.append(state)


class FakeSliceDisplay:
    def __init__(self) -> None:
        self.restored: list[tuple[Any, Any, Any]] = []
        self.plotted_channels: list[Any] = []
        self.perpendicular_refreshes = 0

    def restore_selection(
        self,
        slice_menu_state: Any,
        previous_selection: Any,
        previous_label: Any,
    ) -> None:
        self.restored.append((slice_menu_state, previous_selection, previous_label))

    def plot_channels(self, projection: Any = None) -> None:
        self.plotted_channels.append(projection)

    def refresh_perpendicular_histology(self) -> None:
        self.perpendicular_refreshes += 1


class FakeReferenceLineDisplay:
    def __init__(self) -> None:
        self.clear_count = 0
        self.reattach_count = 0
        self.sync_count = 0
        self.add_count = 0

    def clear(self) -> None:
        self.clear_count += 1

    def reattach(self) -> None:
        self.reattach_count += 1

    def sync_track_to_feature(self) -> None:
        self.sync_count += 1

    def add_to_plots(self) -> None:
        self.add_count += 1


def _displays(
    *,
    ephys: Any | None = None,
    histology: Any | None = None,
    reference_lines: Any | None = None,
    slice_display: Any | None = None,
) -> DesktopDisplays:
    return DesktopDisplays(
        ephys=ephys or FakeEphysDisplay(),
        histology=histology or FakeHistologyDisplay(),
        reference_lines=reference_lines or FakeReferenceLineDisplay(),
        slice=slice_display or FakeSliceDisplay(),
    )


class FakeInteractionPresenter:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def display_session_notes(self) -> None:
        self.calls.append("notes")

    def popup_closed(self, popup: Any) -> None:
        self.calls.append(("popup-closed", popup))

    def popup_moved(self) -> None:
        self.calls.append("popup-moved")

    def close_popups(self) -> None:
        self.calls.append("close-popups")

    def minimise_popups(self) -> None:
        self.calls.append("minimise-popups")

    def cluster_clicked(self, item: Any, point: Any) -> str:
        self.calls.append(("cluster-clicked", item, point))
        return "cluster"

    def describe_labels_pressed(self) -> bool:
        self.calls.append("describe-labels")
        return True

    def label_closed(self, popup: Any) -> None:
        self.calls.append(("label-closed", popup))

    def label_moved(self) -> None:
        self.calls.append("label-moved")

    def label_pressed(self, item: Any) -> None:
        self.calls.append(("label-pressed", item))

    def on_mouse_double_clicked(self, event: Any) -> bool:
        self.calls.append(("double-clicked", event))
        return True

    def on_mouse_hover(self, items: list[Any]) -> None:
        self.calls.append(("hover", items))


def _workbench(
    alignment: Any,
    shank: Any,
    histology: Any | None = None,
    load_data: Any | None = None,
    mouse_root: Any | None = None,
    session_selection: Any | None = None,
    probe_selection: Any | None = None,
    output_path: Any | None = None,
    path_dialog: Any | None = None,
    load_workflow: Any | None = None,
    output_folder_prompt: Any | None = None,
    folder_dialog: Any | None = None,
    save_workflow: Any | None = None,
    previous_alignment_load: Any | None = None,
    plot_exporter: Any | None = None,
    interaction: Any | None = None,
    lifecycle: Any | None = None,
    ephys_display: Any | None = None,
    slice_display: Any | None = None,
    reference_line_display: Any | None = None,
) -> DesktopWorkbench:
    displays = _displays(
        ephys=ephys_display,
        histology=histology,
        reference_lines=reference_line_display,
        slice_display=slice_display,
    )
    return DesktopWorkbench(
        app=object(),
        displays=displays,
        alignment_presenter=alignment,
        shank_presenter=shank,
        load_data_presenter=load_data or FakeLoadDataPresenter(),
        probe_selection_presenter=probe_selection or FakeProbeSelectionPresenter(),
        session_selection_presenter=(
            session_selection or FakeSessionSelectionPresenter()
        ),
        mouse_root_presenter=mouse_root or FakeMouseRootPresenter(),
        output_path_presenter=output_path or FakeOutputPathPresenter(),
        path_dialog_presenter=path_dialog or FakePathDialogPresenter(),
        load_workflow_presenter=load_workflow or FakeLoadWorkflowPresenter(),
        output_folder_prompt=output_folder_prompt or FakeOutputFolderPrompt(),
        folder_dialog=folder_dialog or FakeFolderDialog(),
        save_workflow_presenter=save_workflow or FakeSaveWorkflowPresenter(),
        previous_alignment_load_presenter=(
            previous_alignment_load or FakePreviousAlignmentLoadPresenter()
        ),
        plot_exporter=plot_exporter or FakePlotExporter(),
        interaction_presenter=interaction or FakeInteractionPresenter(),
        lifecycle_presenter=lifecycle or object(),
    )


def test_workbench_owns_event_subscription_lifecycle() -> None:
    alignment_sub = FakeSubscription()
    shank_sub = FakeSubscription()
    alignment = FakeAlignmentPresenter([alignment_sub])
    shank = FakeShankPresenter([shank_sub])
    workbench = _workbench(alignment, shank, FakeHistologyDisplay())

    subscriptions = workbench.connect_events()
    second_connect = workbench.connect_events()

    assert subscriptions == [alignment_sub, shank_sub]
    assert second_connect == subscriptions
    assert alignment.connect_count == 1
    assert shank.connect_count == 1

    workbench.disconnect_events()
    workbench.disconnect_events()

    assert alignment_sub.disconnect_count == 1
    assert shank_sub.disconnect_count == 1


def test_workbench_delegates_focused_presenter_entry_points() -> None:
    shank = FakeShankPresenter([])
    histology = FakeHistologyDisplay()
    workbench = _workbench(FakeAlignmentPresenter([]), shank, histology)

    workbench.render_loaded_shank(shank_idx=2, preserve_plot_selection=True)
    workbench.render_active_aligned_histology("fig", movable=False)
    workbench.render_active_reference_histology("ref", movable=True)
    workbench.render_active_scale_factor()
    workbench.render_active_fit()
    workbench.render_active_histology_panels()

    assert shank.render_calls == [(2, True)]
    assert histology.calls == [
        ("aligned", "fig", False),
        ("reference", "ref", True),
        "scale",
        "fit",
        "panels",
    ]


def test_workbench_delegates_selection_and_load_entry_points() -> None:
    load_data = FakeLoadDataPresenter()
    mouse_root = FakeMouseRootPresenter()
    session_selection = FakeSessionSelectionPresenter()
    probe_selection = FakeProbeSelectionPresenter()
    output_path = FakeOutputPathPresenter()
    path_dialog = FakePathDialogPresenter()
    load_workflow = FakeLoadWorkflowPresenter()
    output_folder_prompt = FakeOutputFolderPrompt()
    folder_dialog = FakeFolderDialog()
    save_workflow = FakeSaveWorkflowPresenter()
    previous_alignment_load = FakePreviousAlignmentLoadPresenter()
    plot_exporter = FakePlotExporter()
    interaction = FakeInteractionPresenter()
    workbench = _workbench(
        FakeAlignmentPresenter([]),
        FakeShankPresenter([]),
        FakeHistologyDisplay(),
        load_data=load_data,
        mouse_root=mouse_root,
        session_selection=session_selection,
        probe_selection=probe_selection,
        output_path=output_path,
        path_dialog=path_dialog,
        load_workflow=load_workflow,
        output_folder_prompt=output_folder_prompt,
        folder_dialog=folder_dialog,
        save_workflow=save_workflow,
        previous_alignment_load=previous_alignment_load,
        plot_exporter=plot_exporter,
        interaction=interaction,
    )

    assert workbench.load_heavy_data()
    assert workbench.set_mouse_root("root")
    assert workbench.mouse_root_edited()
    assert workbench.session_selected()
    assert workbench.probe_selected()
    assert workbench.load_data_button_pressed()
    assert workbench.set_save_root("save-root")
    assert workbench.select_mouse_root()
    assert workbench.select_output_root()
    assert workbench.output_folder_edited()
    assert workbench.ensure_output_directory_for_save("requirement")
    workbench.log_load_requirement("log-me")
    assert workbench.select_existing_directory_text("Choose") == "/selected"
    assert workbench.save_alignment_outputs()
    assert workbench.display_qc_options()
    assert workbench.qc_button_clicked()
    assert workbench.load_existing_alignments()
    workbench.export_plots("plots", sess_info="session-")
    workbench.display_session_notes()
    workbench.popup_closed("popup")
    workbench.popup_moved()
    workbench.close_popups()
    workbench.minimise_popups()
    assert workbench.cluster_clicked("item", "point") == "cluster"
    assert workbench.describe_labels_pressed()
    workbench.label_closed("label-popup")
    workbench.label_moved()
    workbench.label_pressed("label")
    assert workbench.on_mouse_double_clicked("event")
    workbench.on_mouse_hover(["items"])

    assert load_data.load_count == 1
    assert mouse_root.set_roots == ["root"]
    assert mouse_root.edited_count == 1
    assert session_selection.selected_count == 1
    assert probe_selection.selected_count == 1
    assert load_workflow.load_count == 1
    assert load_workflow.logged == ["log-me"]
    assert output_path.save_roots == ["save-root"]
    assert output_path.edited_count == 1
    assert path_dialog.mouse_root_count == 1
    assert path_dialog.output_root_count == 1
    assert output_folder_prompt.requirements == ["requirement"]
    assert folder_dialog.titles == ["Choose"]
    assert save_workflow.saved_count == 1
    assert save_workflow.qc_display_count == 1
    assert save_workflow.qc_clicked_count == 1
    assert previous_alignment_load.load_count == 1
    assert plot_exporter.exports == [("plots", "session-")]
    assert interaction.calls == [
        "notes",
        ("popup-closed", "popup"),
        "popup-moved",
        "close-popups",
        "minimise-popups",
        ("cluster-clicked", "item", "point"),
        "describe-labels",
        ("label-closed", "label-popup"),
        "label-moved",
        ("label-pressed", "label"),
        ("double-clicked", "event"),
        ("hover", ["items"]),
    ]


def _render_ports() -> DesktopRenderPorts:
    return DesktopRenderPorts(
        alignment=DesktopAlignmentRenderPorts(
            restore_lin_fit=lambda _lin_fit: None,
            capture_depth_plot_y_ranges=lambda: None,
            restore_depth_plot_y_ranges=lambda _ranges: None,
            create_reference_lines_for_previous_alignment=lambda: None,
            set_default_feature_y_range=lambda: None,
            update_status=lambda: None,
        ),
        shank=DesktopShankRenderPorts(
            capture_plot_selection=lambda _preserve: DesktopShankSelectionState(),
            render_alignment_choices=lambda _choices: None,
            apply_plot_data_state=lambda _state: None,
            raw_image_payloads=dict,
            render_plot_menus=lambda _state: None,
            render_histology_plots=lambda _shank_idx: None,
            configure_view=lambda _preserve: None,
            offline=lambda: True,
        ),
    )


def _selection_workflow_callbacks() -> DesktopSelectionWorkflowCallbacks:
    return DesktopSelectionWorkflowCallbacks(
        capture_pending_reference_lines=lambda: None,
        select_shank_for_view=lambda _shank_idx, _source: 0,
        clear_empty_state=lambda: None,
        set_histology_available=lambda _available: None,
        mouse_root_loaded=lambda: True,
        clear_histology_context=lambda: None,
        select_first_session=lambda: None,
        select_first_probe=lambda: None,
        active_shank_idx=lambda: 0,
        busy_context=lambda *args, **kwargs: SimpleNamespace(
            __enter__=lambda: None,
            __exit__=lambda *_args: None,
        ),
    )


def _workbench_ports() -> DesktopWorkbenchPorts:
    return DesktopWorkbenchPorts(
        render=_render_ports(),
        selection=_selection_workflow_callbacks(),
        lifecycle=DesktopLifecyclePorts(
            close_popups=lambda: None,
            reset_raw_image_payloads=lambda: None,
            show_empty_state=lambda: None,
            collect_garbage=lambda: None,
        ),
        save_workflow=DesktopSaveWorkflowPorts(
            use_docdb=lambda: False,
            render_alignment_choices=lambda _choices: None,
            busy_context=lambda *args, **kwargs: SimpleNamespace(
                __enter__=lambda: None,
                __exit__=lambda *_args: None,
            ),
            complete_button=lambda: object(),
            histology_available=lambda: True,
            open_qc_dialog=lambda: None,
            ephys_qc=lambda: "Pass",
            selected_qc_descriptions=list,
            warning=lambda _title, _message: None,
        ),
        previous_alignment_load=DesktopPreviousAlignmentLoadPorts(
            use_docdb=lambda: False,
            set_reload_folder_text=lambda _text: None,
            render_alignment_choices=lambda _choices: None,
            select_alignment=lambda _idx: None,
            busy_context=lambda *args, **kwargs: SimpleNamespace(
                __enter__=lambda: None,
                __exit__=lambda *_args: None,
            ),
            reload_button=lambda: object(),
        ),
        export=DesktopExportPorts(
            ephys_graphics_layout=object(),
            ephys_data_area=object(),
            slice_plot=object(),
            slice_trajectory_pen=object(),
            reset_axis=lambda: None,
            set_view=lambda **_kwargs: None,
            set_axis=lambda *_args, **_kwargs: None,
            set_font=lambda *_args, **_kwargs: None,
            ephys_sizes=lambda: (11.0, 3.0),
            slice_geometry=lambda: (100.0, 200.0, "rect"),
        ),
        interaction=DesktopInteractionPorts(
            popup_manager=object(),
            region_lookup_service=object(),
            struct_list=object(),
            struct_view=object(),
            struct_description=object(),
            scale_plot=object(),
            histology_plot=object(),
            histology_reference_plot=object(),
            scale_axis=object(),
            bar_colour=object(),
            line_pen=object(),
            histology_available=lambda: True,
            activate_window=lambda: None,
            set_axis=lambda *_args, **_kwargs: None,
            capture_pending_reference_lines=lambda: None,
        ),
    )


def test_workbench_factory_configures_focused_presenters() -> None:
    ports = _workbench_ports()
    queries = SimpleNamespace(
        active_mouse_root_path=lambda: None,
        active_output_root=lambda: None,
        has_output_directory=lambda: False,
    )
    commands = SimpleNamespace(can_load_data=lambda: Ok())
    app = SimpleNamespace(events=EventBus(), queries=queries, commands=commands)
    panel = FakeHistologyDisplay()
    ephys_display = FakeEphysDisplay()
    slice_display = FakeSliceDisplay()
    reference_line_display = FakeReferenceLineDisplay()

    displays = _displays(
        ephys=ephys_display,
        histology=panel,
        reference_lines=reference_line_display,
        slice_display=slice_display,
    )

    workbench = DesktopWorkbench.create(
        app=app,
        selection_view=object(),
        path_view=object(),
        parent=object(),
        displays=displays,
        ports=ports,
    )

    assert workbench.displays.histology is panel
    assert workbench.alignment_presenter.callbacks is not None
    assert workbench.shank_presenter.callbacks is not None
    assert workbench.load_data_presenter.callbacks is not None
    assert workbench.probe_selection_presenter.callbacks is not None
    assert workbench.session_selection_presenter.callbacks is not None
    assert workbench.mouse_root_presenter.callbacks is not None
    assert workbench.output_path_presenter.commands is commands
    assert workbench.path_dialog_presenter.callbacks.active_mouse_root is (
        queries.active_mouse_root_path
    )
    assert workbench.output_folder_prompt.callbacks.has_output_directory is (
        queries.has_output_directory
    )
    assert workbench.load_workflow_presenter.can_load_data is commands.can_load_data
    workbench.alignment_presenter.callbacks.render_histology_alignment("edit-state")
    assert panel.calls == [("edit", "edit-state")]
    workbench.alignment_presenter.callbacks.plot_channels("projection")
    assert slice_display.plotted_channels == ["projection"]
    workbench.alignment_presenter.callbacks.refresh_perpendicular_histology()
    assert slice_display.perpendicular_refreshes == 1
    assert workbench.shank_presenter.callbacks.render_alignment_choices is (
        ports.render.shank.render_alignment_choices
    )
    assert workbench.save_workflow_presenter.callbacks.use_docdb is (
        ports.save_workflow.use_docdb
    )
    assert workbench.previous_alignment_load_presenter.callbacks.use_docdb is (
        ports.previous_alignment_load.use_docdb
    )
    assert workbench.displays.ephys is ephys_display
    assert workbench.displays.slice is slice_display
    assert workbench.displays.reference_lines is reference_line_display
    workbench.alignment_presenter.callbacks.clear_reference_lines()
    workbench.alignment_presenter.callbacks.reattach_reference_lines()
    workbench.alignment_presenter.callbacks.update_reference_lines_to_alignment()
    workbench.shank_presenter.callbacks.clear_reference_lines()
    assert reference_line_display.clear_count == 2
    assert reference_line_display.reattach_count == 1
    assert reference_line_display.sync_count == 1
    workbench.shank_presenter.callbacks.render_ephys_plots("state")
    assert ephys_display.rendered_states == ["state"]
    workbench.shank_presenter.callbacks.restore_slice_selection(
        "menu",
        "selection",
        "label",
    )
    assert slice_display.restored == [("menu", "selection", "label")]
    assert workbench.plot_exporter.ephys_exporter.presenter is (
        ephys_display.plot_presenter
    )
    assert workbench.plot_exporter.ephys_exporter.panel is ephys_display.panel
    assert workbench.plot_exporter.slice_handles.slice_display is slice_display
    workbench.plot_exporter.add_lines_points()
    assert reference_line_display.add_count == 1
    assert workbench.plot_exporter.callbacks.set_axis is ports.export.set_axis
    assert workbench.plot_exporter.ephys_exporter.callbacks.set_view is (
        ports.export.set_view
    )
    assert workbench.interaction_presenter.popup_manager is (
        ports.interaction.popup_manager
    )
    assert workbench.interaction_presenter.reference_line_display is (
        reference_line_display
    )
    assert workbench.interaction_presenter.region_lookup_service is (
        ports.interaction.region_lookup_service
    )
    assert workbench.interaction_presenter.callbacks.set_axis is (
        ports.interaction.set_axis
    )
