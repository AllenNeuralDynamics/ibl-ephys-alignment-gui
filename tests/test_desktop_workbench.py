"""Tests for desktop workbench presenter composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop_displays import DesktopDisplays
from ephys_alignment_gui.desktop_export_view import DesktopExportView
from ephys_alignment_gui.desktop_render_composition import DesktopRenderCluster
from ephys_alignment_gui.desktop_shank_presenter import DesktopShankSelectionState
from ephys_alignment_gui.desktop_views import DesktopViews
from ephys_alignment_gui.desktop_workbench import DesktopWorkbench
from ephys_alignment_gui.desktop_workbench_composition import (
    DesktopWorkbenchPresenterCluster,
)
from ephys_alignment_gui.desktop_workbench_port_types import (
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


class FakeAlignmentEditActions:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def fit_button_pressed(self) -> bool:
        self.calls.append("fit")
        return True

    def offset_button_pressed(self, *, track_shift_m: float = 0.0) -> bool:
        self.calls.append(("offset", track_shift_m))
        return True

    def movedown_button_pressed(self) -> bool:
        self.calls.append("movedown")
        return True

    def moveup_button_pressed(self) -> bool:
        self.calls.append("moveup")
        return True

    def next_button_pressed(self) -> bool:
        self.calls.append("next")
        return True

    def prev_button_pressed(self) -> bool:
        self.calls.append("prev")
        return True

    def reset_button_pressed(self) -> bool:
        self.calls.append("reset")
        return True


class FakeShankSelectionActions:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def shank_selected(self) -> bool:
        self.calls.append("shank")
        return True


class FakeAlignmentSelectionActions:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def alignment_selected(self, idx: int) -> bool:
        self.calls.append(("alignment", idx))
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


class FakeLoadPreflightPresenter:
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


class FakeSavePresenter:
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
        self.created_lines: list[tuple[Any, Any]] = []
        self.lines_changed_callback = None
        self.current_positions = ([1.0], [2.0])

    def set_lines_changed_callback(self, callback: Any) -> None:
        self.lines_changed_callback = callback

    def positions(self) -> Any:
        return self.current_positions

    def clear(self) -> None:
        self.clear_count += 1

    def reattach(self) -> None:
        self.reattach_count += 1

    def sync_track_to_feature(self) -> None:
        self.sync_count += 1

    def add_to_plots(self) -> None:
        self.add_count += 1

    def create_lines(self, positions: Any, track_positions: Any = None) -> None:
        self.created_lines.append((positions, track_positions))


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
    load_preflight: Any | None = None,
    output_folder_prompt: Any | None = None,
    folder_dialog: Any | None = None,
    save: Any | None = None,
    previous_alignment_load: Any | None = None,
    plot_exporter: Any | None = None,
    interaction: Any | None = None,
    lifecycle: Any | None = None,
    reference_line_presenter: Any | None = None,
    histology_refresh_presenter: Any | None = None,
    alignment_edit_actions: Any | None = None,
    shank_selection_actions: Any | None = None,
    alignment_selection_actions: Any | None = None,
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
    render_cluster = DesktopRenderCluster(
        alignment_presenter=alignment,
        shank_presenter=shank,
        reference_line_presenter=reference_line_presenter or object(),
        histology_refresh_presenter=histology_refresh_presenter or object(),
        alignment_edit_actions=alignment_edit_actions or FakeAlignmentEditActions(),
        shank_selection_actions=(
            shank_selection_actions or FakeShankSelectionActions()
        ),
        alignment_selection_actions=(
            alignment_selection_actions or FakeAlignmentSelectionActions()
        ),
    )
    presenter_cluster = DesktopWorkbenchPresenterCluster(
        load_data_presenter=load_data or FakeLoadDataPresenter(),
        probe_selection_presenter=probe_selection or FakeProbeSelectionPresenter(),
        session_selection_presenter=(
            session_selection or FakeSessionSelectionPresenter()
        ),
        mouse_root_presenter=mouse_root or FakeMouseRootPresenter(),
        output_path_presenter=output_path or FakeOutputPathPresenter(),
        path_dialog_presenter=path_dialog or FakePathDialogPresenter(),
        load_preflight_presenter=load_preflight or FakeLoadPreflightPresenter(),
        output_folder_prompt=output_folder_prompt or FakeOutputFolderPrompt(),
        folder_dialog=folder_dialog or FakeFolderDialog(),
        save_presenter=save or FakeSavePresenter(),
        previous_alignment_load_presenter=(
            previous_alignment_load or FakePreviousAlignmentLoadPresenter()
        ),
        plot_exporter=plot_exporter or FakePlotExporter(),
        interaction_presenter=interaction or FakeInteractionPresenter(),
        lifecycle_presenter=lifecycle or object(),
    )
    return DesktopWorkbench(
        app=object(),
        views=object(),
        displays=displays,
        render_cluster=render_cluster,
        presenter_cluster=presenter_cluster,
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
    load_preflight = FakeLoadPreflightPresenter()
    output_folder_prompt = FakeOutputFolderPrompt()
    folder_dialog = FakeFolderDialog()
    save = FakeSavePresenter()
    previous_alignment_load = FakePreviousAlignmentLoadPresenter()
    plot_exporter = FakePlotExporter()
    interaction = FakeInteractionPresenter()
    alignment_edit_actions = FakeAlignmentEditActions()
    shank_selection_actions = FakeShankSelectionActions()
    alignment_selection_actions = FakeAlignmentSelectionActions()
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
        load_preflight=load_preflight,
        output_folder_prompt=output_folder_prompt,
        folder_dialog=folder_dialog,
        save=save,
        previous_alignment_load=previous_alignment_load,
        plot_exporter=plot_exporter,
        interaction=interaction,
        alignment_edit_actions=alignment_edit_actions,
        shank_selection_actions=shank_selection_actions,
        alignment_selection_actions=alignment_selection_actions,
    )

    assert workbench.load_heavy_data()
    assert workbench.fit_button_pressed()
    assert workbench.offset_button_pressed(track_shift_m=0.5)
    assert workbench.movedown_button_pressed()
    assert workbench.moveup_button_pressed()
    assert workbench.next_button_pressed()
    assert workbench.prev_button_pressed()
    assert workbench.reset_button_pressed()
    assert workbench.set_mouse_root("root")
    assert workbench.mouse_root_edited()
    assert workbench.session_selected()
    assert workbench.probe_selected()
    assert workbench.shank_selected(2)
    assert workbench.alignment_selected(3)
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
    assert load_preflight.load_count == 1
    assert load_preflight.logged == ["log-me"]
    assert output_path.save_roots == ["save-root"]
    assert output_path.edited_count == 1
    assert path_dialog.mouse_root_count == 1
    assert path_dialog.output_root_count == 1
    assert output_folder_prompt.requirements == ["requirement"]
    assert folder_dialog.titles == ["Choose"]
    assert save.saved_count == 1
    assert save.qc_display_count == 1
    assert save.qc_clicked_count == 1
    assert previous_alignment_load.load_count == 1
    assert plot_exporter.exports == [("plots", "session-")]
    assert alignment_edit_actions.calls == [
        "fit",
        ("offset", 0.5),
        "movedown",
        "moveup",
        "next",
        "prev",
        "reset",
    ]
    assert shank_selection_actions.calls == ["shank"]
    assert alignment_selection_actions.calls == [("alignment", 3)]
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
            configure_view=lambda _preserve: None,
            offline=lambda: True,
        ),
    )


def _workbench_ports() -> DesktopWorkbenchPorts:
    return DesktopWorkbenchPorts(
        render=_render_ports(),
        alignment_edit_actions=DesktopAlignmentEditActionPorts(
            histology_available=lambda: True,
            tip_position_um=lambda: 42.0,
        ),
        busy=DesktopBusyPorts(
            busy_context=lambda *args, **kwargs: SimpleNamespace(
                __enter__=lambda: None,
                __exit__=lambda *_args: None,
            ),
        ),
        load_data=DesktopLoadDataPorts(
            clear_empty_state=lambda: None,
            set_histology_available=lambda _available: None,
        ),
        lifecycle=DesktopLifecyclePorts(
            close_popups=lambda: None,
            reset_raw_image_payloads=lambda: None,
            show_empty_state=lambda: None,
            collect_garbage=lambda: None,
        ),
        save=DesktopSavePorts(
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
            busy_context=lambda *args, **kwargs: SimpleNamespace(
                __enter__=lambda: None,
                __exit__=lambda *_args: None,
            ),
            reload_button=lambda: object(),
        ),
        export=DesktopExportView(
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
        ),
    )


def test_workbench_factory_configures_focused_presenters() -> None:
    ports = _workbench_ports()
    queries = SimpleNamespace(
        active_mouse_root_path=lambda: None,
        active_output_root=lambda: None,
        has_output_directory=lambda: False,
        active_reference_line_state=lambda _shank_idx: SimpleNamespace(
            feature_positions_um=[1.0],
            track_positions_um=[2.0],
        ),
        active_shank_selection=lambda: SimpleNamespace(shank_idx=0),
        mouse_root_loaded=lambda: True,
    )
    captured_reference_lines: list[Any] = []
    commands = SimpleNamespace(
        can_load_data=lambda: Ok(),
        capture_active_reference_lines=lambda positions: (
            captured_reference_lines.append(positions) or Ok()
        ),
    )
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
    views = DesktopViews(
        selection=object(),
        path=object(),
        displays=displays,
        depth=object(),
        shank_screen=object(),
        alignment_screen=object(),
        export=ports.export,
    )

    workbench = DesktopWorkbench.create(
        app=app,
        parent=object(),
        views=views,
        ports=ports,
    )

    render_cluster = workbench.render_cluster
    presenter_cluster = workbench.presenter_cluster

    assert workbench.views is views
    assert workbench.displays.histology is panel
    assert render_cluster.alignment_presenter.callbacks is not None
    assert render_cluster.shank_presenter.callbacks is not None
    assert presenter_cluster.load_data_presenter.callbacks is not None
    assert presenter_cluster.probe_selection_presenter.callbacks is not None
    assert presenter_cluster.session_selection_presenter.callbacks is not None
    assert presenter_cluster.mouse_root_presenter.callbacks is not None
    assert presenter_cluster.output_path_presenter.commands is commands
    assert presenter_cluster.path_dialog_presenter.callbacks.active_mouse_root is (
        queries.active_mouse_root_path
    )
    assert presenter_cluster.output_folder_prompt.callbacks.has_output_directory is (
        queries.has_output_directory
    )
    assert presenter_cluster.load_preflight_presenter.can_load_data is (
        commands.can_load_data
    )
    assert render_cluster.alignment_edit_actions.commands is commands
    assert render_cluster.alignment_edit_actions.callbacks.tip_position_um() == 42.0
    assert render_cluster.shank_selection_actions.app is app
    assert render_cluster.shank_selection_actions.selection_view is not None
    assert render_cluster.alignment_selection_actions.app is app
    render_cluster.alignment_presenter.callbacks.render_histology_alignment(
        "edit-state"
    )
    assert panel.calls == [("edit", "edit-state")]
    render_cluster.alignment_presenter.callbacks.plot_channels("projection")
    assert slice_display.plotted_channels == ["projection"]
    render_cluster.alignment_presenter.callbacks.refresh_perpendicular_histology()
    assert slice_display.perpendicular_refreshes == 1
    assert render_cluster.shank_presenter.callbacks.render_alignment_choices is (
        ports.render.shank.render_alignment_choices
    )
    assert presenter_cluster.save_presenter.callbacks.use_docdb is (
        ports.save.use_docdb
    )
    assert presenter_cluster.previous_alignment_load_presenter.callbacks.use_docdb is (
        ports.previous_alignment_load.use_docdb
    )
    assert (
        presenter_cluster.previous_alignment_load_presenter.callbacks.select_alignment.__self__
        is render_cluster.alignment_selection_actions
    )
    assert (
        presenter_cluster.mouse_root_presenter.callbacks.select_first_session.__self__
        is presenter_cluster.session_selection_presenter
    )
    assert (
        presenter_cluster.session_selection_presenter.callbacks.select_first_probe.__self__
        is presenter_cluster.probe_selection_presenter
    )
    assert workbench.displays.ephys is ephys_display
    assert workbench.displays.slice is slice_display
    assert workbench.displays.reference_lines is reference_line_display
    assert reference_line_display.lines_changed_callback is not None
    reference_line_display.lines_changed_callback()
    assert captured_reference_lines == [([1.0], [2.0])]
    render_cluster.alignment_presenter.callbacks.clear_reference_lines()
    render_cluster.alignment_presenter.callbacks.reattach_reference_lines()
    render_cluster.alignment_presenter.callbacks.update_reference_lines_to_alignment()
    render_cluster.shank_presenter.callbacks.clear_reference_lines()
    assert reference_line_display.clear_count == 2
    assert reference_line_display.reattach_count == 1
    assert reference_line_display.sync_count == 1
    render_cluster.shank_presenter.callbacks.render_ephys_plots("state")
    assert ephys_display.rendered_states == ["state"]
    render_cluster.shank_presenter.callbacks.render_histology_plots(1)
    assert panel.calls[-1] == "panels"
    assert slice_display.perpendicular_refreshes == 2
    assert reference_line_display.created_lines == [([1.0], [2.0])]
    render_cluster.shank_presenter.callbacks.restore_slice_selection(
        "menu",
        "selection",
        "label",
    )
    assert slice_display.restored == [("menu", "selection", "label")]
    assert presenter_cluster.plot_exporter.ephys_exporter.presenter is (
        ephys_display.plot_presenter
    )
    assert presenter_cluster.plot_exporter.ephys_exporter.panel is ephys_display.panel
    assert presenter_cluster.plot_exporter.slice_handles.slice_display is slice_display
    presenter_cluster.plot_exporter.add_lines_points()
    assert reference_line_display.add_count == 1
    assert presenter_cluster.plot_exporter.callbacks.set_axis is ports.export.set_axis
    assert presenter_cluster.plot_exporter.ephys_exporter.callbacks.set_view is (
        ports.export.set_view
    )
    assert presenter_cluster.interaction_presenter.popup_manager is (
        ports.interaction.popup_manager
    )
    assert presenter_cluster.interaction_presenter.reference_line_display is (
        reference_line_display
    )
    assert presenter_cluster.interaction_presenter.callbacks.set_axis is (
        ports.interaction.set_axis
    )
