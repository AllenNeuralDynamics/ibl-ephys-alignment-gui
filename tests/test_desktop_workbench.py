"""Tests for desktop workbench presenter composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop_alignment_presenter import (
    DesktopAlignmentRenderCallbacks,
)
from ephys_alignment_gui.desktop_histology_presenter import (
    DesktopHistologyPresenter,
    DesktopHistologyRenderCallbacks,
)
from ephys_alignment_gui.desktop_shank_presenter import (
    DesktopShankRenderCallbacks,
    DesktopShankSelectionState,
)
from ephys_alignment_gui.desktop_workbench import (
    DesktopSelectionWorkflowCallbacks,
    DesktopWorkbench,
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


class FakeHistologyPresenter:
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


def _workbench(
    alignment: Any,
    shank: Any,
    histology: Any,
    load_data: Any | None = None,
    mouse_root: Any | None = None,
    session_selection: Any | None = None,
    probe_selection: Any | None = None,
    output_path: Any | None = None,
    path_dialog: Any | None = None,
    load_workflow: Any | None = None,
    output_folder_prompt: Any | None = None,
    folder_dialog: Any | None = None,
) -> DesktopWorkbench:
    return DesktopWorkbench(
        app=object(),
        alignment_presenter=alignment,
        shank_presenter=shank,
        histology_presenter=histology,
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
    )


def test_workbench_owns_event_subscription_lifecycle() -> None:
    alignment_sub = FakeSubscription()
    shank_sub = FakeSubscription()
    alignment = FakeAlignmentPresenter([alignment_sub])
    shank = FakeShankPresenter([shank_sub])
    workbench = _workbench(alignment, shank, FakeHistologyPresenter())

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
    histology = FakeHistologyPresenter()
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
    workbench = _workbench(
        FakeAlignmentPresenter([]),
        FakeShankPresenter([]),
        FakeHistologyPresenter(),
        load_data=load_data,
        mouse_root=mouse_root,
        session_selection=session_selection,
        probe_selection=probe_selection,
        output_path=output_path,
        path_dialog=path_dialog,
        load_workflow=load_workflow,
        output_folder_prompt=output_folder_prompt,
        folder_dialog=folder_dialog,
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


def _alignment_callbacks(histology: DesktopHistologyPresenter):
    return DesktopAlignmentRenderCallbacks(
        restore_lin_fit=lambda _lin_fit: None,
        clear_reference_lines=lambda: None,
        capture_depth_plot_y_ranges=lambda: None,
        restore_depth_plot_y_ranges=lambda _ranges: None,
        reattach_reference_lines=lambda: None,
        render_histology_alignment=histology.render_alignment_edit,
        plot_channels=lambda _projection: None,
        refresh_perpendicular_histology=lambda: None,
        update_reference_lines_to_alignment=lambda: None,
        create_reference_lines_for_previous_alignment=lambda: None,
        set_default_feature_y_range=lambda: None,
        update_status=lambda: None,
    )


def _shank_callbacks() -> DesktopShankRenderCallbacks:
    return DesktopShankRenderCallbacks(
        capture_plot_selection=lambda _preserve: DesktopShankSelectionState(),
        clear_reference_lines=lambda: None,
        prepare_runtime=lambda _shank_idx: None,
        prepare_histology=lambda _shank_idx: True,
        apply_plot_data_state=lambda _state: None,
        raw_image_payloads=dict,
        render_plot_menus=lambda _state: None,
        render_ephys_plots=lambda _state: None,
        render_histology_plots=lambda _shank_idx: None,
        restore_slice_selection=lambda _state, _selection, _label: None,
        configure_view=lambda _preserve: None,
        histology_available=lambda: True,
        offline=lambda: True,
    )


def _selection_workflow_callbacks() -> DesktopSelectionWorkflowCallbacks:
    return DesktopSelectionWorkflowCallbacks(
        capture_pending_reference_lines=lambda: None,
        stash_and_detach_current=lambda: None,
        teardown_session=lambda: None,
        init_session_variables=lambda: None,
        select_shank_for_view=lambda _shank_idx, _source: 0,
        setup_session_view=lambda _preserve, _shank_idx: None,
        clear_empty_state=lambda: None,
        set_histology_available=lambda _available: None,
        mouse_root_loaded=lambda: True,
        show_empty_state=lambda: None,
        evict_stream_cache=lambda: None,
        clear_histology_context=lambda: None,
        select_first_session=lambda: None,
        select_first_probe=lambda: None,
        active_shank_idx=lambda: 0,
        busy_context=lambda *args, **kwargs: SimpleNamespace(
            __enter__=lambda: None,
            __exit__=lambda *_args: None,
        ),
    )


def test_workbench_factory_configures_focused_presenters() -> None:
    callbacks_seen: list[DesktopHistologyPresenter] = []

    def alignment_callbacks_factory(
        histology: DesktopHistologyPresenter,
    ) -> DesktopAlignmentRenderCallbacks:
        callbacks_seen.append(histology)
        return _alignment_callbacks(histology)

    histology_callbacks = DesktopHistologyRenderCallbacks(
        probe_extent_query_kwargs=dict,
        fit_depth_um=lambda: [],
        lin_fit_enabled=lambda: False,
        scale_factor_y_range=lambda: (0.0, 1.0),
    )
    queries = SimpleNamespace(
        active_mouse_root_path=lambda: None,
        active_output_root=lambda: None,
        has_output_directory=lambda: False,
    )
    commands = SimpleNamespace(can_load_data=lambda: Ok())
    app = SimpleNamespace(events=EventBus(), queries=queries, commands=commands)
    panel = object()

    workbench = DesktopWorkbench.create(
        app=app,
        selection_view=object(),
        path_view=object(),
        parent=object(),
        histology_panel=panel,
        histology_callbacks=histology_callbacks,
        alignment_callbacks_factory=alignment_callbacks_factory,
        shank_callbacks=_shank_callbacks(),
        selection_callbacks=_selection_workflow_callbacks(),
    )

    assert isinstance(workbench.histology_presenter, DesktopHistologyPresenter)
    assert workbench.histology_presenter.panel is panel
    assert callbacks_seen == [workbench.histology_presenter]
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
