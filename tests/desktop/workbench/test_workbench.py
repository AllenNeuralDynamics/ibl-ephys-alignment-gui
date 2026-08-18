"""Tests for desktop workbench presenter composition."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.core.event_bus import EventBus
from ephys_alignment_gui.core.workflow import Ok
from ephys_alignment_gui.desktop.displays import DesktopDisplays
from ephys_alignment_gui.desktop.presenters.shank_presenter import (
    DesktopShankSelectionState,
)
from ephys_alignment_gui.desktop.views import DesktopViews
from ephys_alignment_gui.desktop.views.export_view import DesktopExportView
from ephys_alignment_gui.desktop.workbench import DesktopWorkbench
from ephys_alignment_gui.desktop.workbench.composition import (
    DesktopWorkbenchCoordinatorCluster,
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
from ephys_alignment_gui.desktop.workbench.render_composition import (
    DesktopRenderCluster,
)


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
        self.panel = object()

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

    def render_active_nearby(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        self.calls.append(("nearby", fig, movable))
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
    def __init__(
        self,
        subscriptions: list[FakeSubscription] | None = None,
        *,
        shutdown_result: bool = True,
    ) -> None:
        self.load_count = 0
        self.subscriptions = subscriptions or []
        self.connect_count = 0
        self.shutdown_result = shutdown_result
        self.shutdown_timeouts: list[int] = []
        self.active_work = False
        self.async_shutdown_reasons: list[str] = []

    def connect_load_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions

    def load_heavy_data(self) -> bool:
        self.load_count += 1
        return True

    def shutdown_active_load(self, *, timeout_ms: int = 5000) -> bool:
        self.shutdown_timeouts.append(timeout_ms)
        return self.shutdown_result

    def has_active_work(self) -> bool:
        return self.active_work

    def request_async_shutdown(self, reason: str = "application closing") -> bool:
        self.async_shutdown_reasons.append(reason)
        return self.active_work


class FakeLifecyclePresenter:
    def __init__(self, subscriptions: list[FakeSubscription] | None = None) -> None:
        self.subscriptions = subscriptions or []
        self.connect_count = 0
        self.startup_count = 0

    def connect_lifecycle_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions

    def initialize_startup_stream_state(self) -> None:
        self.startup_count += 1


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


class FakeDisplayActions:
    def __init__(self, subscriptions: list[FakeSubscription] | None = None) -> None:
        self.calls: list[Any] = []
        self.subscriptions = subscriptions or []
        self.connect_count = 0

    def connect_display_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions

    def toggle_histology_boundaries(self) -> bool:
        self.calls.append("toggle-histology-boundaries")
        return True

    def toggle_region_annotation_source(self) -> None:
        self.calls.append("toggle-region-annotation-source")

    def toggle_labels(self) -> None:
        self.calls.append("toggle-labels")

    def toggle_reference_lines(self) -> None:
        self.calls.append("toggle-reference-lines")

    def toggle_channels(self) -> None:
        self.calls.append("toggle-channels")

    def delete_selected_reference_line(self) -> None:
        self.calls.append("delete-selected-reference-line")

    def reset_axis(self) -> None:
        self.calls.append("reset-axis")

    def set_linear_fit_enabled(self, enabled: bool) -> bool:
        self.calls.append(("set-linear-fit", enabled))
        return True

    def sync_histology_top_to_tip(self) -> None:
        self.calls.append("sync-histology-top-to-tip")

    def sync_histology_tip_to_top(self) -> None:
        self.calls.append("sync-histology-tip-to-top")


class FakeEphysPlotPresenter:
    def __init__(self) -> None:
        self.rendered_states: list[Any] = []
        self.menu_attaches: list[Any] = []
        self.unit_filter_attaches: list[tuple[Any, Any]] = []
        self.toggles: list[tuple[Any, bool]] = []

    def render_shank_ephys_plots(self, state: Any) -> None:
        self.rendered_states.append(state)

    def attach_plot_menus(self, menu_bar: Any) -> None:
        self.menu_attaches.append(menu_bar)

    def attach_unit_filter_menu(self, menu_bar: Any, parent: Any) -> None:
        self.unit_filter_attaches.append((menu_bar, parent))

    def toggle_plot(self, menu: Any, *, reverse: bool = False) -> None:
        self.toggles.append((menu, reverse))


class FakeSlicePanelPresenter:
    def __init__(self) -> None:
        self.plotted_channels: list[Any] = []
        self.perpendicular_refreshes = 0

    def plot_channels(self, projection: Any = None) -> None:
        self.plotted_channels.append(projection)

    def refresh_perpendicular_histology(self, selection: Any = None) -> None:
        self.perpendicular_refreshes += 1


class FakeSliceMenuCoordinator:
    def __init__(self) -> None:
        self.restored: list[tuple[Any, Any, Any]] = []
        self.selection = "slice-selection"
        self.menu_attaches: list[tuple[Any, Any, bool]] = []
        self.toggles: list[bool] = []

    def current_selection(self) -> Any:
        return self.selection

    def restore_selection(
        self,
        slice_menu_state: Any,
        previous_selection: Any,
        previous_label: Any,
    ) -> None:
        self.restored.append((slice_menu_state, previous_selection, previous_label))

    def attach_menu(self, menu_bar: Any, *, parent: Any, offline: bool) -> None:
        self.menu_attaches.append((menu_bar, parent, offline))

    def toggle_plot(self, *, reverse: bool = False) -> None:
        self.toggles.append(reverse)


class FakeShankScreenView:
    def __init__(self) -> None:
        self.views: list[int] = []

    def set_view(self, *, view: int) -> None:
        self.views.append(view)


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
        self.selected_indices: list[int | None] = []

    def session_selected(self, idx: int | None = None) -> bool:
        self.selected_count += 1
        self.selected_indices.append(idx)
        return True


class FakeProbeSelectionPresenter:
    def __init__(self) -> None:
        self.selected_count = 0
        self.selected_indices: list[int | None] = []

    def probe_selected(self, idx: int | None = None) -> bool:
        self.selected_count += 1
        self.selected_indices.append(idx)
        return True


class FakeOutputPathPresenter:
    def __init__(self, subscriptions: list[FakeSubscription] | None = None) -> None:
        self.subscriptions = subscriptions or []
        self.connect_count = 0
        self.save_roots: list[Any] = []
        self.edited_count = 0

    def connect_path_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions

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

    def activate_selected_stream(self) -> bool:
        self.load_count += 1
        return True

    def log_requirement(self, requirement: Any) -> None:
        self.logged.append(requirement)


class FakeSelectionActivationPresenter:
    def __init__(self) -> None:
        self.session_indices: list[int | None] = []
        self.probe_indices: list[int | None] = []
        self.shank_indices: list[int | None] = []
        self.load_count = 0

    def session_selected(self, idx: int | None = None) -> bool:
        self.session_indices.append(idx)
        return True

    def probe_selected(self, idx: int | None = None) -> bool:
        self.probe_indices.append(idx)
        return True

    def shank_selected(self, idx: int | None = None) -> bool:
        self.shank_indices.append(idx)
        return True

    def activate_selected_stream(self) -> bool:
        self.load_count += 1
        return True


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
    def __init__(
        self,
        subscriptions: list[FakeSubscription] | None = None,
        *,
        shutdown_result: bool = True,
    ) -> None:
        self.saved_count = 0
        self.qc_display_count = 0
        self.qc_clicked_count = 0
        self.subscriptions = subscriptions or []
        self.connect_count = 0
        self.shutdown_result = shutdown_result
        self.shutdown_timeouts: list[int] = []
        self.active_work = False
        self.async_shutdown_reasons: list[str] = []

    def connect_save_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions

    def save_alignment_outputs(self) -> bool:
        self.saved_count += 1
        return True

    def shutdown_active_save(self, *, timeout_ms: int = 5000) -> bool:
        self.shutdown_timeouts.append(timeout_ms)
        return self.shutdown_result

    def has_active_work(self) -> bool:
        return self.active_work

    def request_async_shutdown(self, reason: str = "application closing") -> bool:
        self.async_shutdown_reasons.append(reason)
        return self.active_work

    def display_qc_options(self) -> bool:
        self.qc_display_count += 1
        return True

    def qc_button_clicked(self) -> bool:
        self.qc_clicked_count += 1
        return True


class FakePreviousAlignmentLoadPresenter:
    def __init__(self, subscriptions: list[FakeSubscription] | None = None) -> None:
        self.load_count = 0
        self.subscriptions = subscriptions or []
        self.connect_count = 0

    def connect_previous_alignment_events(self) -> list[FakeSubscription]:
        self.connect_count += 1
        return self.subscriptions

    def load_existing_alignments(self) -> bool:
        self.load_count += 1
        return True


class FakeAutosaveRecoveryCoordinator:
    def __init__(self) -> None:
        self.recover_count = 0

    def recover_autosave(self) -> bool:
        self.recover_count += 1
        return True


class FakePlotExporter:
    def __init__(self) -> None:
        self.exports: list[tuple[Any, str]] = []

    def export(self, output_dir: Any, *, sess_info: str = "") -> None:
        self.exports.append((output_dir, sess_info))


class FakePlotExportPresenter:
    def __init__(self) -> None:
        self.saved_paths: list[Any] = []

    def save_plots(self, save_path: Any = None) -> bool:
        self.saved_paths.append(save_path)
        return True


class FakeEphysDisplay:
    def __init__(self) -> None:
        self.panel = SimpleNamespace(
            image_raster_request=lambda: None,
            render_image=lambda data: self.rendered_states.append(("image", data)),
            render_scatter=lambda data: self.rendered_states.append(("scatter", data)),
            render_line=lambda data: self.rendered_states.append(("line", data)),
            render_probe=lambda data, bounds: self.rendered_states.append(
                ("probe", data, bounds)
            ),
            feature_y_range=lambda: (0.0, 1.0),
        )
        self.rendered_states: list[Any] = []


class FakeSliceDisplay:
    def __init__(self) -> None:
        self.restored: list[tuple[Any, Any, Any]] = []
        self.plotted_channels: list[Any] = []
        self.perpendicular_refreshes = 0
        self.view = SimpleNamespace(
            plot_channels=lambda projection: self.plotted_channels.append(projection),
            toggle_channel_visibility=lambda: None,
            current_channel_locations_ras=lambda: None,
            render_export_trajectory_overlay=lambda *args, **kwargs: None,
        )

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
        self.add_count = 0
        self.remove_count = 0
        self.reattach_count = 0
        self.created_lines: list[tuple[Any, Any]] = []
        self.replaced_lines: list[tuple[Any, Any]] = []
        self.raw_replaced_lines: list[tuple[Any, Any]] = []
        self.lines_changed_callback = None
        self.current_positions = ([1.0], [2.0])

    def set_lines_changed_callback(self, callback: Any) -> None:
        self.lines_changed_callback = callback

    def set_track_display_transform(
        self,
        *,
        track_to_warped_position: Any,
        warped_position_to_track: Any,
    ) -> None:
        self.track_to_warped_position = track_to_warped_position
        self.warped_position_to_track = warped_position_to_track

    def positions(self) -> Any:
        return self.current_positions

    def clear(self) -> None:
        self.clear_count += 1

    def add_to_plots(self) -> None:
        self.add_count += 1

    def remove_from_plots(self) -> None:
        self.remove_count += 1

    def reattach(self) -> None:
        self.reattach_count += 1

    def create_lines(self, positions: Any, track_positions: Any = None) -> None:
        self.created_lines.append((positions, track_positions))

    def replace_lines(self, positions: Any, track_positions: Any = None) -> None:
        self.replaced_lines.append((positions, track_positions))

    def replace_lines_from_raw_track(
        self,
        positions: Any,
        raw_track_positions: Any,
    ) -> None:
        self.raw_replaced_lines.append((positions, raw_track_positions))


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
    selection_activation: Any | None = None,
    output_folder_prompt: Any | None = None,
    folder_dialog: Any | None = None,
    save: Any | None = None,
    previous_alignment_load: Any | None = None,
    autosave_recovery: Any | None = None,
    plot_exporter: Any | None = None,
    plot_export_coordinator: Any | None = None,
    interaction: Any | None = None,
    lifecycle: Any | None = None,
    reference_line_presenter: Any | None = None,
    histology_refresh_presenter: Any | None = None,
    ephys_plot_presenter: Any | None = None,
    histology_presenter: Any | None = None,
    slice_panel_presenter: Any | None = None,
    slice_menu_coordinator: Any | None = None,
    alignment_edit_actions: Any | None = None,
    display_actions: Any | None = None,
    shank_selection_actions: Any | None = None,
    alignment_selection_actions: Any | None = None,
    ephys_display: Any | None = None,
    slice_display: Any | None = None,
    reference_line_display: Any | None = None,
    views: Any | None = None,
) -> DesktopWorkbench:
    displays = _displays(
        ephys=ephys_display,
        histology=histology,
        reference_lines=reference_line_display,
        slice_display=slice_display,
    )
    render_cluster = DesktopRenderCluster(
        alignment_presenter=alignment,
        ephys_plot_presenter=ephys_plot_presenter or FakeEphysPlotPresenter(),
        histology_presenter=histology_presenter or histology or FakeHistologyDisplay(),
        slice_panel_presenter=slice_panel_presenter or FakeSlicePanelPresenter(),
        slice_menu_coordinator=slice_menu_coordinator or FakeSliceMenuCoordinator(),
        shank_presenter=shank,
        reference_line_presenter=reference_line_presenter or object(),
        histology_refresh_presenter=histology_refresh_presenter or object(),
        alignment_edit_actions=alignment_edit_actions or FakeAlignmentEditActions(),
        display_actions=display_actions or FakeDisplayActions(),
        shank_selection_actions=(
            shank_selection_actions or FakeShankSelectionActions()
        ),
        alignment_selection_actions=(
            alignment_selection_actions or FakeAlignmentSelectionActions()
        ),
    )
    coordinator_cluster = DesktopWorkbenchCoordinatorCluster(
        load_data_coordinator=load_data or FakeLoadDataPresenter(),
        probe_selection_coordinator=probe_selection or FakeProbeSelectionPresenter(),
        session_selection_coordinator=(
            session_selection or FakeSessionSelectionPresenter()
        ),
        mouse_root_coordinator=mouse_root or FakeMouseRootPresenter(),
        output_path_coordinator=output_path or FakeOutputPathPresenter(),
        path_dialog_coordinator=path_dialog or FakePathDialogPresenter(),
        load_preflight_coordinator=load_preflight or FakeLoadPreflightPresenter(),
        selection_activation_coordinator=(
            selection_activation or FakeSelectionActivationPresenter()
        ),
        output_folder_prompt=output_folder_prompt or FakeOutputFolderPrompt(),
        folder_dialog=folder_dialog or FakeFolderDialog(),
        save_coordinator=save or FakeSavePresenter(),
        previous_alignment_load_coordinator=(
            previous_alignment_load or FakePreviousAlignmentLoadPresenter()
        ),
        autosave_recovery_coordinator=(
            autosave_recovery or FakeAutosaveRecoveryCoordinator()
        ),
        plot_exporter=plot_exporter or FakePlotExporter(),
        plot_export_coordinator=plot_export_coordinator or FakePlotExportPresenter(),
        interaction_coordinator=interaction or FakeInteractionPresenter(),
        lifecycle_coordinator=lifecycle or FakeLifecyclePresenter(),
    )
    return DesktopWorkbench(
        app=object(),
        views=views or SimpleNamespace(shank_screen=FakeShankScreenView()),
        displays=displays,
        render_cluster=render_cluster,
        coordinator_cluster=coordinator_cluster,
    )


def test_workbench_owns_event_subscription_lifecycle() -> None:
    alignment_sub = FakeSubscription()
    shank_sub = FakeSubscription()
    display_sub = FakeSubscription()
    load_sub = FakeSubscription()
    output_path_sub = FakeSubscription()
    lifecycle_sub = FakeSubscription()
    save_sub = FakeSubscription()
    previous_alignment_sub = FakeSubscription()
    alignment = FakeAlignmentPresenter([alignment_sub])
    shank = FakeShankPresenter([shank_sub])
    display_actions = FakeDisplayActions([display_sub])
    load_data = FakeLoadDataPresenter([load_sub])
    output_path = FakeOutputPathPresenter([output_path_sub])
    lifecycle = FakeLifecyclePresenter([lifecycle_sub])
    save = FakeSavePresenter([save_sub])
    previous_alignment_load = FakePreviousAlignmentLoadPresenter(
        [previous_alignment_sub]
    )
    workbench = _workbench(
        alignment,
        shank,
        FakeHistologyDisplay(),
        display_actions=display_actions,
        load_data=load_data,
        output_path=output_path,
        lifecycle=lifecycle,
        save=save,
        previous_alignment_load=previous_alignment_load,
    )

    subscriptions = workbench.connect_events()
    second_connect = workbench.connect_events()

    assert subscriptions == [
        alignment_sub,
        shank_sub,
        display_sub,
        load_sub,
        output_path_sub,
        lifecycle_sub,
        save_sub,
        previous_alignment_sub,
    ]
    assert second_connect == subscriptions
    assert alignment.connect_count == 1
    assert shank.connect_count == 1
    assert display_actions.connect_count == 1
    assert load_data.connect_count == 1
    assert output_path.connect_count == 1
    assert lifecycle.connect_count == 1
    assert save.connect_count == 1
    assert previous_alignment_load.connect_count == 1

    workbench.disconnect_events()
    workbench.disconnect_events()

    assert alignment_sub.disconnect_count == 1
    assert shank_sub.disconnect_count == 1
    assert display_sub.disconnect_count == 1
    assert load_sub.disconnect_count == 1
    assert output_path_sub.disconnect_count == 1
    assert lifecycle_sub.disconnect_count == 1
    assert save_sub.disconnect_count == 1
    assert previous_alignment_sub.disconnect_count == 1


def test_workbench_exposes_shell_plot_menu_actions() -> None:
    ephys = FakeEphysPlotPresenter()
    slice_menu = FakeSliceMenuCoordinator()
    shank_screen = FakeShankScreenView()
    workbench = _workbench(
        FakeAlignmentPresenter([]),
        FakeShankPresenter([]),
        FakeHistologyDisplay(),
        ephys_plot_presenter=ephys,
        slice_menu_coordinator=slice_menu,
        views=SimpleNamespace(shank_screen=shank_screen),
    )
    menu_bar = object()
    parent = object()

    workbench.attach_plot_menus(menu_bar, parent=parent, offline=True)
    workbench.toggle_ephys_plot("image")
    workbench.toggle_ephys_plot("line", reverse=True)
    workbench.toggle_slice_plot()
    workbench.toggle_slice_plot(reverse=True)
    workbench.set_ephys_view(view=2)

    assert ephys.menu_attaches == [menu_bar]
    assert ephys.unit_filter_attaches == [(menu_bar, parent)]
    assert ephys.toggles == [("image", False), ("line", True)]
    assert slice_menu.menu_attaches == [(menu_bar, parent, True)]
    assert slice_menu.toggles == [False, True]
    assert shank_screen.views == [2]


def test_workbench_shutdown_settles_load_before_disconnecting_events() -> None:
    alignment_sub = FakeSubscription()
    shank_sub = FakeSubscription()
    display_sub = FakeSubscription()
    load_sub = FakeSubscription()
    output_path_sub = FakeSubscription()
    lifecycle_sub = FakeSubscription()
    save_sub = FakeSubscription()
    previous_alignment_sub = FakeSubscription()
    load_data = FakeLoadDataPresenter([load_sub])
    save = FakeSavePresenter([save_sub])
    output_path = FakeOutputPathPresenter([output_path_sub])
    workbench = _workbench(
        FakeAlignmentPresenter([alignment_sub]),
        FakeShankPresenter([shank_sub]),
        FakeHistologyDisplay(),
        display_actions=FakeDisplayActions([display_sub]),
        load_data=load_data,
        output_path=output_path,
        lifecycle=FakeLifecyclePresenter([lifecycle_sub]),
        save=save,
        previous_alignment_load=FakePreviousAlignmentLoadPresenter(
            [previous_alignment_sub]
        ),
    )
    workbench.connect_events()

    assert workbench.shutdown(timeout_ms=123)

    assert load_data.shutdown_timeouts == [123]
    assert save.shutdown_timeouts == [123]
    assert alignment_sub.disconnect_count == 1
    assert shank_sub.disconnect_count == 1
    assert display_sub.disconnect_count == 1
    assert load_sub.disconnect_count == 1
    assert output_path_sub.disconnect_count == 1
    assert lifecycle_sub.disconnect_count == 1
    assert save_sub.disconnect_count == 1
    assert previous_alignment_sub.disconnect_count == 1


def test_workbench_shutdown_leaves_events_connected_when_load_does_not_stop() -> None:
    alignment_sub = FakeSubscription()
    load_data = FakeLoadDataPresenter(shutdown_result=False)
    workbench = _workbench(
        FakeAlignmentPresenter([alignment_sub]),
        FakeShankPresenter([]),
        FakeHistologyDisplay(),
        load_data=load_data,
    )
    workbench.connect_events()

    assert not workbench.shutdown(timeout_ms=123)

    assert load_data.shutdown_timeouts == [123]
    assert alignment_sub.disconnect_count == 0


def test_workbench_shutdown_leaves_events_connected_when_save_does_not_stop() -> None:
    alignment_sub = FakeSubscription()
    save = FakeSavePresenter(shutdown_result=False)
    workbench = _workbench(
        FakeAlignmentPresenter([alignment_sub]),
        FakeShankPresenter([]),
        FakeHistologyDisplay(),
        save=save,
    )
    workbench.connect_events()

    assert not workbench.shutdown(timeout_ms=123)

    assert save.shutdown_timeouts == [123]
    assert alignment_sub.disconnect_count == 0


def test_workbench_async_shutdown_requests_cancellation_and_waits_to_finalize() -> None:
    alignment_sub = FakeSubscription()
    load_data = FakeLoadDataPresenter()
    save = FakeSavePresenter()
    load_data.active_work = True
    save.active_work = True
    workbench = _workbench(
        FakeAlignmentPresenter([alignment_sub]),
        FakeShankPresenter([]),
        FakeHistologyDisplay(),
        load_data=load_data,
        save=save,
    )
    workbench.connect_events()

    assert workbench.has_active_work()
    assert workbench.request_async_shutdown("closing")
    assert not workbench.shutdown_ready()
    assert not workbench.finalize_shutdown()

    load_data.active_work = False
    save.active_work = False

    assert workbench.shutdown_ready()
    assert workbench.finalize_shutdown()
    assert alignment_sub.disconnect_count == 1
    assert load_data.async_shutdown_reasons == ["closing"]
    assert save.async_shutdown_reasons == ["closing"]


def test_workbench_delegates_focused_presenter_entry_points() -> None:
    shank = FakeShankPresenter([])
    histology = FakeHistologyDisplay()
    workbench = _workbench(FakeAlignmentPresenter([]), shank, histology)

    workbench.render_loaded_shank(shank_idx=2, preserve_plot_selection=True)
    workbench.render_active_aligned_histology("fig", movable=False)
    workbench.render_active_reference_histology("ref", movable=True)
    workbench.render_active_nearby_histology("nearby", movable=True)
    workbench.render_active_scale_factor()
    workbench.render_active_fit()
    workbench.render_active_histology_panels()

    assert shank.render_calls == [(2, True)]
    assert histology.calls == [
        ("aligned", "fig", False),
        ("reference", "ref", True),
        ("nearby", "nearby", True),
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
    selection_activation = FakeSelectionActivationPresenter()
    output_folder_prompt = FakeOutputFolderPrompt()
    folder_dialog = FakeFolderDialog()
    save = FakeSavePresenter()
    previous_alignment_load = FakePreviousAlignmentLoadPresenter()
    autosave_recovery = FakeAutosaveRecoveryCoordinator()
    plot_exporter = FakePlotExporter()
    plot_export_coordinator = FakePlotExportPresenter()
    interaction = FakeInteractionPresenter()
    alignment_edit_actions = FakeAlignmentEditActions()
    display_actions = FakeDisplayActions()
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
        selection_activation=selection_activation,
        output_folder_prompt=output_folder_prompt,
        folder_dialog=folder_dialog,
        save=save,
        previous_alignment_load=previous_alignment_load,
        autosave_recovery=autosave_recovery,
        plot_exporter=plot_exporter,
        plot_export_coordinator=plot_export_coordinator,
        interaction=interaction,
        alignment_edit_actions=alignment_edit_actions,
        display_actions=display_actions,
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
    assert workbench.activate_selected_stream()
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
    assert workbench.recover_autosave()
    workbench.export_plots("plots", sess_info="session-")
    assert workbench.save_plots("save-plots")
    assert workbench.toggle_histology_boundaries()
    workbench.toggle_region_annotation_source()
    workbench.toggle_labels()
    workbench.toggle_reference_lines()
    workbench.toggle_channels()
    workbench.delete_selected_reference_line()
    workbench.reset_axis()
    assert workbench.set_linear_fit_enabled(True)
    workbench.sync_histology_top_to_tip()
    workbench.sync_histology_tip_to_top()
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
    assert session_selection.selected_count == 0
    assert probe_selection.selected_count == 0
    assert selection_activation.session_indices == [None]
    assert selection_activation.probe_indices == [None]
    assert selection_activation.shank_indices == [2]
    assert selection_activation.load_count == 1
    assert load_preflight.load_count == 0
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
    assert autosave_recovery.recover_count == 1
    assert plot_exporter.exports == [("plots", "session-")]
    assert plot_export_coordinator.saved_paths == ["save-plots"]
    assert display_actions.calls == [
        "toggle-histology-boundaries",
        "toggle-region-annotation-source",
        "toggle-labels",
        "toggle-reference-lines",
        "toggle-channels",
        "delete-selected-reference-line",
        "reset-axis",
        ("set-linear-fit", True),
        "sync-histology-top-to-tip",
        "sync-histology-tip-to-top",
    ]
    assert alignment_edit_actions.calls == [
        "fit",
        ("offset", 0.5),
        "movedown",
        "moveup",
        "next",
        "prev",
        "reset",
    ]
    assert shank_selection_actions.calls == []
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
            capture_depth_plot_y_ranges=lambda: None,
            restore_depth_plot_y_ranges=lambda _ranges: None,
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
            save_progress_dialog=lambda: object(),
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
            struct_list=object,
            struct_view=object,
            struct_description=object,
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
    workspace_queries = SimpleNamespace(
        active_mouse_root_path=lambda: None,
        active_output_root=lambda: None,
        active_output_directory=lambda: None,
        active_output_package_directory=lambda: None,
        has_output_directory=lambda: False,
        active_reference_line_state=lambda _shank_idx: SimpleNamespace(
            feature_positions_um=[1.0],
            track_positions_um=[2.0],
        ),
        active_shank_selection=lambda: SimpleNamespace(shank_idx=0),
        fit_depth_um=lambda: [],
        linear_fit_enabled=lambda: False,
        reference_lines_visible=lambda: True,
        track_to_warped_feature_positions_um=lambda values: values,
        warped_feature_to_track_positions_um=lambda values: values,
        mouse_root_loaded=lambda: True,
    )
    queries = SimpleNamespace(workspace=workspace_queries)
    captured_reference_lines: list[Any] = []
    command_impl = SimpleNamespace(
        can_load_data=lambda: Ok(),
        capture_active_reference_lines=lambda positions: (
            captured_reference_lines.append(positions) or Ok()
        ),
        evict_stream_cache=lambda: Ok(),
        start_histology_warmup=lambda _mouse_root: Ok(),
    )
    commands = SimpleNamespace(
        paths=command_impl,
        metadata=command_impl,
        shanks=command_impl,
        load=command_impl,
        loaded_shank=command_impl,
        persistence=command_impl,
        edit=command_impl,
        display=command_impl,
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
        depth=object(),
        shank_screen=SimpleNamespace(raw_image_payload_mapping=lambda: {}),
        alignment_screen=object(),
        export=ports.export,
    )

    workbench = DesktopWorkbench.create(
        app=app,
        parent=object(),
        views=views,
        displays=displays,
        ports=ports,
    )

    render_cluster = workbench.render_cluster
    coordinator_cluster = workbench.coordinator_cluster

    assert workbench.views is views
    assert workbench.displays.histology is panel
    assert render_cluster.alignment_presenter.callbacks is not None
    assert render_cluster.shank_presenter.callbacks is not None
    assert coordinator_cluster.load_data_coordinator.callbacks is not None
    assert coordinator_cluster.probe_selection_coordinator.callbacks is not None
    assert coordinator_cluster.session_selection_coordinator.callbacks is not None
    assert coordinator_cluster.mouse_root_coordinator.callbacks is not None
    assert coordinator_cluster.output_path_coordinator.commands is commands.paths
    assert coordinator_cluster.path_dialog_coordinator.callbacks.active_mouse_root is (
        queries.workspace.active_mouse_root_path
    )
    assert coordinator_cluster.output_folder_prompt.callbacks.has_output_directory is (
        queries.workspace.has_output_directory
    )
    assert coordinator_cluster.load_preflight_coordinator.can_load_data is (
        commands.load.can_load_data
    )
    assert (
        coordinator_cluster.selection_activation_coordinator.session_selection_coordinator
        is coordinator_cluster.session_selection_coordinator
    )
    assert (
        coordinator_cluster.selection_activation_coordinator.probe_selection_coordinator
        is coordinator_cluster.probe_selection_coordinator
    )
    assert (
        coordinator_cluster.selection_activation_coordinator.shank_selection_actions
        is render_cluster.shank_selection_actions
    )
    assert (
        coordinator_cluster.selection_activation_coordinator.load_preflight_coordinator
        is coordinator_cluster.load_preflight_coordinator
    )
    assert render_cluster.alignment_edit_actions.commands is commands.edit
    assert render_cluster.alignment_edit_actions.callbacks.tip_position_um() == 42.0
    assert render_cluster.shank_selection_actions.app is app
    assert render_cluster.shank_selection_actions.selection_view is not None
    assert render_cluster.alignment_selection_actions.app is app
    assert (
        render_cluster.alignment_presenter.callbacks.render_histology_alignment.__self__
        is render_cluster.histology_presenter
    )
    assert (
        render_cluster.alignment_presenter.callbacks.plot_channels.__self__
        is render_cluster.slice_panel_presenter
    )
    assert render_cluster.shank_presenter.callbacks.render_alignment_choices is (
        ports.render.shank.render_alignment_choices
    )
    assert coordinator_cluster.save_coordinator.callbacks.use_docdb is (
        ports.save.use_docdb
    )
    previous_alignment_callbacks = (
        coordinator_cluster.previous_alignment_load_coordinator.callbacks
    )
    assert previous_alignment_callbacks.use_docdb is (
        ports.previous_alignment_load.use_docdb
    )
    assert previous_alignment_callbacks.default_folder is (
        queries.workspace.active_output_package_directory
    )
    assert (
        coordinator_cluster.session_selection_coordinator.callbacks.capture_pending_reference_lines.__self__
        is render_cluster.reference_line_presenter
    )
    assert workbench.displays.ephys is ephys_display
    assert workbench.displays.slice is slice_display
    assert workbench.displays.reference_lines is reference_line_display
    assert reference_line_display.lines_changed_callback is not None
    reference_line_display.lines_changed_callback()
    assert captured_reference_lines == [([1.0], [2.0])]
    line_state = SimpleNamespace(
        feature_positions_um=[3.0],
        raw_track_positions_um=[4.0],
    )
    render_cluster.alignment_presenter.callbacks.clear_reference_lines()
    render_cluster.alignment_presenter.callbacks.render_reference_lines_from_alignment(
        line_state
    )
    render_cluster.shank_presenter.callbacks.clear_reference_lines()
    assert reference_line_display.clear_count == 2
    assert reference_line_display.replaced_lines == []
    assert reference_line_display.raw_replaced_lines == [([3.0], [4.0])]
    assert reference_line_display.reattach_count == 1
    assert (
        render_cluster.shank_presenter.callbacks.render_ephys_plots.__self__
        is render_cluster.ephys_plot_presenter
    )
    assert (
        render_cluster.shank_presenter.callbacks.render_histology_plots.__self__
        is render_cluster.histology_refresh_presenter
    )
    assert reference_line_display.created_lines == []
    assert (
        render_cluster.shank_presenter.callbacks.restore_slice_selection.__self__
        is render_cluster.slice_menu_coordinator
    )
    assert coordinator_cluster.plot_exporter.ephys_exporter.presenter is (
        render_cluster.ephys_plot_presenter
    )
    assert coordinator_cluster.plot_exporter.ephys_exporter.panel is ephys_display.panel
    slice_handles = coordinator_cluster.plot_exporter.slice_handles
    assert slice_handles.slice_display is slice_display
    assert slice_handles.slice_panel_presenter is (render_cluster.slice_panel_presenter)
    assert slice_handles.slice_menu_coordinator is (
        render_cluster.slice_menu_coordinator
    )
    coordinator_cluster.plot_exporter.add_lines_points()
    assert reference_line_display.add_count == 1
    assert coordinator_cluster.plot_exporter.callbacks.set_axis is ports.export.set_axis
    assert coordinator_cluster.plot_exporter.ephys_exporter.callbacks.set_view is (
        ports.export.set_view
    )
    assert coordinator_cluster.interaction_coordinator.popup_manager is (
        ports.interaction.popup_manager
    )
    assert coordinator_cluster.interaction_coordinator.reference_line_display is (
        reference_line_display
    )
    assert coordinator_cluster.interaction_coordinator.callbacks.set_axis is (
        ports.interaction.set_axis
    )
