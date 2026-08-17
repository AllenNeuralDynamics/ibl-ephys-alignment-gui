"""Desktop composition shell for focused coordinators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ephys_alignment_gui.core.event_bus import EventSubscription
from ephys_alignment_gui.desktop.displays import DesktopDisplays
from ephys_alignment_gui.desktop.views import DesktopViews
from ephys_alignment_gui.desktop.workbench.composition import (
    DesktopWorkbenchCoordinatorCluster,
    build_desktop_workbench_coordinator_cluster,
)
from ephys_alignment_gui.desktop.workbench.port_types import (
    DesktopWorkbenchPorts,
)
from ephys_alignment_gui.desktop.workbench.render_composition import (
    DesktopRenderCluster,
    build_desktop_render_cluster,
)


@dataclass
class DesktopWorkbench:
    """Own focused desktop coordinators and desktop event subscription lifecycle."""

    app: Any
    views: DesktopViews
    displays: DesktopDisplays
    render_cluster: DesktopRenderCluster
    coordinator_cluster: DesktopWorkbenchCoordinatorCluster
    _event_subscriptions: list[EventSubscription] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        app: Any,
        parent: Any,
        views: DesktopViews,
        displays: DesktopDisplays,
        ports: DesktopWorkbenchPorts,
    ) -> DesktopWorkbench:
        """Build and configure the focused desktop coordinators."""
        render_cluster = build_desktop_render_cluster(
            app=app,
            views=views,
            displays=displays,
            ports=ports,
        )
        coordinator_cluster = build_desktop_workbench_coordinator_cluster(
            app=app,
            parent=parent,
            views=views,
            displays=displays,
            ports=ports,
            render_cluster=render_cluster,
        )
        return cls(
            app=app,
            views=views,
            displays=displays,
            render_cluster=render_cluster,
            coordinator_cluster=coordinator_cluster,
        )

    def connect_events(self) -> list[EventSubscription]:
        """Subscribe desktop coordinators to semantic app events."""
        if self._event_subscriptions:
            return list(self._event_subscriptions)
        self._event_subscriptions.extend(
            self.render_cluster.alignment_presenter.connect_alignment_events()
        )
        self._event_subscriptions.extend(
            self.render_cluster.shank_presenter.connect_shank_events()
        )
        self._event_subscriptions.extend(
            self.render_cluster.display_actions.connect_display_events()
        )
        self._event_subscriptions.extend(
            self.coordinator_cluster.load_data_coordinator.connect_load_events()
        )
        self._event_subscriptions.extend(
            self.coordinator_cluster.output_path_coordinator.connect_path_events()
        )
        self._event_subscriptions.extend(
            self.coordinator_cluster.lifecycle_coordinator.connect_lifecycle_events()
        )
        self._event_subscriptions.extend(
            self.coordinator_cluster.save_coordinator.connect_save_events()
        )
        self._event_subscriptions.extend(
            self.coordinator_cluster.previous_alignment_load_coordinator.connect_previous_alignment_events()
        )
        return list(self._event_subscriptions)

    def disconnect_events(self) -> None:
        """Disconnect desktop event subscriptions."""
        for subscription in self._event_subscriptions:
            subscription.disconnect()
        self._event_subscriptions.clear()

    def shutdown(self, *, timeout_ms: int = 5000) -> bool:
        """Settle foreground desktop work before the workbench is torn down."""
        stopped = self.coordinator_cluster.load_data_coordinator.shutdown_active_load(
            timeout_ms=timeout_ms,
        )
        if not stopped:
            return False
        stopped = self.coordinator_cluster.save_coordinator.shutdown_active_save(
            timeout_ms=timeout_ms,
        )
        if not stopped:
            return False
        self.disconnect_events()
        return True

    def initialize_startup_stream_state(self) -> None:
        """Initialize stream-dependent app and desktop state at startup."""
        self.coordinator_cluster.lifecycle_coordinator.initialize_startup_stream_state()

    def initialize_region_lookup(self, init_region_lookup: Any) -> None:
        """Populate desktop region lookup widgets from app atlas metadata."""
        self.coordinator_cluster.interaction_coordinator.initialize_region_lookup(
            init_region_lookup
        )

    def attach_plot_menus(self, menu_bar: Any, *, parent: Any, offline: bool) -> None:
        """Attach desktop plot menus without exposing render-cluster internals."""
        self.render_cluster.ephys_plot_presenter.attach_plot_menus(menu_bar)
        self.render_cluster.slice_menu_coordinator.attach_menu(
            menu_bar,
            parent=parent,
            offline=offline,
        )
        self.render_cluster.ephys_plot_presenter.attach_unit_filter_menu(
            menu_bar,
            parent,
        )

    def toggle_ephys_plot(self, menu: Any, *, reverse: bool = False) -> None:
        """Toggle the selected ephys plot menu."""
        self.render_cluster.ephys_plot_presenter.toggle_plot(menu, reverse=reverse)

    def toggle_slice_plot(self, *, reverse: bool = False) -> None:
        """Toggle the selected slice plot menu."""
        self.render_cluster.slice_menu_coordinator.toggle_plot(reverse=reverse)

    def set_ephys_view(self, view: int) -> None:
        """Apply one of the desktop shank-screen ephys layouts."""
        self.views.shank_screen.set_view(view=view)

    def render_loaded_shank(
        self,
        *,
        shank_idx: int,
        preserve_plot_selection: bool | None = None,
    ) -> None:
        """Render the loaded desktop view for one active shank."""
        self.render_cluster.shank_presenter.render_loaded_shank(
            shank_idx=shank_idx,
            preserve_plot_selection=preserve_plot_selection,
        )

    def render_active_aligned_histology(
        self,
        fig: Any | None = None,
        *,
        movable: bool = True,
    ) -> bool:
        """Render the active aligned histology panel."""
        return self.render_cluster.histology_presenter.render_active_aligned(
            fig,
            movable=movable,
        )

    def render_active_reference_histology(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render the active reference histology panel."""
        return self.render_cluster.histology_presenter.render_active_reference(
            fig,
            movable=movable,
        )

    def render_active_nearby_histology(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render the active nearby-boundary histology panel."""
        return self.render_cluster.histology_presenter.render_active_nearby(
            fig,
            movable=movable,
        )

    def render_active_scale_factor(self) -> bool:
        """Render the active scale-factor panel."""
        return self.render_cluster.histology_presenter.render_active_scale_factor()

    def render_active_fit(self) -> bool:
        """Render the active feature/track fit panel."""
        return self.render_cluster.histology_presenter.render_active_fit()

    def render_active_histology_panels(self) -> bool:
        """Render reference histology, aligned histology, scale, and fit panels."""
        return self.render_cluster.histology_presenter.render_active_panels()

    def render_loaded_shank_histology(self, shank_idx: int | None = None) -> bool:
        """Render loaded-shank histology, perpendicular slice, and line overlays."""
        coordinator = self.render_cluster.histology_refresh_presenter
        return coordinator.render_loaded_shank_histology(shank_idx)

    def load_heavy_data(self) -> bool:
        """Load or activate the selected stream/shank for desktop display."""
        return self.coordinator_cluster.load_data_coordinator.load_heavy_data()

    def set_mouse_root(self, mouse_root: Any) -> bool:
        """Load a mouse-root datapackage through the desktop coordinator."""
        return self.coordinator_cluster.mouse_root_coordinator.set_mouse_root(
            mouse_root
        )

    def mouse_root_edited(self) -> bool:
        """Handle direct text edits to the mouse-root line edit."""
        return self.coordinator_cluster.mouse_root_coordinator.mouse_root_edited()

    def session_selected(self, idx: int | None = None) -> bool:
        """Select and load or activate the current recording/session."""
        return (
            self.coordinator_cluster.selection_activation_coordinator.session_selected(
                idx
            )
        )

    def probe_selected(self, idx: int | None = None) -> bool:
        """Select and load or activate the current probe."""
        return self.coordinator_cluster.selection_activation_coordinator.probe_selected(
            idx
        )

    def shank_selected(self, _idx: int | None = None) -> bool:
        """Select and load or activate the current shank."""
        return self.coordinator_cluster.selection_activation_coordinator.shank_selected(
            _idx
        )

    def alignment_selected(self, idx: int) -> bool:
        """Select the current previous/original alignment choice."""
        return self.render_cluster.alignment_selection_actions.alignment_selected(idx)

    def activate_selected_stream(self) -> bool:
        """Load or activate the selected stream/shank through the shared path."""
        return self.coordinator_cluster.selection_activation_coordinator.activate_selected_stream()

    def fit_button_pressed(self) -> bool:
        """Fit the active alignment from current desktop reference lines."""
        return self.render_cluster.alignment_edit_actions.fit_button_pressed()

    def offset_button_pressed(self, *, track_shift_m: float = 0.0) -> bool:
        """Offset the active alignment from the desktop probe-tip line."""
        return self.render_cluster.alignment_edit_actions.offset_button_pressed(
            track_shift_m=track_shift_m
        )

    def movedown_button_pressed(self) -> bool:
        """Nudge the active alignment down by one fixed step."""
        return self.render_cluster.alignment_edit_actions.movedown_button_pressed()

    def moveup_button_pressed(self) -> bool:
        """Nudge the active alignment up by one fixed step."""
        return self.render_cluster.alignment_edit_actions.moveup_button_pressed()

    def next_button_pressed(self) -> bool:
        """Move the active alignment edit cursor forward."""
        return self.render_cluster.alignment_edit_actions.next_button_pressed()

    def prev_button_pressed(self) -> bool:
        """Move the active alignment edit cursor backward."""
        return self.render_cluster.alignment_edit_actions.prev_button_pressed()

    def reset_button_pressed(self) -> bool:
        """Reset the active alignment to initialized geometry."""
        return self.render_cluster.alignment_edit_actions.reset_button_pressed()

    def toggle_histology_boundaries(self) -> bool:
        """Toggle nearby/reference histology boundary display."""
        return self.render_cluster.display_actions.toggle_histology_boundaries()

    def toggle_region_annotation_source(self) -> None:
        """Toggle region annotation source and refresh histology panels."""
        self.render_cluster.display_actions.toggle_region_annotation_source()

    def toggle_labels(self) -> None:
        """Toggle atlas label visibility on histology panels."""
        self.render_cluster.display_actions.toggle_labels()

    def toggle_reference_lines(self) -> None:
        """Toggle reference-line visibility on desktop plots."""
        self.render_cluster.display_actions.toggle_reference_lines()

    def toggle_channels(self) -> None:
        """Toggle channel overlays on slice panels."""
        self.render_cluster.display_actions.toggle_channels()

    def delete_selected_reference_line(self) -> None:
        """Delete the currently selected reference line."""
        self.render_cluster.display_actions.delete_selected_reference_line()

    def reset_axis(self) -> None:
        """Reset feature-depth y-range and feature image x-range."""
        self.render_cluster.display_actions.reset_axis()

    def set_linear_fit_enabled(self, enabled: bool) -> bool:
        """Set linear-fit option and recompute when reference lines exist."""
        return self.render_cluster.display_actions.set_linear_fit_enabled(enabled)

    def sync_histology_top_to_tip(self) -> None:
        """Keep histology top line synchronized to the current tip line."""
        self.render_cluster.display_actions.sync_histology_top_to_tip()

    def sync_histology_tip_to_top(self) -> None:
        """Keep histology tip line synchronized to the current top line."""
        self.render_cluster.display_actions.sync_histology_tip_to_top()

    def ensure_output_directory_for_save(self, requirement: Any | None = None) -> bool:
        """Require a save location before writing alignment outputs."""
        return self.coordinator_cluster.output_folder_prompt.ensure_for_save(
            requirement
        )

    def set_save_root(self, save_root: Path) -> bool:
        """Set the save-root directory. Per-probe output lands under it."""
        return self.coordinator_cluster.output_path_coordinator.set_save_root(save_root)

    def select_mouse_root(self) -> bool:
        """Prompt for a mouse-root directory."""
        return self.coordinator_cluster.path_dialog_coordinator.select_mouse_root()

    def select_output_root(self) -> bool:
        """Prompt for a save-root directory."""
        return self.coordinator_cluster.path_dialog_coordinator.select_output_root()

    def output_folder_edited(self) -> bool:
        """Handle direct edits to the output-folder text field."""
        return self.coordinator_cluster.output_path_coordinator.output_folder_edited()

    def log_load_requirement(self, requirement: Any) -> None:
        """Log a load policy requirement that has no desktop prompt action."""
        self.coordinator_cluster.load_preflight_coordinator.log_requirement(requirement)

    def select_existing_directory_text(self, title: str) -> str:
        """Prompt for an existing directory and return Qt-style text."""
        return self.coordinator_cluster.folder_dialog.select_existing_directory_text(
            title
        )

    def load_existing_alignments(self) -> bool:
        """Prompt for and load previous alignments."""
        coordinator = self.coordinator_cluster.previous_alignment_load_coordinator
        return coordinator.load_existing_alignments()

    def save_alignment_outputs(self) -> bool:
        """Save edited alignment outputs."""
        return self.coordinator_cluster.save_coordinator.save_alignment_outputs()

    def display_qc_options(self) -> bool:
        """Display alignment QC choices."""
        return self.coordinator_cluster.save_coordinator.display_qc_options()

    def qc_button_clicked(self) -> bool:
        """Handle the QC save button."""
        return self.coordinator_cluster.save_coordinator.qc_button_clicked()

    def export_plots(self, output_dir: Path, *, sess_info: str = "") -> None:
        """Export all desktop plot panels for the active shank."""
        self.coordinator_cluster.plot_exporter.export(output_dir, sess_info=sess_info)

    def save_plots(self, save_path: Any = None) -> bool:
        """Save all desktop plot panels to an explicit or app-derived folder."""
        return self.coordinator_cluster.plot_export_coordinator.save_plots(save_path)

    def display_session_notes(self) -> None:
        """Show session notes for the active stream."""
        self.coordinator_cluster.interaction_coordinator.display_session_notes()

    def popup_closed(self, popup: Any) -> None:
        """Forget a closed cluster popup."""
        self.coordinator_cluster.interaction_coordinator.popup_closed(popup)

    def popup_moved(self) -> None:
        """Bring the main window back to front after popup movement."""
        self.coordinator_cluster.interaction_coordinator.popup_moved()

    def close_popups(self) -> None:
        """Close cluster detail popups."""
        self.coordinator_cluster.interaction_coordinator.close_popups()

    def minimise_popups(self) -> None:
        """Toggle cluster detail popups between minimized and normal."""
        self.coordinator_cluster.interaction_coordinator.minimise_popups()

    def cluster_clicked(self, item: Any, point: Any) -> Any | None:
        """Open cluster detail popup for a clicked ephys cluster point."""
        return self.coordinator_cluster.interaction_coordinator.cluster_clicked(
            item, point
        )

    def describe_labels_pressed(self) -> bool:
        """Show region information for the selected histology label."""
        return (
            self.coordinator_cluster.interaction_coordinator.describe_labels_pressed()
        )

    def label_closed(self, popup: Any) -> None:
        """Hide the label popup without forgetting reusable widgets."""
        self.coordinator_cluster.interaction_coordinator.label_closed(popup)

    def label_moved(self) -> None:
        """Bring the main window back to front after label popup movement."""
        self.coordinator_cluster.interaction_coordinator.label_moved()

    def label_pressed(self, item: Any) -> None:
        """Render region information for a clicked structure tree item."""
        self.coordinator_cluster.interaction_coordinator.label_pressed(item)

    def on_mouse_double_clicked(self, event: Any) -> bool:
        """Add a reference line from a double-clicked feature plot position."""
        return self.coordinator_cluster.interaction_coordinator.on_mouse_double_clicked(
            event
        )

    def on_mouse_hover(self, items: list[Any]) -> None:
        """Dispatch hover interactions to reference-line and histology views."""
        self.coordinator_cluster.interaction_coordinator.on_mouse_hover(items)
