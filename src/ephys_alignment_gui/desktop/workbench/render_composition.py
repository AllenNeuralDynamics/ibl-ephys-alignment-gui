"""Desktop render cluster composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.actions.alignment_edit_actions import (
    DesktopAlignmentEditActionCallbacks,
    DesktopAlignmentEditActions,
)
from ephys_alignment_gui.desktop.actions.alignment_selection_actions import (
    DesktopAlignmentSelectionActions,
    DesktopAlignmentSelectionCallbacks,
)
from ephys_alignment_gui.desktop.actions.display_actions import DesktopDisplayActions
from ephys_alignment_gui.desktop.actions.shank_selection_actions import (
    DesktopShankSelectionActions,
)
from ephys_alignment_gui.desktop.coordinators.slice_menu_coordinator import (
    DesktopSliceMenuCoordinator,
)
from ephys_alignment_gui.desktop.displays import DesktopDisplays
from ephys_alignment_gui.desktop.presenters.alignment_presenter import (
    DesktopAlignmentPresenter,
    DesktopAlignmentRenderCallbacks,
)
from ephys_alignment_gui.desktop.presenters.ephys_plot_presenter import (
    DesktopEphysPlotPresenter,
    EphysPlotRenderCallbacks,
)
from ephys_alignment_gui.desktop.presenters.histology_presenter import (
    DesktopHistologyPresenter,
    DesktopHistologyRenderCallbacks,
)
from ephys_alignment_gui.desktop.presenters.histology_refresh_presenter import (
    DesktopHistologyRefreshPresenter,
)
from ephys_alignment_gui.desktop.presenters.reference_line_presenter import (
    DesktopReferenceLinePresenter,
)
from ephys_alignment_gui.desktop.presenters.shank_presenter import (
    DesktopShankPresenter,
    DesktopShankRenderCallbacks,
)
from ephys_alignment_gui.desktop.presenters.slice_panel_presenter import (
    SlicePanelPresenter,
)
from ephys_alignment_gui.desktop.views import DesktopViews


@dataclass(frozen=True)
class DesktopRenderCluster:
    """Focused desktop presenters/actions for render and alignment editing."""

    alignment_presenter: DesktopAlignmentPresenter
    ephys_plot_presenter: DesktopEphysPlotPresenter
    histology_presenter: DesktopHistologyPresenter
    slice_panel_presenter: SlicePanelPresenter
    slice_menu_coordinator: DesktopSliceMenuCoordinator
    shank_presenter: DesktopShankPresenter
    reference_line_presenter: DesktopReferenceLinePresenter
    histology_refresh_presenter: DesktopHistologyRefreshPresenter
    alignment_edit_actions: DesktopAlignmentEditActions
    display_actions: DesktopDisplayActions
    shank_selection_actions: DesktopShankSelectionActions
    alignment_selection_actions: DesktopAlignmentSelectionActions


def build_desktop_render_cluster(
    *,
    app: Any,
    views: DesktopViews,
    displays: DesktopDisplays,
    ports: Any,
) -> DesktopRenderCluster:
    """Build the desktop render/edit cluster."""
    ephys_plot_presenter = DesktopEphysPlotPresenter(
        app=app,
        callbacks=EphysPlotRenderCallbacks(
            raw_image_payloads=views.shank_screen.raw_image_payload_mapping,
            image_raster_request=displays.ephys.panel.image_raster_request,
            render_image=displays.ephys.panel.render_image,
            render_scatter=displays.ephys.panel.render_scatter,
            render_line=displays.ephys.panel.render_line,
            render_probe=displays.ephys.panel.render_probe,
        ),
    )
    slice_panel_presenter = SlicePanelPresenter(
        app=app,
        view=displays.slice.view,
    )
    slice_menu_coordinator = DesktopSliceMenuCoordinator.create(
        app=app,
        panel=slice_panel_presenter,
    )
    histology_presenter = DesktopHistologyPresenter(
        app=app,
        panel=displays.histology.panel,
        callbacks=DesktopHistologyRenderCallbacks(
            probe_extent_query_kwargs=lambda: _probe_extent_query_kwargs(app),
            fit_depth_um=app.queries.workspace.fit_depth_um,
            lin_fit_enabled=app.queries.workspace.linear_fit_enabled,
            scale_factor_y_range=displays.ephys.panel.feature_y_range,
        ),
    )
    alignment_presenter = DesktopAlignmentPresenter(app.events)
    alignment_presenter.configure(
        queries=app.queries,
        callbacks=_alignment_render_callbacks(
            app,
            ports.render.alignment,
            views,
            displays,
            histology_presenter,
            slice_panel_presenter,
            slice_menu_coordinator,
        ),
    )
    histology_refresh_presenter = DesktopHistologyRefreshPresenter(
        app=app,
        histology_presenter=histology_presenter,
        slice_panel_presenter=slice_panel_presenter,
        slice_menu_coordinator=slice_menu_coordinator,
        reference_line_display=displays.reference_lines,
    )
    shank_presenter = DesktopShankPresenter(app)
    shank_presenter.configure(
        callbacks=_shank_render_callbacks(
            ports.render.shank,
            views,
            displays,
            ephys_plot_presenter,
            slice_menu_coordinator,
            histology_refresh_presenter,
        )
    )
    reference_line_presenter = DesktopReferenceLinePresenter(
        app=app,
        reference_line_display=displays.reference_lines,
    )
    displays.reference_lines.set_lines_changed_callback(
        reference_line_presenter.capture_pending_reference_lines
    )
    alignment_edit_actions = DesktopAlignmentEditActions(
        commands=app.commands.edit,
        callbacks=DesktopAlignmentEditActionCallbacks(
            histology_available=ports.alignment_edit_actions.histology_available,
            capture_pending_reference_lines=(
                reference_line_presenter.capture_pending_reference_lines
            ),
            tip_position_um=ports.alignment_edit_actions.tip_position_um,
        ),
    )
    display_actions = DesktopDisplayActions(
        app=app,
        displays=displays,
        histology_presenter=histology_presenter,
        slice_panel_presenter=slice_panel_presenter,
        alignment_screen=views.alignment_screen,
        fit_alignment=alignment_edit_actions.fit_button_pressed,
        histology_available=ports.alignment_edit_actions.histology_available,
    )
    shank_selection_actions = DesktopShankSelectionActions(
        app=app,
        selection_view=views.selection,
        reference_line_display=displays.reference_lines,
    )
    alignment_selection_actions = DesktopAlignmentSelectionActions(
        app=app,
        callbacks=DesktopAlignmentSelectionCallbacks(
            render_loaded_shank_histology=(
                histology_refresh_presenter.render_loaded_shank_histology
            )
        ),
    )
    return DesktopRenderCluster(
        alignment_presenter=alignment_presenter,
        ephys_plot_presenter=ephys_plot_presenter,
        histology_presenter=histology_presenter,
        slice_panel_presenter=slice_panel_presenter,
        slice_menu_coordinator=slice_menu_coordinator,
        shank_presenter=shank_presenter,
        reference_line_presenter=reference_line_presenter,
        histology_refresh_presenter=histology_refresh_presenter,
        alignment_edit_actions=alignment_edit_actions,
        display_actions=display_actions,
        shank_selection_actions=shank_selection_actions,
        alignment_selection_actions=alignment_selection_actions,
    )


def _alignment_render_callbacks(
    app: Any,
    ports: Any,
    views: DesktopViews,
    displays: DesktopDisplays,
    histology_presenter: DesktopHistologyPresenter,
    slice_panel_presenter: SlicePanelPresenter,
    slice_menu_coordinator: DesktopSliceMenuCoordinator,
) -> DesktopAlignmentRenderCallbacks:
    """Build callbacks for alignment edit rendering."""
    return DesktopAlignmentRenderCallbacks(
        restore_lin_fit=lambda lin_fit: _restore_lin_fit_from_edit(
            app,
            views,
            lin_fit,
        ),
        clear_reference_lines=displays.reference_lines.clear,
        capture_depth_plot_y_ranges=ports.capture_depth_plot_y_ranges,
        restore_depth_plot_y_ranges=ports.restore_depth_plot_y_ranges,
        reattach_reference_lines=displays.reference_lines.reattach,
        render_histology_alignment=histology_presenter.render_alignment_edit,
        plot_channels=slice_panel_presenter.plot_channels,
        refresh_perpendicular_histology=(
            lambda: slice_panel_presenter.refresh_perpendicular_histology(
                slice_menu_coordinator.current_selection()
            )
        ),
        update_reference_lines_to_alignment=(
            displays.reference_lines.sync_track_to_feature
        ),
        create_reference_lines_for_previous_alignment=lambda: (
            _create_reference_lines_for_previous_alignment(app, views)
        ),
        set_default_feature_y_range=lambda: _set_default_feature_y_range(app, views),
        update_status=lambda: _update_alignment_status(app, views),
    )


def _restore_lin_fit_from_edit(
    app: Any,
    views: DesktopViews,
    lin_fit: bool | None,
) -> None:
    """Restore app display state and desktop checkbox for an applied edit."""
    if lin_fit is None:
        return
    app.commands.display.set_linear_fit_enabled(lin_fit)
    views.alignment_screen.set_linear_fit_checked(
        app.queries.workspace.linear_fit_enabled()
    )


def _create_reference_lines_for_previous_alignment(
    app: Any,
    views: DesktopViews,
) -> None:
    """Create previous-alignment reference lines from an app read model."""
    state = app.queries.workspace.active_alignment_edit_screen_state()
    views.alignment_screen.create_reference_lines_for_previous_alignment(state)


def _set_default_feature_y_range(app: Any, views: DesktopViews) -> None:
    """Apply the app-derived default feature y-range to desktop plots."""
    views.alignment_screen.set_default_feature_y_range(
        depth_view=app.queries.workspace.depth_view_settings(),
        in_brain_depths_um=app.queries.ephys.active_in_brain_depths_um(),
    )


def _update_alignment_status(app: Any, views: DesktopViews) -> None:
    """Update alignment edit status labels from an app read model."""
    views.alignment_screen.update_status(
        app.queries.workspace.active_alignment_edit_screen_state()
    )


def _shank_render_callbacks(
    ports: Any,
    views: DesktopViews,
    displays: DesktopDisplays,
    ephys_plot_presenter: DesktopEphysPlotPresenter,
    slice_menu_coordinator: DesktopSliceMenuCoordinator,
    histology_refresh_presenter: DesktopHistologyRefreshPresenter,
) -> DesktopShankRenderCallbacks:
    """Build callbacks for shank selection rendering."""
    return DesktopShankRenderCallbacks(
        capture_plot_selection=lambda preserve: ports.capture_plot_selection(
            preserve,
            ephys_plot_presenter=ephys_plot_presenter,
            slice_menu_coordinator=slice_menu_coordinator,
        ),
        clear_reference_lines=displays.reference_lines.clear,
        render_alignment_choices=ports.render_alignment_choices,
        apply_plot_data_state=ports.apply_plot_data_state,
        raw_image_payloads=ports.raw_image_payloads,
        render_plot_menus=lambda state: ports.render_plot_menus(
            state,
            ephys_plot_presenter=ephys_plot_presenter,
        ),
        render_ephys_plots=ephys_plot_presenter.render_shank_ephys_plots,
        render_histology_plots=(
            histology_refresh_presenter.render_loaded_shank_histology
        ),
        restore_slice_selection=slice_menu_coordinator.restore_selection,
        configure_view=ports.configure_view,
        offline=ports.offline,
    )


def _probe_extent_query_kwargs(app: Any) -> dict[str, float]:
    """Return probe extent settings needed for histology-panel queries."""
    depth_view = app.queries.workspace.depth_view_settings()
    return {
        "probe_tip_um": depth_view.probe_tip_um,
        "probe_top_um": depth_view.probe_top_um,
        "probe_extra_um": depth_view.probe_extra_um,
    }
