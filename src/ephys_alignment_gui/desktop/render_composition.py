"""Desktop render cluster composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop.alignment_edit_actions import (
    DesktopAlignmentEditActionCallbacks,
    DesktopAlignmentEditActions,
)
from ephys_alignment_gui.desktop.alignment_presenter import (
    DesktopAlignmentPresenter,
    DesktopAlignmentRenderCallbacks,
)
from ephys_alignment_gui.desktop.alignment_selection_actions import (
    DesktopAlignmentSelectionActions,
    DesktopAlignmentSelectionCallbacks,
)
from ephys_alignment_gui.desktop.display_actions import DesktopDisplayActions
from ephys_alignment_gui.desktop.displays import DesktopDisplays
from ephys_alignment_gui.desktop.histology_refresh_presenter import (
    DesktopHistologyRefreshPresenter,
)
from ephys_alignment_gui.desktop.reference_line_presenter import (
    DesktopReferenceLinePresenter,
)
from ephys_alignment_gui.desktop.shank_presenter import (
    DesktopShankPresenter,
    DesktopShankRenderCallbacks,
)
from ephys_alignment_gui.desktop.shank_selection_actions import (
    DesktopShankSelectionActions,
)
from ephys_alignment_gui.desktop.views import DesktopViews


@dataclass(frozen=True)
class DesktopRenderCluster:
    """Focused desktop presenters/actions for render and alignment editing."""

    alignment_presenter: DesktopAlignmentPresenter
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
    alignment_presenter = DesktopAlignmentPresenter(app.events)
    alignment_presenter.configure(
        queries=app.queries,
        callbacks=_alignment_render_callbacks(
            app,
            ports.render.alignment,
            views,
            displays,
        ),
    )
    histology_refresh_presenter = DesktopHistologyRefreshPresenter(
        app=app,
        histology_display=displays.histology,
        slice_display=displays.slice,
        reference_line_display=displays.reference_lines,
    )
    shank_presenter = DesktopShankPresenter(app)
    shank_presenter.configure(
        callbacks=_shank_render_callbacks(
            ports.render.shank,
            views,
            displays,
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
        render_histology_alignment=displays.histology.render_alignment_edit,
        plot_channels=displays.slice.plot_channels,
        refresh_perpendicular_histology=(
            displays.slice.refresh_perpendicular_histology
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
    histology_refresh_presenter: DesktopHistologyRefreshPresenter,
) -> DesktopShankRenderCallbacks:
    """Build callbacks for shank selection rendering."""
    return DesktopShankRenderCallbacks(
        capture_plot_selection=lambda preserve: ports.capture_plot_selection(
            preserve,
        ),
        clear_reference_lines=displays.reference_lines.clear,
        render_alignment_choices=ports.render_alignment_choices,
        apply_plot_data_state=ports.apply_plot_data_state,
        raw_image_payloads=ports.raw_image_payloads,
        render_plot_menus=lambda state: ports.render_plot_menus(state),
        render_ephys_plots=displays.ephys.render_shank_ephys_plots,
        render_histology_plots=(
            histology_refresh_presenter.render_loaded_shank_histology
        ),
        restore_slice_selection=displays.slice.restore_selection,
        configure_view=ports.configure_view,
        offline=ports.offline,
    )
