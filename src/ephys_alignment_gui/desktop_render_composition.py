"""Desktop render cluster composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ephys_alignment_gui.desktop_alignment_edit_actions import (
    DesktopAlignmentEditActionCallbacks,
    DesktopAlignmentEditActions,
)
from ephys_alignment_gui.desktop_alignment_presenter import (
    DesktopAlignmentPresenter,
    DesktopAlignmentRenderCallbacks,
)
from ephys_alignment_gui.desktop_alignment_selection_actions import (
    DesktopAlignmentSelectionActions,
    DesktopAlignmentSelectionCallbacks,
)
from ephys_alignment_gui.desktop_histology_refresh_presenter import (
    DesktopHistologyRefreshPresenter,
)
from ephys_alignment_gui.desktop_reference_line_presenter import (
    DesktopReferenceLinePresenter,
)
from ephys_alignment_gui.desktop_shank_presenter import (
    DesktopShankPresenter,
    DesktopShankRenderCallbacks,
)
from ephys_alignment_gui.desktop_shank_selection_actions import (
    DesktopShankSelectionActions,
)
from ephys_alignment_gui.desktop_views import DesktopViews


@dataclass(frozen=True)
class DesktopRenderCluster:
    """Focused desktop presenters/actions for render and alignment editing."""

    alignment_presenter: DesktopAlignmentPresenter
    shank_presenter: DesktopShankPresenter
    reference_line_presenter: DesktopReferenceLinePresenter
    histology_refresh_presenter: DesktopHistologyRefreshPresenter
    alignment_edit_actions: DesktopAlignmentEditActions
    shank_selection_actions: DesktopShankSelectionActions
    alignment_selection_actions: DesktopAlignmentSelectionActions


def build_desktop_render_cluster(
    *,
    app: Any,
    views: DesktopViews,
    ports: Any,
) -> DesktopRenderCluster:
    """Build the desktop render/edit cluster."""
    displays = views.displays
    alignment_presenter = DesktopAlignmentPresenter(app.events)
    alignment_presenter.configure(
        queries=app.queries,
        callbacks=_alignment_render_callbacks(
            ports.render.alignment,
            views,
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
        commands=app.commands,
        callbacks=DesktopAlignmentEditActionCallbacks(
            histology_available=ports.alignment_edit_actions.histology_available,
            capture_pending_reference_lines=(
                reference_line_presenter.capture_pending_reference_lines
            ),
            tip_position_um=ports.alignment_edit_actions.tip_position_um,
        ),
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
        shank_selection_actions=shank_selection_actions,
        alignment_selection_actions=alignment_selection_actions,
    )


def _alignment_render_callbacks(
    ports: Any,
    views: DesktopViews,
) -> DesktopAlignmentRenderCallbacks:
    """Build callbacks for alignment edit rendering."""
    displays = views.displays
    return DesktopAlignmentRenderCallbacks(
        restore_lin_fit=ports.restore_lin_fit,
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
        create_reference_lines_for_previous_alignment=(
            ports.create_reference_lines_for_previous_alignment
        ),
        set_default_feature_y_range=ports.set_default_feature_y_range,
        update_status=ports.update_status,
    )


def _shank_render_callbacks(
    ports: Any,
    views: DesktopViews,
    histology_refresh_presenter: DesktopHistologyRefreshPresenter,
) -> DesktopShankRenderCallbacks:
    """Build callbacks for shank selection rendering."""
    displays = views.displays
    return DesktopShankRenderCallbacks(
        capture_plot_selection=ports.capture_plot_selection,
        clear_reference_lines=displays.reference_lines.clear,
        render_alignment_choices=ports.render_alignment_choices,
        apply_plot_data_state=ports.apply_plot_data_state,
        raw_image_payloads=ports.raw_image_payloads,
        render_plot_menus=ports.render_plot_menus,
        render_ephys_plots=displays.ephys.render_shank_ephys_plots,
        render_histology_plots=(
            histology_refresh_presenter.render_loaded_shank_histology
        ),
        restore_slice_selection=displays.slice.restore_selection,
        configure_view=ports.configure_view,
        offline=ports.offline,
    )
