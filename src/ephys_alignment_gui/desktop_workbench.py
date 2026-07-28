"""Desktop composition shell for focused presenters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ephys_alignment_gui.desktop_alignment_presenter import (
    DesktopAlignmentPresenter,
    DesktopAlignmentRenderCallbacks,
)
from ephys_alignment_gui.desktop_histology_presenter import (
    DesktopHistologyPresenter,
    DesktopHistologyRenderCallbacks,
)
from ephys_alignment_gui.desktop_load_data_presenter import (
    DesktopLoadDataCallbacks,
    DesktopLoadDataPresenter,
)
from ephys_alignment_gui.desktop_mouse_root_presenter import (
    DesktopMouseRootCallbacks,
    DesktopMouseRootPresenter,
)
from ephys_alignment_gui.desktop_probe_selection_presenter import (
    DesktopProbeSelectionCallbacks,
    DesktopProbeSelectionPresenter,
)
from ephys_alignment_gui.desktop_session_selection_presenter import (
    DesktopSessionSelectionCallbacks,
    DesktopSessionSelectionPresenter,
)
from ephys_alignment_gui.desktop_shank_presenter import (
    DesktopShankPresenter,
    DesktopShankRenderCallbacks,
)
from ephys_alignment_gui.event_bus import EventSubscription

AlignmentCallbacksFactory = Callable[
    [DesktopHistologyPresenter],
    DesktopAlignmentRenderCallbacks,
]
ProbeSelectionCallbacksFactory = Callable[
    [DesktopLoadDataPresenter],
    DesktopProbeSelectionCallbacks,
]


@dataclass
class DesktopWorkbench:
    """Own focused desktop presenters and desktop event subscription lifecycle."""

    app: Any
    alignment_presenter: DesktopAlignmentPresenter
    shank_presenter: DesktopShankPresenter
    histology_presenter: DesktopHistologyPresenter
    load_data_presenter: DesktopLoadDataPresenter
    probe_selection_presenter: DesktopProbeSelectionPresenter
    session_selection_presenter: DesktopSessionSelectionPresenter
    mouse_root_presenter: DesktopMouseRootPresenter
    _event_subscriptions: list[EventSubscription] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        app: Any,
        selection_view: Any,
        path_view: Any,
        histology_panel: Any,
        histology_callbacks: DesktopHistologyRenderCallbacks,
        alignment_callbacks_factory: AlignmentCallbacksFactory,
        shank_callbacks: DesktopShankRenderCallbacks,
        load_data_callbacks: DesktopLoadDataCallbacks,
        probe_selection_callbacks_factory: ProbeSelectionCallbacksFactory,
        session_selection_callbacks: DesktopSessionSelectionCallbacks,
        mouse_root_callbacks: DesktopMouseRootCallbacks,
    ) -> DesktopWorkbench:
        """Build and configure the focused desktop presenters."""
        histology_presenter = DesktopHistologyPresenter(
            app=app,
            panel=histology_panel,
            callbacks=histology_callbacks,
        )
        alignment_presenter = DesktopAlignmentPresenter(app.events)
        alignment_presenter.configure(
            queries=app.queries,
            callbacks=alignment_callbacks_factory(histology_presenter),
        )
        shank_presenter = DesktopShankPresenter(app)
        shank_presenter.configure(callbacks=shank_callbacks)
        load_data_presenter = DesktopLoadDataPresenter(
            app=app,
            selection_view=selection_view,
            callbacks=load_data_callbacks,
        )
        probe_selection_presenter = DesktopProbeSelectionPresenter(
            commands=app.commands,
            selection_view=selection_view,
            callbacks=probe_selection_callbacks_factory(load_data_presenter),
        )
        session_selection_presenter = DesktopSessionSelectionPresenter(
            commands=app.commands,
            selection_view=selection_view,
            callbacks=session_selection_callbacks,
        )
        mouse_root_presenter = DesktopMouseRootPresenter(
            commands=app.commands,
            path_view=path_view,
            selection_view=selection_view,
            callbacks=mouse_root_callbacks,
        )
        return cls(
            app=app,
            alignment_presenter=alignment_presenter,
            shank_presenter=shank_presenter,
            histology_presenter=histology_presenter,
            load_data_presenter=load_data_presenter,
            probe_selection_presenter=probe_selection_presenter,
            session_selection_presenter=session_selection_presenter,
            mouse_root_presenter=mouse_root_presenter,
        )

    def connect_events(self) -> list[EventSubscription]:
        """Subscribe desktop presenters to semantic app events."""
        if self._event_subscriptions:
            return list(self._event_subscriptions)
        self._event_subscriptions.extend(
            self.alignment_presenter.connect_alignment_events()
        )
        self._event_subscriptions.extend(self.shank_presenter.connect_shank_events())
        return list(self._event_subscriptions)

    def disconnect_events(self) -> None:
        """Disconnect desktop event subscriptions."""
        for subscription in self._event_subscriptions:
            subscription.disconnect()
        self._event_subscriptions.clear()

    def render_loaded_shank(
        self,
        *,
        shank_idx: int,
        preserve_plot_selection: bool | None = None,
    ) -> None:
        """Render the loaded desktop view for one active shank."""
        self.shank_presenter.render_loaded_shank(
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
        return self.histology_presenter.render_active_aligned(fig, movable=movable)

    def render_active_reference_histology(
        self,
        fig: Any | None = None,
        *,
        movable: bool = False,
    ) -> bool:
        """Render the active reference histology panel."""
        return self.histology_presenter.render_active_reference(fig, movable=movable)

    def render_active_scale_factor(self) -> bool:
        """Render the active scale-factor panel."""
        return self.histology_presenter.render_active_scale_factor()

    def render_active_fit(self) -> bool:
        """Render the active feature/track fit panel."""
        return self.histology_presenter.render_active_fit()

    def render_active_histology_panels(self) -> bool:
        """Render reference histology, aligned histology, scale, and fit panels."""
        return self.histology_presenter.render_active_panels()

    def load_heavy_data(self) -> bool:
        """Load or activate the selected stream/shank for desktop display."""
        return self.load_data_presenter.load_heavy_data()

    def set_mouse_root(self, mouse_root: Any) -> bool:
        """Load a mouse-root datapackage through the desktop presenter."""
        return self.mouse_root_presenter.set_mouse_root(mouse_root)

    def mouse_root_edited(self) -> bool:
        """Handle direct text edits to the mouse-root line edit."""
        return self.mouse_root_presenter.mouse_root_edited()

    def session_selected(self) -> bool:
        """Select the current recording/session from the desktop widgets."""
        return self.session_selection_presenter.session_selected()

    def probe_selected(self) -> bool:
        """Select the current probe from the desktop widgets."""
        return self.probe_selection_presenter.probe_selected()
