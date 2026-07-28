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
from ephys_alignment_gui.desktop_shank_presenter import (
    DesktopShankPresenter,
    DesktopShankRenderCallbacks,
)
from ephys_alignment_gui.event_bus import EventSubscription

AlignmentCallbacksFactory = Callable[
    [DesktopHistologyPresenter],
    DesktopAlignmentRenderCallbacks,
]


@dataclass
class DesktopWorkbench:
    """Own focused desktop presenters and desktop event subscription lifecycle."""

    app: Any
    alignment_presenter: DesktopAlignmentPresenter
    shank_presenter: DesktopShankPresenter
    histology_presenter: DesktopHistologyPresenter
    _event_subscriptions: list[EventSubscription] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        app: Any,
        histology_panel: Any,
        histology_callbacks: DesktopHistologyRenderCallbacks,
        alignment_callbacks_factory: AlignmentCallbacksFactory,
        shank_callbacks: DesktopShankRenderCallbacks,
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
        return cls(
            app=app,
            alignment_presenter=alignment_presenter,
            shank_presenter=shank_presenter,
            histology_presenter=histology_presenter,
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
