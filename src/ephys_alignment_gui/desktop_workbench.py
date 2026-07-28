"""Desktop composition shell for focused presenters."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ephys_alignment_gui.desktop_alignment_presenter import (
    DesktopAlignmentPresenter,
    DesktopAlignmentRenderCallbacks,
)
from ephys_alignment_gui.desktop_folder_dialog import DesktopFolderDialog
from ephys_alignment_gui.desktop_histology_presenter import (
    DesktopHistologyPresenter,
    DesktopHistologyRenderCallbacks,
)
from ephys_alignment_gui.desktop_load_data_presenter import (
    DesktopLoadDataCallbacks,
    DesktopLoadDataPresenter,
)
from ephys_alignment_gui.desktop_load_workflow_presenter import (
    DesktopLoadWorkflowPresenter,
    DesktopOutputFolderPrompt,
    OutputFolderPromptCallbacks,
)
from ephys_alignment_gui.desktop_mouse_root_presenter import (
    DesktopMouseRootCallbacks,
    DesktopMouseRootPresenter,
)
from ephys_alignment_gui.desktop_output_path_presenter import DesktopOutputPathPresenter
from ephys_alignment_gui.desktop_path_dialog_presenter import (
    DesktopPathDialogCallbacks,
    DesktopPathDialogPresenter,
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


@dataclass(frozen=True)
class DesktopSelectionWorkflowCallbacks:
    """MainWindow bridge callbacks for selection and load presenters."""

    capture_pending_reference_lines: Callable[[], None]
    stash_and_detach_current: Callable[[], None]
    teardown_session: Callable[[], None]
    init_session_variables: Callable[[], None]
    select_shank_for_view: Callable[[int, str], int | None]
    setup_session_view: Callable[[bool | None, int], None]
    clear_empty_state: Callable[[], None]
    set_histology_available: Callable[[bool], None]
    busy_context: Callable[..., AbstractContextManager[Any]]
    mouse_root_loaded: Callable[[], bool]
    active_shank_idx: Callable[[], int]
    show_empty_state: Callable[[], None]
    evict_stream_cache: Callable[[], None]
    clear_histology_context: Callable[[], None]
    select_first_session: Callable[[], None]
    select_first_probe: Callable[[], None]


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
    output_path_presenter: DesktopOutputPathPresenter
    path_dialog_presenter: DesktopPathDialogPresenter
    load_workflow_presenter: DesktopLoadWorkflowPresenter
    output_folder_prompt: DesktopOutputFolderPrompt
    folder_dialog: DesktopFolderDialog
    _event_subscriptions: list[EventSubscription] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        app: Any,
        selection_view: Any,
        path_view: Any,
        parent: Any,
        histology_panel: Any,
        histology_callbacks: DesktopHistologyRenderCallbacks,
        alignment_callbacks_factory: AlignmentCallbacksFactory,
        shank_callbacks: DesktopShankRenderCallbacks,
        selection_callbacks: DesktopSelectionWorkflowCallbacks,
    ) -> DesktopWorkbench:
        """Build and configure the focused desktop presenters."""
        histology_presenter = DesktopHistologyPresenter(
            app=app,
            panel=histology_panel,
            callbacks=histology_callbacks,
        )
        output_path_presenter = DesktopOutputPathPresenter(
            commands=app.commands,
            path_view=path_view,
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
            callbacks=cls._load_data_callbacks(
                selection_callbacks,
                output_path_presenter,
            ),
        )
        probe_selection_presenter = DesktopProbeSelectionPresenter(
            commands=app.commands,
            selection_view=selection_view,
            callbacks=cls._probe_selection_callbacks(
                selection_callbacks,
                output_path_presenter,
                load_data_presenter,
            ),
        )
        session_selection_presenter = DesktopSessionSelectionPresenter(
            commands=app.commands,
            selection_view=selection_view,
            callbacks=cls._session_selection_callbacks(selection_callbacks),
        )
        mouse_root_presenter = DesktopMouseRootPresenter(
            commands=app.commands,
            path_view=path_view,
            selection_view=selection_view,
            callbacks=cls._mouse_root_callbacks(selection_callbacks),
        )
        folder_dialog = DesktopFolderDialog(parent=None)
        path_dialog_presenter = DesktopPathDialogPresenter(
            folder_dialog=folder_dialog,
            callbacks=DesktopPathDialogCallbacks(
                active_mouse_root=app.queries.active_mouse_root_path,
                set_mouse_root=mouse_root_presenter.set_mouse_root,
                active_output_root=app.queries.active_output_root,
                set_save_root=output_path_presenter.set_save_root,
            ),
        )
        output_folder_prompt = DesktopOutputFolderPrompt(
            parent=parent,
            callbacks=OutputFolderPromptCallbacks(
                derive_output_directory_from_save_root=(
                    output_path_presenter.derive_output_directory_from_save_root
                ),
                has_output_directory=app.queries.has_output_directory,
                select_output_folder=path_dialog_presenter.select_output_root,
            ),
        )
        load_workflow_presenter = DesktopLoadWorkflowPresenter(
            can_load_data=app.commands.can_load_data,
            load_heavy_data=load_data_presenter.load_heavy_data,
            output_folder_prompt=output_folder_prompt,
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
            output_path_presenter=output_path_presenter,
            path_dialog_presenter=path_dialog_presenter,
            load_workflow_presenter=load_workflow_presenter,
            output_folder_prompt=output_folder_prompt,
            folder_dialog=folder_dialog,
        )

    @staticmethod
    def _load_data_callbacks(
        callbacks: DesktopSelectionWorkflowCallbacks,
        output_path_presenter: DesktopOutputPathPresenter,
    ) -> DesktopLoadDataCallbacks:
        """Build callbacks for cached/fresh data loading."""
        return DesktopLoadDataCallbacks(
            capture_pending_reference_lines=callbacks.capture_pending_reference_lines,
            stash_and_detach_current=callbacks.stash_and_detach_current,
            teardown_session=callbacks.teardown_session,
            init_session_variables=callbacks.init_session_variables,
            select_shank_for_view=callbacks.select_shank_for_view,
            display_output_directory=output_path_presenter.display_output_directory,
            setup_session_view=callbacks.setup_session_view,
            clear_empty_state=callbacks.clear_empty_state,
            set_histology_available=callbacks.set_histology_available,
            busy_context=callbacks.busy_context,
        )

    @staticmethod
    def _probe_selection_callbacks(
        callbacks: DesktopSelectionWorkflowCallbacks,
        output_path_presenter: DesktopOutputPathPresenter,
        load_data_presenter: DesktopLoadDataPresenter,
    ) -> DesktopProbeSelectionCallbacks:
        """Build callbacks for probe selection."""
        return DesktopProbeSelectionCallbacks(
            mouse_root_loaded=callbacks.mouse_root_loaded,
            active_shank_idx=callbacks.active_shank_idx,
            capture_pending_reference_lines=callbacks.capture_pending_reference_lines,
            stash_and_detach_current=callbacks.stash_and_detach_current,
            present_cached_probe_selection=(
                lambda session, probe, shank: (
                    load_data_presenter.present_cached_probe_selection(
                        session_name=session,
                        probe_name=probe,
                        target_shank=shank,
                    )
                )
            ),
            show_empty_state=callbacks.show_empty_state,
            busy_context=callbacks.busy_context,
            init_session_variables=callbacks.init_session_variables,
            select_shank_for_view=callbacks.select_shank_for_view,
            display_output_directory=output_path_presenter.display_output_directory,
        )

    @staticmethod
    def _session_selection_callbacks(
        callbacks: DesktopSelectionWorkflowCallbacks,
    ) -> DesktopSessionSelectionCallbacks:
        """Build callbacks for session selection."""
        return DesktopSessionSelectionCallbacks(
            mouse_root_loaded=callbacks.mouse_root_loaded,
            capture_pending_reference_lines=callbacks.capture_pending_reference_lines,
            evict_stream_cache=callbacks.evict_stream_cache,
            show_empty_state=callbacks.show_empty_state,
            select_first_probe=callbacks.select_first_probe,
        )

    @staticmethod
    def _mouse_root_callbacks(
        callbacks: DesktopSelectionWorkflowCallbacks,
    ) -> DesktopMouseRootCallbacks:
        """Build callbacks for mouse-root loading."""
        return DesktopMouseRootCallbacks(
            clear_histology_context=callbacks.clear_histology_context,
            busy_context=callbacks.busy_context,
            select_first_session=callbacks.select_first_session,
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

    def load_data_button_pressed(self) -> bool:
        """Run desktop load workflow policy and load data when allowed."""
        return self.load_workflow_presenter.load_data_button_pressed()

    def ensure_output_directory_for_save(self, requirement: Any | None = None) -> bool:
        """Require a save location before writing alignment outputs."""
        return self.output_folder_prompt.ensure_for_save(requirement)

    def set_save_root(self, save_root: Path) -> bool:
        """Set the save-root directory. Per-probe output lands under it."""
        return self.output_path_presenter.set_save_root(save_root)

    def select_mouse_root(self) -> bool:
        """Prompt for a mouse-root directory."""
        return self.path_dialog_presenter.select_mouse_root()

    def select_output_root(self) -> bool:
        """Prompt for a save-root directory."""
        return self.path_dialog_presenter.select_output_root()

    def output_folder_edited(self) -> bool:
        """Handle direct edits to the output-folder text field."""
        return self.output_path_presenter.output_folder_edited()

    def log_load_requirement(self, requirement: Any) -> None:
        """Log a load workflow requirement that has no desktop prompt action."""
        self.load_workflow_presenter.log_requirement(requirement)

    def select_existing_directory_text(self, title: str) -> str:
        """Prompt for an existing directory and return Qt-style text."""
        return self.folder_dialog.select_existing_directory_text(title)
