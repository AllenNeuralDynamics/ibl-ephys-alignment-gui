"""Qt callback adapter for the desktop shell."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt5 import QtCore

from ephys_alignment_gui.core.workflow import Requirement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesktopShellActions:
    """Own Qt callback methods that dispatch to the desktop workbench."""

    window: Any

    @property
    def workbench(self) -> Any:
        return self.window.desktop_workbench

    def load_existing_alignments(self, *_args: Any) -> bool:
        """Prompt for and load previously saved alignments."""
        return self.workbench.load_existing_alignments()

    def set_mouse_root(self, mouse_root: Path) -> bool:
        """Point the GUI at a preprocessed mouse-root directory."""
        return self.workbench.set_mouse_root(mouse_root)

    def on_mouse_root_selected(self, *_args: Any) -> bool:
        """Prompt for the mouse-root directory."""
        return self.workbench.select_mouse_root()

    def on_mouse_root_edited(self) -> bool:
        """Handle direct edits to the mouse-root line edit."""
        return self.workbench.mouse_root_edited()

    def on_subject_selected(self, _idx: int) -> None:
        """Report that legacy ONE/Alyx subject selection is unsupported."""
        self.window._show_one_unsupported("Subject selection")

    def on_session_selected(self, _idx: int) -> None:
        """Report that legacy ONE/Alyx session selection is unsupported."""
        self.window._show_one_unsupported("Session selection")

    def on_session_combobox_activated(self, idx: int) -> bool:
        """Select the session and load or activate its selected stream."""
        return self.workbench.session_selected(idx)

    def on_probe_combobox_activated(self, idx: int) -> bool:
        """Select the probe and load or activate its selected stream."""
        return self.workbench.probe_selected(idx)

    def on_use_docdb_changed(self, state: int) -> None:
        """Log DocDB checkbox state changes."""
        use_docdb = state == QtCore.Qt.Checked
        logger.info("Use DocDB: %s", use_docdb)

    def ensure_output_directory_for_save(
        self,
        requirement: Requirement | None = None,
    ) -> bool:
        """Require a save location before writing alignment outputs."""
        return self.workbench.ensure_output_directory_for_save(requirement)

    def set_save_root(self, save_root: Path) -> bool:
        """Set the save-root directory. Per-probe output lands under it."""
        return self.workbench.set_save_root(save_root)

    def on_output_folder_selected(self, *_args: Any) -> bool:
        """Prompt the user for a save-root directory."""
        return self.workbench.select_output_root()

    def on_output_folder_edited(self) -> bool:
        """Handle direct edits to the output-folder text field."""
        return self.workbench.output_folder_edited()

    def on_shank_selected(self, idx: int) -> bool:
        """Select the shank and load or activate its selected stream."""
        return self.workbench.shank_selected(idx)

    def on_alignment_selected(self, idx: int) -> bool:
        """Select the current previous/original alignment choice."""
        return self.workbench.alignment_selected(idx)

    def toggle_histology_button_pressed(self, *_args: Any) -> bool:
        """Toggle nearby/reference histology boundary display."""
        return self.workbench.toggle_histology_boundaries()

    def toggle_histology_map_button_pressed(self, *_args: Any) -> None:
        """Toggle region annotation source and refresh histology panels."""
        self.workbench.toggle_region_annotation_source()

    def fit_button_pressed(self, *_args: Any) -> bool:
        """Fit the active alignment from current desktop reference lines."""
        return self.workbench.fit_button_pressed()

    def offset_button_pressed(
        self,
        _checked: bool = False,
        *,
        track_shift_m: float = 0.0,
    ) -> bool:
        """Offset the active alignment from the desktop probe-tip line."""
        return self.workbench.offset_button_pressed(track_shift_m=track_shift_m)

    def movedown_button_pressed(self, *_args: Any) -> bool:
        """Nudge the active alignment down by one fixed step."""
        return self.workbench.movedown_button_pressed()

    def moveup_button_pressed(self, *_args: Any) -> bool:
        """Nudge the active alignment up by one fixed step."""
        return self.workbench.moveup_button_pressed()

    def toggle_labels_button_pressed(self, *_args: Any) -> None:
        """Toggle atlas label visibility on histology panels."""
        self.workbench.toggle_labels()

    def toggle_line_button_pressed(self, *_args: Any) -> None:
        """Toggle reference-line visibility on desktop plots."""
        self.workbench.toggle_reference_lines()

    def toggle_channel_button_pressed(self, *_args: Any) -> None:
        """Toggle channel overlays on slice panels."""
        self.workbench.toggle_channels()

    def delete_line_button_pressed(self, *_args: Any) -> None:
        """Delete the currently selected reference line."""
        self.workbench.delete_selected_reference_line()

    def describe_labels_pressed(self, *_args: Any) -> bool:
        """Show region information for the selected histology label."""
        return self.workbench.describe_labels_pressed()

    def label_closed(self, popup: Any) -> None:
        """Hide the label popup without forgetting reusable widgets."""
        self.workbench.label_closed(popup)

    def label_moved(self) -> None:
        """Bring the main window back to front after label popup movement."""
        self.workbench.label_moved()

    def label_pressed(self, item: Any) -> None:
        """Render region information for a clicked structure tree item."""
        self.workbench.label_pressed(item)

    def next_button_pressed(self, *_args: Any) -> bool:
        """Move the active alignment edit cursor forward."""
        return self.workbench.next_button_pressed()

    def prev_button_pressed(self, *_args: Any) -> bool:
        """Move the active alignment edit cursor backward."""
        return self.workbench.prev_button_pressed()

    def reset_button_pressed(self, *_args: Any) -> bool:
        """Reset the active alignment to initialized geometry."""
        return self.workbench.reset_button_pressed()

    def complete_button_pressed_offline(self, *_args: Any) -> bool:
        """Save edited alignment outputs."""
        return self.workbench.save_alignment_outputs()

    def display_qc_options(self, *_args: Any) -> bool:
        """Display alignment QC choices."""
        return self.workbench.display_qc_options()

    def qc_button_clicked(self) -> bool:
        """Handle the QC save button."""
        return self.workbench.qc_button_clicked()

    def selected_qc_descriptions(self) -> list[str]:
        """Return selected QC description labels."""
        ephys_desc = []
        if not hasattr(self.window, "desc_buttons"):
            return ephys_desc
        for button in self.window.desc_buttons.buttons():
            if button.isChecked():
                ephys_desc.append(button.text())
        return ephys_desc

    def reset_axis_button_pressed(self, *_args: Any) -> None:
        """Reset feature-depth y-range and feature image x-range."""
        self.workbench.reset_axis()

    def save_plots(self, *_args: Any) -> bool:
        """Save all desktop plot panels for the active shank."""
        return self.workbench.save_plots()

    def display_session_notes(self, *_args: Any) -> None:
        """Show session notes for the active stream."""
        self.workbench.display_session_notes()

    def display_nearby_sessions(self, *_args: Any) -> None:
        """Report that nearby sessions need unsupported ONE/Alyx online mode."""
        self.window._show_one_unsupported("Nearby sessions")

    def popup_closed(self, popup: Any) -> None:
        """Forget a closed cluster popup."""
        self.workbench.popup_closed(popup)

    def popup_moved(self) -> None:
        """Bring the main window back to front after popup movement."""
        self.workbench.popup_moved()

    def close_popups(self, *_args: Any) -> None:
        """Close cluster detail popups."""
        self.workbench.close_popups()

    def minimise_popups(self, *_args: Any) -> None:
        """Toggle cluster detail popups between minimized and normal."""
        self.workbench.minimise_popups()

    def lin_fit_option_changed(self, state: int) -> bool:
        """Set linear-fit option and recompute when reference lines exist."""
        return self.workbench.set_linear_fit_enabled(state != 0)

    def cluster_clicked(self, item: Any, point: Any) -> Any | None:
        """Open cluster detail popup for a clicked ephys cluster point."""
        return self.workbench.cluster_clicked(item, point)

    def on_mouse_double_clicked(self, event: Any) -> bool:
        """Add a reference line from a double-clicked feature plot position."""
        return self.workbench.on_mouse_double_clicked(event)

    def on_mouse_hover(self, items: list[Any]) -> None:
        """Dispatch hover interactions to reference-line and histology views."""
        self.workbench.on_mouse_hover(items)

    def tip_line_moved(self) -> None:
        """Keep histology top line synchronized to the current tip line."""
        self.workbench.sync_histology_top_to_tip()

    def top_line_moved(self) -> None:
        """Keep histology tip line synchronized to the current top line."""
        self.workbench.sync_histology_tip_to_top()
