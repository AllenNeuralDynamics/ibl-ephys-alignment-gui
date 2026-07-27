"""Desktop view wrapper for session/probe/shank selection widgets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from PyQt5 import QtGui, QtWidgets


@dataclass
class DesktopSelectionView:
    """Own Qt widget operations for session/probe/shank selection controls."""

    session_model: Any
    session_combobox: Any
    probe_model: Any
    probe_combobox: Any
    shank_model: Any
    shank_combobox: Any
    load_data_button: Any
    item_factory: Callable[[str], Any] = QtGui.QStandardItem

    def current_session(self) -> str:
        """Return the selected recording/session label."""
        return self.session_combobox.currentText()

    def current_probe(self) -> str:
        """Return the selected probe label."""
        return self.probe_combobox.currentText()

    def selection_widgets(self) -> list[Any]:
        """Widgets disabled while probe metadata is loading."""
        return [self.probe_combobox, self.session_combobox]

    def populate_sessions(self, sessions: Sequence[str]) -> None:
        """Render session choices."""
        self._populate(sessions, self.session_model, self.session_combobox)

    def populate_probes(self, probes: Sequence[str]) -> None:
        """Render probe choices."""
        self._populate(probes, self.probe_model, self.probe_combobox)

    def populate_probe_shanks(self, shanks: Sequence[str]) -> None:
        """Render shank choices before a stream is loaded."""
        self._populate(shanks, self.shank_model, self.shank_combobox)

    def populate_loaded_shanks(self, shanks: Sequence[str], target_shank: int) -> None:
        """Render shank choices and select the active loaded shank."""
        self.populate_probe_shanks(shanks)
        self.shank_combobox.setCurrentIndex(target_shank)

    def clear_probes(self) -> None:
        """Clear probe choices."""
        self.probe_model.clear()

    def clear_shanks(self) -> None:
        """Clear shank choices."""
        self.shank_model.clear()

    def select_session_index(self, idx: int) -> None:
        """Select a session by combobox index."""
        self.session_combobox.setCurrentIndex(idx)

    def select_probe_index(self, idx: int) -> None:
        """Select a probe by combobox index."""
        self.probe_combobox.setCurrentIndex(idx)

    def set_load_data_enabled(self, enabled: bool) -> None:
        """Enable or disable the Load Data button."""
        self.load_data_button.setEnabled(enabled)

    def load_data_widget(self) -> Any:
        """Return the Load Data button widget for busy contexts."""
        return self.load_data_button

    def _populate(self, values: Sequence[str], model: Any, combobox: Any) -> None:
        model.clear()
        labels = [str(value) for value in values]
        for label in labels:
            item = self.item_factory(label)
            item.setEditable(False)
            model.appendRow(item)

        if not labels:
            return

        metrics = combobox.fontMetrics()
        width = getattr(metrics, "horizontalAdvance", None)
        if width is None:
            width = metrics.width
        min_width = width(max(labels, key=len))
        min_width += combobox.view().autoScrollMargin()
        min_width += combobox.style().pixelMetric(
            QtWidgets.QStyle.PM_ScrollBarExtent
        )
        combobox.view().setMinimumWidth(min_width)
