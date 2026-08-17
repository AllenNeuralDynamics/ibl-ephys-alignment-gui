"""Desktop view wrapper for session/probe/shank selection widgets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from PyQt5 import QtGui

from ephys_alignment_gui.desktop.views.choice_list import populate_choice_list


@dataclass
class DesktopSelectionView:
    """Own Qt widget operations for session/probe/shank selection controls."""

    session_model: Any
    session_combobox: Any
    probe_model: Any
    probe_combobox: Any
    shank_model: Any
    shank_combobox: Any
    item_factory: Callable[[str], Any] = QtGui.QStandardItem

    def current_session(self) -> str:
        """Return the selected recording/session label."""
        return self.session_combobox.currentText()

    def current_probe(self) -> str:
        """Return the selected probe label."""
        return self.probe_combobox.currentText()

    def session_at_index(self, idx: int) -> str | None:
        """Return the session label at a combobox/model index."""
        return self._text_at_index(self.session_model, idx)

    def probe_at_index(self, idx: int) -> str | None:
        """Return the probe label at a combobox/model index."""
        return self._text_at_index(self.probe_model, idx)

    def current_shank_index(self) -> int | None:
        """Return the selected zero-based shank index, if the label is valid."""
        text = self.shank_combobox.currentText()
        try:
            shank_id = int(str(text).split("/")[0])
        except (TypeError, ValueError):
            return None
        return shank_id - 1

    def selection_widgets(self) -> list[Any]:
        """Widgets disabled while selected stream state is changing."""
        return [self.session_combobox, self.probe_combobox, self.shank_combobox]

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

    def _populate(self, values: Sequence[str], model: Any, combobox: Any) -> None:
        populate_choice_list(
            values,
            model,
            combobox,
            item_factory=self.item_factory,
        )

    @staticmethod
    def _text_at_index(model: Any, idx: int) -> str | None:
        if idx < 0:
            return None
        try:
            item = model.item(idx)
        except AttributeError:
            return None
        if item is None:
            return None
        text = getattr(item, "text", None)
        if callable(text):
            return text()
        if text is not None:
            return str(text)
        return None
