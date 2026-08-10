"""Tests for desktop popup lifecycle ownership."""

from __future__ import annotations

from ephys_alignment_gui.desktop.popup_manager import DesktopPopupManager


class FakeSignal:
    def __init__(self) -> None:
        self.disconnects = 0

    def disconnect(self) -> None:
        self.disconnects += 1


class FakePopup:
    def __init__(self) -> None:
        self.closed = FakeSignal()
        self.moved = FakeSignal()
        self.blocked = False
        self.closed_count = 0
        self.normal_count = 0
        self.minimized_count = 0

    def blockSignals(self, blocked: bool) -> None:
        self.blocked = blocked

    def close(self) -> None:
        self.closed_count += 1

    def showNormal(self) -> None:
        self.normal_count += 1

    def showMinimized(self) -> None:
        self.minimized_count += 1


def test_cluster_popups_can_be_tracked_removed_and_closed() -> None:
    manager = DesktopPopupManager()
    first = FakePopup()
    second = FakePopup()

    manager.add_cluster_popup(first)
    manager.add_cluster_popup(second)
    manager.remove_cluster_popup(first)
    manager.close_cluster_popups()

    assert manager.cluster_popups == []
    assert second.blocked
    assert second.closed_count == 1
    assert second.closed.disconnects == 1
    assert second.moved.disconnects == 1


def test_cluster_popups_toggle_minimized_state() -> None:
    manager = DesktopPopupManager()
    popup = FakePopup()
    manager.add_cluster_popup(popup)

    assert manager.toggle_cluster_minimized() is False
    assert popup.minimized_count == 1
    assert manager.toggle_cluster_minimized() is True
    assert popup.normal_count == 1


def test_close_all_closes_named_popups() -> None:
    manager = DesktopPopupManager()
    label = FakePopup()
    notes = FakePopup()
    manager.label_window = label
    manager.notes_window = notes
    manager.nearby_table = object()

    manager.close_all()

    assert manager.label_window is None
    assert manager.notes_window is None
    assert manager.nearby_table is None
    assert label.closed_count == 1
    assert notes.closed_count == 1
