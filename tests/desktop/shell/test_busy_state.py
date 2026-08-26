"""Tests for shared desktop busy-state ownership."""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt

from ephys_alignment_gui.desktop.shell.busy_state import BusyStateManager


class FakeStatusBar:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def showMessage(self, message: str, timeout: int | None = None) -> None:
        self.calls.append(("show", message, timeout))

    def clearMessage(self) -> None:
        self.calls.append(("clear",))


class FakeWindow:
    def __init__(self) -> None:
        self.status_bar = FakeStatusBar()

    def statusBar(self) -> FakeStatusBar:
        return self.status_bar


class FakeWidget:
    def __init__(
        self,
        *,
        enabled: bool = True,
        parent: FakeWidget | None = None,
    ) -> None:
        self.locally_enabled = enabled
        self.parent = parent
        self.enabled_calls: list[bool] = []

    def isEnabled(self) -> bool:
        return self.locally_enabled and (
            self.parent is None or self.parent.isEnabled()
        )

    def setEnabled(self, enabled: bool) -> None:
        self.locally_enabled = enabled
        self.enabled_calls.append(enabled)

    def testAttribute(self, attribute: Any) -> bool:
        assert attribute == Qt.WA_ForceDisabled
        return not self.locally_enabled


def _manager() -> tuple[BusyStateManager, FakeWindow, list[str]]:
    window = FakeWindow()
    cursor_calls: list[str] = []
    manager = BusyStateManager(
        window,
        set_wait_cursor=lambda: cursor_calls.append("set"),
        restore_wait_cursor=lambda: cursor_calls.append("restore"),
    )
    return manager, window, cursor_calls


def test_overlapping_leases_restore_widget_after_last_out_of_order_release() -> None:
    manager, window, cursor_calls = _manager()
    widget = FakeWidget()
    recovery = manager.context(
        "Recovering autosave...",
        "Autosave recovered",
        disable_widgets=[widget],
    )
    load = manager.context(
        "Loading heavy data...",
        "Data loaded",
        disable_widgets=[widget],
    )

    recovery.__enter__()
    load.__enter__()
    recovery.__exit__(None, None, None)

    assert not widget.isEnabled()
    assert cursor_calls == ["set"]
    assert window.status_bar.calls[-1] == ("show", "Loading heavy data...", None)

    load.__exit__(None, None, None)

    assert widget.isEnabled()
    assert widget.enabled_calls == [False, True]
    assert cursor_calls == ["set", "restore"]
    assert window.status_bar.calls[-1] == ("show", "Data loaded", 3000)


def test_parent_busy_state_does_not_make_child_desired_state_false() -> None:
    manager, _window, _cursor_calls = _manager()
    parent = FakeWidget()
    child = FakeWidget(parent=parent)
    save = manager.context(disable_widgets=parent)
    load = manager.context(disable_widgets=child)

    save.__enter__()
    assert not child.isEnabled()
    load.__enter__()
    save.__exit__(None, None, None)
    assert not child.isEnabled()

    load.__exit__(None, None, None)

    assert parent.isEnabled()
    assert child.isEnabled()
    assert child.enabled_calls == [False, True]


def test_busy_state_preserves_a_widget_that_was_locally_disabled() -> None:
    manager, _window, cursor_calls = _manager()
    widget = FakeWidget(enabled=False)

    with manager.context(disable_widgets=[widget, widget]):
        assert not widget.isEnabled()

    assert not widget.isEnabled()
    assert widget.enabled_calls == [False, False]
    assert cursor_calls == ["set", "restore"]


def test_only_top_active_message_is_rendered_and_can_be_updated() -> None:
    manager, window, _cursor_calls = _manager()
    first = manager.context("First")
    second = manager.context("Second")

    first.__enter__()
    second.__enter__()
    first.update_message("Updated first")
    assert window.status_bar.calls[-1] == ("show", "Second", None)

    second.__exit__(None, None, None)
    assert window.status_bar.calls[-1] == ("show", "Updated first", None)

    first.__exit__(None, None, None)
    assert window.status_bar.calls[-1] == ("clear",)
