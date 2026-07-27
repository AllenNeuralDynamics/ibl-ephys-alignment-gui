"""Tests for desktop selection widget wrapper."""

from __future__ import annotations

from ephys_alignment_gui.desktop_selection_view import DesktopSelectionView


class FakeItem:
    def __init__(self, text: str) -> None:
        self.text = text
        self.editable: bool | None = None

    def setEditable(self, editable: bool) -> None:
        self.editable = editable


class FakeModel:
    def __init__(self) -> None:
        self.rows: list[FakeItem] = []
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1
        self.rows = []

    def appendRow(self, item: FakeItem) -> None:
        self.rows.append(item)


class FakeMetrics:
    def width(self, text: str) -> int:
        return len(text) * 10


class FakePopup:
    def __init__(self) -> None:
        self.minimum_width: int | None = None

    def autoScrollMargin(self) -> int:
        return 2

    def setMinimumWidth(self, width: int) -> None:
        self.minimum_width = width


class FakeStyle:
    def pixelMetric(self, _metric) -> int:
        return 3


class FakeCombobox:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.current_index: int | None = None
        self.popup = FakePopup()

    def currentText(self) -> str:
        return self.text

    def setCurrentIndex(self, idx: int) -> None:
        self.current_index = idx

    def fontMetrics(self) -> FakeMetrics:
        return FakeMetrics()

    def view(self) -> FakePopup:
        return self.popup

    def style(self) -> FakeStyle:
        return FakeStyle()


class FakeButton:
    def __init__(self) -> None:
        self.enabled: bool | None = None

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


def _view() -> tuple[DesktopSelectionView, FakeModel, FakeCombobox, FakeButton]:
    session_model = FakeModel()
    session_combobox = FakeCombobox("rec1")
    button = FakeButton()
    view = DesktopSelectionView(
        session_model=session_model,
        session_combobox=session_combobox,
        probe_model=FakeModel(),
        probe_combobox=FakeCombobox("probeA"),
        shank_model=FakeModel(),
        shank_combobox=FakeCombobox(),
        load_data_button=button,
        item_factory=FakeItem,
    )
    return view, session_model, session_combobox, button


def test_selection_view_populates_models_and_sizes_combobox_popup() -> None:
    view, model, combobox, _button = _view()

    view.populate_sessions(["short", "much-longer"])

    assert [item.text for item in model.rows] == ["short", "much-longer"]
    assert [item.editable for item in model.rows] == [False, False]
    assert combobox.popup.minimum_width == 115


def test_selection_view_handles_empty_population_without_sizing() -> None:
    view, model, combobox, _button = _view()

    view.populate_sessions([])

    assert model.rows == []
    assert model.clear_calls == 1
    assert combobox.popup.minimum_width is None


def test_selection_view_wraps_current_values_and_load_button() -> None:
    view, _model, _combobox, button = _view()

    view.set_load_data_enabled(True)

    assert view.current_session() == "rec1"
    assert view.current_probe() == "probeA"
    assert view.load_data_widget() is button
    assert button.enabled is True
