"""Tests for desktop selection widget wrapper."""

from __future__ import annotations

from ephys_alignment_gui.desktop.views.selection_view import DesktopSelectionView


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

    def item(self, idx: int) -> FakeItem | None:
        try:
            return self.rows[idx]
        except IndexError:
            return None

    def rowCount(self) -> int:
        return len(self.rows)


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


def _view() -> tuple[DesktopSelectionView, FakeModel, FakeCombobox]:
    session_model = FakeModel()
    session_combobox = FakeCombobox("rec1")
    shank_combobox = FakeCombobox("2/4")
    view = DesktopSelectionView(
        session_model=session_model,
        session_combobox=session_combobox,
        probe_model=FakeModel(),
        probe_combobox=FakeCombobox("probeA"),
        shank_model=FakeModel(),
        shank_combobox=shank_combobox,
        item_factory=FakeItem,
    )
    return view, session_model, session_combobox


def test_selection_view_populates_models_and_sizes_combobox_popup() -> None:
    view, model, combobox = _view()

    view.populate_sessions(["short", "much-longer"])

    assert [item.text for item in model.rows] == ["short", "much-longer"]
    assert [item.editable for item in model.rows] == [False, False]
    assert combobox.popup.minimum_width == 115
    assert combobox.current_index == 0


def test_selection_view_reads_session_and_probe_labels_by_index() -> None:
    view, session_model, _combobox = _view()
    view.populate_sessions(["rec1", "rec2"])
    view.populate_probes(["probeA", "probeB"])

    assert view.session_at_index(1) == "rec2"
    assert view.probe_at_index(1) == "probeB"
    assert view.session_at_index(-1) is None
    assert view.probe_at_index(2) is None
    assert [item.text for item in session_model.rows] == ["rec1", "rec2"]


def test_selection_view_handles_empty_population_without_sizing() -> None:
    view, model, combobox = _view()

    view.populate_sessions([])

    assert model.rows == []
    assert model.clear_calls == 1
    assert combobox.popup.minimum_width is None


def test_selection_view_wraps_current_values() -> None:
    view, _model, _combobox = _view()

    assert view.current_session() == "rec1"
    assert view.current_probe() == "probeA"
    assert view.current_shank_index() == 1
    assert view.selection_widgets() == [
        view.session_combobox,
        view.probe_combobox,
        view.shank_combobox,
    ]


def test_selection_view_returns_none_for_invalid_shank_label() -> None:
    view, _model, _combobox = _view()
    view.shank_combobox.text = "not-a-shank"

    assert view.current_shank_index() is None


def test_selection_view_selects_session_and_probe_by_label() -> None:
    view, _model, combobox = _view()
    view.populate_sessions(["rec1", "rec2"])
    view.populate_probes(["probeA", "probeB"])

    assert view.select_session_text("rec2") == 1
    assert combobox.current_index == 1
    assert view.select_probe_text("probeB") == 1
    assert view.probe_combobox.current_index == 1
    assert view.select_session_text("missing") is None

    view.select_shank_index(2)

    assert view.shank_combobox.current_index == 2
