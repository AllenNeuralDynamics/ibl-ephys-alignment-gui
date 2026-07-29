"""Tests for desktop popup and mouse-interaction presentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ephys_alignment_gui.desktop_interaction_presenter import (
    DesktopInteractionCallbacks,
    DesktopInteractionPresenter,
    DesktopInteractionWidgets,
)
from ephys_alignment_gui.desktop_popup_manager import DesktopPopupManager


class FakeSignal:
    def __init__(self) -> None:
        self.connected: list[Any] = []

    def connect(self, callback: Any) -> None:
        self.connected.append(callback)


class FakeLayout:
    def __init__(self) -> None:
        self.widgets: list[Any] = []
        self.stretches: list[tuple[int, int]] = []

    def addWidget(self, widget: Any) -> None:
        self.widgets.append(widget)

    def setRowStretch(self, row: int, stretch: int) -> None:
        self.stretches.append((row, stretch))


class FakePopupWidget:
    def __init__(self) -> None:
        self.items: list[tuple[Any, int, int]] = []

    def addItem(self, item: Any, row: int, column: int) -> None:
        self.items.append((item, row, column))


class FakePopupWindow:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = FakeSignal()
        self.moved = FakeSignal()
        self.layout = FakeLayout()
        self.popup_widget = FakePopupWidget()
        self.show_calls = 0
        self.hide_calls = 0

    def show(self) -> None:
        self.show_calls += 1

    def hide(self) -> None:
        self.hide_calls += 1


class FakeTextEdit:
    def __init__(self) -> None:
        self.read_only = False
        self.wrap_mode = None
        self.text = ""

    def setReadOnly(self, read_only: bool) -> None:
        self.read_only = read_only

    def setLineWrapMode(self, mode: Any) -> None:
        self.wrap_mode = mode

    def setText(self, text: str) -> None:
        self.text = text


class FakePlotItem:
    def __init__(self) -> None:
        self.x_range = None
        self.y_range = None
        self.items: list[Any] = []

    def setXRange(self, **kwargs: Any) -> None:
        self.x_range = kwargs

    def setYRange(self, **kwargs: Any) -> None:
        self.y_range = kwargs

    def addItem(self, item: Any) -> None:
        self.items.append(item)


class FakeBarGraphItem:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakePlotCurveItem:
    def __init__(self) -> None:
        self.data = None

    def setData(self, **kwargs: Any) -> None:
        self.data = kwargs


class FakeEphysPanel:
    def __init__(self) -> None:
        self.cluster_idx = 3
        self.feature_y_um = 125.0
        self.cluster_x_calls: list[float] = []
        self.scene_pos_calls: list[Any] = []

    def cluster_index_for_plot_x(self, x_position: float) -> int | None:
        self.cluster_x_calls.append(x_position)
        return self.cluster_idx

    def feature_y_from_scene(self, scene_pos: Any) -> float | None:
        self.scene_pos_calls.append(scene_pos)
        return self.feature_y_um


class FakeHistologyPanel:
    def __init__(self) -> None:
        self.selected_idx: int | None = 1
        self.scale_factor = 1.234
        self.scale_factor_calls: list[Any] = []
        self.selected_regions: list[Any] = []

    def selected_region_index(self) -> int | None:
        return self.selected_idx

    def scale_factor_for_region_item(self, item: Any) -> float | None:
        self.scale_factor_calls.append(item)
        return self.scale_factor

    def select_region(self, item: Any) -> None:
        self.selected_regions.append(item)


class FakeReferenceLines:
    def __init__(self) -> None:
        self.created: list[list[float]] = []
        self.clear_calls = 0
        self.selected: list[Any] = []

    def create_lines(self, positions: list[float]) -> None:
        self.created.append(positions)

    def clear_selection(self) -> None:
        self.clear_calls += 1

    def select_line(self, line: Any) -> None:
        self.selected.append(line)


class FakeRegionLookupService:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def get_region_description(self, region_id: int) -> tuple[str, str]:
        self.calls.append(region_id)
        return f"description {region_id}", f"lookup-{region_id}"


class FakeTreeItem:
    def __init__(self, region_id: int = 42, index: str = "index") -> None:
        self.region_id = region_id
        self.index = index

    def model(self) -> Any:
        return self

    def itemFromIndex(self, _item: Any) -> Any:
        return self

    def accessibleText(self) -> str:
        return str(self.region_id)


class FakeStructList:
    def __init__(self) -> None:
        self.find_calls: list[tuple[str, Any]] = []
        self.item = FakeTreeItem(index="model-index")

    def findItems(self, lookup: str, *, flags: Any) -> list[FakeTreeItem]:
        self.find_calls.append((lookup, flags))
        return [self.item]

    def indexFromItem(self, item: FakeTreeItem) -> str:
        return item.index


class FakeStructView:
    def __init__(self) -> None:
        self.collapse_calls = 0
        self.scroll_calls: list[Any] = []
        self.current: Any = None

    def collapseAll(self) -> None:
        self.collapse_calls += 1

    def scrollTo(self, item: Any) -> None:
        self.scroll_calls.append(item)

    def setCurrentIndex(self, item: Any) -> None:
        self.current = item


class FakeStructDescription:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class FakeScaleAxis:
    def __init__(self) -> None:
        self.label = ""

    def setLabel(self, label: str) -> None:
        self.label = label


class FakeInfiniteLine:
    pass


class FakeLinearRegion:
    pass


def _presenter(
    *,
    histology_available: bool = True,
    active_cluster_detail: Any | None = None,
    active_region_id: int | None = 42,
) -> tuple[DesktopInteractionPresenter, dict[str, Any]]:
    calls: dict[str, Any] = {"axis": [], "activate": 0, "capture": 0}
    popup_manager = DesktopPopupManager()
    ephys_panel = FakeEphysPanel()
    histology_display = FakeHistologyPanel()
    reference_line_display = FakeReferenceLines()
    region_lookup = FakeRegionLookupService()
    struct_list = FakeStructList()
    struct_view = FakeStructView()
    struct_description = FakeStructDescription()
    scale_axis = FakeScaleAxis()
    widgets = DesktopInteractionWidgets(
        struct_list=struct_list,
        struct_view=struct_view,
        struct_description=struct_description,
        scale_plot=object(),
        histology_plot=object(),
        histology_reference_plot=object(),
        scale_axis=scale_axis,
        bar_colour="bar-colour",
        line_pen="line-pen",
    )
    detail = active_cluster_detail
    if detail is None:
        detail = SimpleNamespace(
            cluster_no=13,
            t_autocorr=np.array([0.0, 1.0, 2.0]),
            autocorr=np.array([3.0, 4.0, 5.0]),
            t_template=np.array([0.0, 0.5]),
            template_waveform=np.array([10.0, 11.0]),
        )
    app = SimpleNamespace(
        queries=SimpleNamespace(
            active_cluster_detail=lambda _idx: detail,
            active_session_notes=lambda: "notes",
            active_histology_region_id=lambda _idx: active_region_id,
        )
    )
    presenter = DesktopInteractionPresenter(
        app=app,
        popup_manager=popup_manager,
        ephys_panel=ephys_panel,
        histology_display=histology_display,
        reference_line_display=reference_line_display,
        region_lookup_service=region_lookup,
        widgets=widgets,
        callbacks=DesktopInteractionCallbacks(
            histology_available=lambda: histology_available,
            activate_window=lambda: calls.__setitem__(
                "activate",
                calls["activate"] + 1,
            ),
            set_axis=lambda *args, **kwargs: calls["axis"].append((args, kwargs)),
            capture_pending_reference_lines=lambda: calls.__setitem__(
                "capture",
                calls["capture"] + 1,
            ),
        ),
        popup_window_factory=FakePopupWindow,
        text_edit_factory=FakeTextEdit,
        plot_item_factory=FakePlotItem,
        bar_graph_item_factory=FakeBarGraphItem,
        plot_curve_item_factory=FakePlotCurveItem,
        infinite_line_type=FakeInfiniteLine,
        linear_region_type=FakeLinearRegion,
    )
    return presenter, {
        "calls": calls,
        "popup_manager": popup_manager,
        "ephys_panel": ephys_panel,
        "histology_display": histology_display,
        "reference_line_display": reference_line_display,
        "region_lookup": region_lookup,
        "widgets": widgets,
        "struct_list": struct_list,
        "struct_view": struct_view,
        "struct_description": struct_description,
        "scale_axis": scale_axis,
    }


def test_display_session_notes_creates_notes_popup() -> None:
    presenter, state = _presenter()

    presenter.display_session_notes()

    notes_window = state["popup_manager"].notes_window
    assert notes_window.kwargs["title"] == "Session notes from Alyx"
    notes = notes_window.layout.widgets[0]
    assert notes.read_only
    assert notes.text == "notes"


def test_cluster_clicked_renders_cluster_popup() -> None:
    presenter, state = _presenter()
    point = [SimpleNamespace(pos=lambda: SimpleNamespace(x=lambda: 12.0))]

    cluster_no = presenter.cluster_clicked(None, point)

    assert cluster_no == 13
    assert state["ephys_panel"].cluster_x_calls == [12.0]
    assert len(state["popup_manager"].cluster_popups) == 1
    popup = state["popup_manager"].cluster_popups[0]
    assert popup.kwargs["title"] == "Cluster 13"
    assert len(popup.popup_widget.items) == 2
    assert len(state["calls"]["axis"]) == 4
    assert state["calls"]["activate"] == 1


def test_cluster_clicked_fails_closed_without_cluster_index() -> None:
    presenter, state = _presenter()
    state["ephys_panel"].cluster_idx = None
    point = [SimpleNamespace(pos=lambda: SimpleNamespace(x=lambda: 12.0))]

    assert presenter.cluster_clicked(None, point) is None

    assert state["popup_manager"].cluster_popups == []


def test_double_click_creates_reference_line_and_captures_pending() -> None:
    presenter, state = _presenter()
    event = SimpleNamespace(double=lambda: True, scenePos=lambda: "scene-pos")

    assert presenter.on_mouse_double_clicked(event)

    assert state["ephys_panel"].scene_pos_calls == ["scene-pos"]
    assert state["reference_line_display"].created == [[125.0]]
    assert state["calls"]["capture"] == 1


def test_double_click_noops_without_histology() -> None:
    presenter, state = _presenter(histology_available=False)
    event = SimpleNamespace(double=lambda: True, scenePos=lambda: "scene-pos")

    assert not presenter.on_mouse_double_clicked(event)

    assert state["reference_line_display"].created == []


def test_mouse_hover_dispatches_reference_scale_and_region_items() -> None:
    presenter, state = _presenter()
    line = FakeInfiniteLine()
    scale_region = FakeLinearRegion()
    hist_region = FakeLinearRegion()
    ref_region = FakeLinearRegion()

    presenter.on_mouse_hover([line, object()])
    presenter.on_mouse_hover([state["widgets"].scale_plot, scale_region])
    presenter.on_mouse_hover([state["widgets"].histology_plot, hist_region])
    presenter.on_mouse_hover(
        [state["widgets"].histology_reference_plot, ref_region]
    )

    assert state["reference_line_display"].clear_calls == 4
    assert state["reference_line_display"].selected == [line]
    assert state["scale_axis"].label == "Scale Factor = 1.23"
    assert state["histology_display"].selected_regions == [hist_region, ref_region]


def test_describe_labels_pressed_creates_region_popup() -> None:
    presenter, state = _presenter(active_region_id=314)

    assert presenter.describe_labels_pressed()

    assert state["region_lookup"].calls == [314]
    assert state["struct_view"].collapse_calls == 1
    assert state["struct_view"].scroll_calls == ["model-index"]
    assert state["struct_view"].current == "model-index"
    assert state["struct_description"].text == "description 314"
    popup = state["popup_manager"].label_window
    assert popup.kwargs["title"] == "Structure Information"
    assert popup.layout.widgets == [
        state["widgets"].struct_view,
        state["widgets"].struct_description,
    ]
    assert state["calls"]["activate"] == 1


def test_label_pressed_updates_region_selection() -> None:
    presenter, state = _presenter()

    presenter.label_pressed(FakeTreeItem(region_id=101))

    assert state["region_lookup"].calls == [101]
    assert state["struct_view"].current == "model-index"
    assert state["struct_description"].text == "description 101"
