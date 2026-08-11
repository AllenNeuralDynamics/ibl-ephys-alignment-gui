"""Tests for desktop ephys plot export orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ephys_alignment_gui.desktop.displays.ephys_panel_view import EphysPanelPlots
from ephys_alignment_gui.desktop.displays.ephys_plot_exporter import (
    DesktopEphysPlotExporter,
    EphysExportCallbacks,
    EphysExportLayout,
    EphysExportSizes,
)


class FakeLabel:
    def __init__(self, text: str) -> None:
        self.text = text

    def toPlainText(self) -> str:
        return self.text


class FakeAxis:
    def __init__(self, *, label: str = "", width: int = 10, height: int = 5) -> None:
        self.label = FakeLabel(label)
        self._width = width
        self._height = height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class FakePlot:
    def __init__(self, name: str) -> None:
        self.name = name
        self.axes = {
            "bottom": FakeAxis(label=f"{name}-bottom"),
            "left": FakeAxis(label=f"{name}-left", width=12),
            "top": FakeAxis(label=f"{name}-top", height=7),
        }
        self.fixed_widths: list[float] = []
        self.added_items: list[Any] = []
        self.clears = 0

    def getAxis(self, axis: str) -> FakeAxis:
        return self.axes[axis]

    def setFixedWidth(self, width: float) -> None:
        self.fixed_widths.append(width)

    def clear(self) -> None:
        self.clears += 1

    def addItem(self, item: Any) -> None:
        self.added_items.append(item)


class FakeLayout:
    def __init__(self) -> None:
        self.removed: list[Any] = []
        self.added: list[tuple[Any, tuple[Any, ...]]] = []

    def removeItem(self, item: Any) -> None:
        self.removed.append(item)

    def addItem(self, item: Any, *args: Any) -> None:
        self.added.append((item, args))

    def scene(self) -> str:
        return "export-scene"


class FakeDataArea:
    def __init__(self) -> None:
        self._width = 640
        self._height = 480
        self.resizes: list[tuple[float, float]] = []

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def resize(self, width: float, height: float) -> None:
        self.resizes.append((width, height))
        self._width = width
        self._height = height


class FakeAction:
    def __init__(self, label: str) -> None:
        self.label = label

    def text(self) -> str:
        return self.label


class FakePresenter:
    def __init__(self, actions: dict[str, list[FakeAction]]) -> None:
        self.actions = actions
        self.indices = {menu: 0 for menu in actions}
        self.toggles: list[str] = []

    def checked_action(self, menu: str) -> FakeAction | None:
        actions = self.actions.get(menu, [])
        if not actions:
            return None
        return actions[self.indices[menu]]

    def toggle_plot(self, menu: str) -> None:
        actions = self.actions.get(menu, [])
        if not actions:
            return
        self.toggles.append(menu)
        self.indices[menu] = (self.indices[menu] + 1) % len(actions)


def _exporter() -> tuple[
    DesktopEphysPlotExporter,
    dict[str, FakePlot],
    FakeLayout,
    FakeDataArea,
    FakePresenter,
    dict[str, Any],
]:
    plots = {
        "image": FakePlot("image"),
        "image_colorbar": FakePlot("image-colorbar"),
        "line": FakePlot("line"),
        "probe": FakePlot("probe"),
        "probe_colorbar": FakePlot("probe-colorbar"),
    }
    layout = FakeLayout()
    data_area = FakeDataArea()
    presenter = FakePresenter(
        {
            "image": [FakeAction("raw"), FakeAction("rms")],
            "probe": [FakeAction("depth")],
            "line": [FakeAction("spikes")],
        }
    )
    calls: dict[str, Any] = {
        "reset_axis": 0,
        "set_view": [],
        "set_axis": [],
        "set_font": [],
        "add_lines_points": 0,
        "exports": [],
    }

    class FakeImageExporter:
        def __init__(self, scene: Any) -> None:
            self.scene = scene

        def export(self, path: str) -> None:
            calls["exports"].append((self.scene, path))

    exporter = DesktopEphysPlotExporter(
        presenter=presenter,
        panel=SimpleNamespace(
            plots=EphysPanelPlots(**plots),
            probe_colorbars=["probe-cbar"],
        ),
        layout=EphysExportLayout(graphics_layout=layout, data_area=data_area),
        callbacks=EphysExportCallbacks(
            reset_axis=lambda: calls.__setitem__(
                "reset_axis",
                calls["reset_axis"] + 1,
            ),
            set_view=lambda **kwargs: calls["set_view"].append(kwargs),
            set_axis=lambda *args, **kwargs: calls["set_axis"].append(
                (args, kwargs)
            ),
            set_font=lambda *args, **kwargs: calls["set_font"].append(
                (args, kwargs)
            ),
            add_lines_points=lambda: calls.__setitem__(
                "add_lines_points",
                calls["add_lines_points"] + 1,
            ),
            sizes=lambda: EphysExportSizes(probe_width=30, axis_width=5),
        ),
        image_exporter_factory=FakeImageExporter,
    )
    return exporter, plots, layout, data_area, presenter, calls


def test_ephys_plot_exporter_exports_all_ephys_plot_actions_once() -> None:
    exporter, _plots, _layout, _data_area, presenter, calls = _exporter()

    exporter.export(Path("/tmp/out"), sess_info="session_")

    assert calls["exports"] == [
        ("export-scene", "/tmp/out/session_img_raw.png"),
        ("export-scene", "/tmp/out/session_img_rms.png"),
        ("export-scene", "/tmp/out/session_probe_depth.png"),
        ("export-scene", "/tmp/out/session_line_spikes.png"),
    ]
    assert presenter.toggles == ["image", "image", "probe", "line"]
    assert calls["add_lines_points"] == 4


def test_ephys_plot_exporter_restores_layout_and_desktop_state() -> None:
    exporter, plots, layout, data_area, _presenter, calls = _exporter()

    exporter.export(Path("/tmp/out"))

    assert calls["reset_axis"] == 1
    assert calls["set_view"] == [
        {"view": 1, "configure": False},
        {"view": 1, "configure": False},
    ]
    assert data_area.resizes == [(700, 480), (250, 480), (200, 480), (640, 480)]
    assert plots["probe"].fixed_widths == [55, 35, 35]
    assert plots["probe_colorbar"].clears == 1
    assert plots["probe_colorbar"].added_items == ["probe-cbar"]
    assert layout.removed[:3] == [
        plots["probe"],
        plots["probe_colorbar"],
        plots["line"],
    ]
    assert layout.added[-5:] == [
        (plots["probe_colorbar"], (0, 0, 1, 2)),
        (plots["image_colorbar"], (0, 2)),
        (plots["probe"], (1, 0)),
        (plots["line"], (1, 1)),
        (plots["image"], (1, 2)),
    ]
