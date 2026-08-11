"""Desktop shell style primitives shared by views and displays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui


@dataclass(frozen=True)
class DesktopShellStyle:
    """Qt/pyqtgraph style values owned by the desktop shell layer."""

    dotted_pen: Any
    reference_line_pen: Any
    linear_fit_pen: Any
    solid_pen: Any
    fit_pen: Any
    bar_colour: QtGui.QColor
    padding: float = 0.05

    @classmethod
    def default(cls) -> DesktopShellStyle:
        """Return the default desktop alignment GUI style."""
        return cls(
            dotted_pen=pg.mkPen(color="k", style=QtCore.Qt.DotLine, width=2),
            reference_line_pen=pg.mkPen(
                color="k",
                style=QtCore.Qt.DotLine,
                width=10,
            ),
            linear_fit_pen=pg.mkPen(color="r", style=QtCore.Qt.DotLine, width=2),
            solid_pen=pg.mkPen(color="k", style=QtCore.Qt.SolidLine, width=2),
            fit_pen=pg.mkPen(color="b", style=QtCore.Qt.SolidLine, width=3),
            bar_colour=QtGui.QColor(160, 160, 160),
        )
