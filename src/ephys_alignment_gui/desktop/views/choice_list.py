"""Qt combobox/list-model population helpers for desktop views."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from PyQt5 import QtGui, QtWidgets


def populate_choice_list(
    values: Sequence[str],
    model: Any,
    combobox: Any,
    *,
    item_factory: Callable[[str], Any] = QtGui.QStandardItem,
) -> None:
    """Populate a Qt item model and size the combobox popup for its labels."""
    model.clear()
    labels = [str(value) for value in values]
    for label in labels:
        item = item_factory(label)
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
    min_width += combobox.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
    combobox.view().setMinimumWidth(min_width)
    combobox.setCurrentIndex(0)
