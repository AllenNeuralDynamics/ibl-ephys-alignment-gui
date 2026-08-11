"""Desktop Allen region lookup widget construction."""

from __future__ import annotations

import numpy as np
from PyQt5 import QtGui, QtWidgets


def initialize_region_lookup(window, allen) -> None:
    """Create the Allen Atlas structure tree widgets."""
    allen = allen.drop([0]).reset_index(drop=True)

    def parent_path(struct_path):
        return struct_path.rsplit("/", 2)[0] + "/"

    allen["parent_path"] = allen["structure_id_path"].apply(parent_path)

    window.struct_list = QtGui.QStandardItemModel()
    window.struct_view = QtWidgets.QTreeView()
    window.struct_view.setModel(window.struct_list)
    window.struct_view.clicked.connect(window.label_pressed)
    window.struct_view.setHeaderHidden(True)

    unique_levels = np.unique(allen["depth"]).astype(int)
    parent_info = {}
    idx = np.where(allen["depth"] == unique_levels[0])[0]
    item = QtGui.QStandardItem(allen["acronym"][idx[0]] + ": " + allen["name"][idx[0]])
    icon = QtGui.QPixmap(20, 20)
    icon.fill(QtGui.QColor("#" + allen["color_hex_triplet"][idx[0]]))
    item.setIcon(QtGui.QIcon(icon))
    item.setAccessibleText(str(allen["id"][idx[0]]))
    item.setEditable(False)
    window.struct_list.appendRow(item)
    parent_info.update({allen["structure_id_path"][idx[0]]: item})

    for level in unique_levels[1:]:
        idx_levels = np.where(allen["depth"] == level)[0]
        for idx in idx_levels:
            parent = allen["parent_path"][idx]
            parent_item = parent_info[parent]
            item = QtGui.QStandardItem(
                allen["acronym"][idx] + ": " + allen["name"][idx]
            )
            icon.fill(QtGui.QColor("#" + allen["color_hex_triplet"][idx]))
            item.setIcon(QtGui.QIcon(icon))
            item.setAccessibleText(str(allen["id"][idx]))
            item.setEditable(False)
            parent_item.appendRow(item)
            parent_info.update({allen["structure_id_path"][idx]: item})

    window.struct_description = QtWidgets.QTextEdit()
