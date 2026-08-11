"""Desktop popup window lifecycle ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DesktopPopupManager:
    """Own transient desktop popup windows for the active view."""

    cluster_popups: list[Any] = field(default_factory=list)
    clusters_normal: bool = True
    label_window: Any = None
    notes_window: Any = None
    subject_window: Any = None
    nearby_window: Any = None
    nearby_table: Any = None
    region_window: Any = None

    def add_cluster_popup(self, popup: Any) -> None:
        """Track a cluster detail popup."""
        self.cluster_popups.append(popup)

    def remove_cluster_popup(self, popup: Any) -> None:
        """Forget a closed cluster detail popup."""
        try:
            self.cluster_popups.remove(popup)
        except ValueError:
            pass

    def close_cluster_popups(self) -> None:
        """Close all cluster popups and forget them."""
        for popup in list(self.cluster_popups):
            self._close_popup(popup)
        self.cluster_popups = []
        self.clusters_normal = True

    def toggle_cluster_minimized(self) -> bool:
        """Toggle cluster popups between normal and minimized state."""
        self.clusters_normal = not self.clusters_normal
        method_name = "showNormal" if self.clusters_normal else "showMinimized"
        for popup in self.cluster_popups:
            try:
                getattr(popup, method_name)()
            except RuntimeError:
                pass
        return self.clusters_normal

    def close_all(self) -> None:
        """Close every popup owned by this manager."""
        self.close_cluster_popups()
        for attr in (
            "label_window",
            "notes_window",
            "subject_window",
            "nearby_window",
            "region_window",
        ):
            popup = getattr(self, attr)
            if popup is not None:
                self._close_popup(popup)
                setattr(self, attr, None)
        self.nearby_table = None

    @staticmethod
    def _close_popup(popup: Any) -> None:
        for signal_name in ("closed", "moved"):
            signal = getattr(popup, signal_name, None)
            if signal is None:
                continue
            try:
                signal.disconnect()
            except (TypeError, AttributeError, RuntimeError):
                pass
        try:
            popup.blockSignals(True)
            popup.close()
        except (AttributeError, RuntimeError):
            pass
