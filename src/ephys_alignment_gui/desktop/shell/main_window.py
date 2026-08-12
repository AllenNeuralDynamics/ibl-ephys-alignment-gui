import logging
from typing import Any

from PyQt5 import QtWidgets

from ephys_alignment_gui.application.workspace import AlignmentWorkspace
from ephys_alignment_gui.core.settings import (
    OUTPUT_ROOT_ENV_VAR,
    output_root_from_environment,
)
from ephys_alignment_gui.desktop.displays import DesktopDisplays
from ephys_alignment_gui.desktop.displays.config import (
    desktop_display_config_from_handles,
)
from ephys_alignment_gui.desktop.shell import window_setup
from ephys_alignment_gui.desktop.shell.actions import DesktopShellActions
from ephys_alignment_gui.desktop.shell.handles import (
    display_config_handles_from_main_window,
    view_handles_from_main_window,
    workbench_port_handles_from_main_window,
)
from ephys_alignment_gui.desktop.shell.popup_manager import DesktopPopupManager
from ephys_alignment_gui.desktop.shell.region_lookup_setup import (
    initialize_region_lookup,
)
from ephys_alignment_gui.desktop.shell.style import DesktopShellStyle
from ephys_alignment_gui.desktop.views import DesktopViews
from ephys_alignment_gui.desktop.workbench import DesktopWorkbench
from ephys_alignment_gui.desktop.workbench.ports import (
    desktop_workbench_ports_from_handles,
)

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]

    @staticmethod
    def _get_or_create(title="Electrophysiology Atlas", **kwargs):
        av = next(
            filter(
                lambda e: e.isVisible() and e.windowTitle() == title,
                MainWindow._instances(),
            ),
            None,
        )
        if av is None:
            av = MainWindow(**kwargs)
            av.setWindowTitle(title)
        return av

    def __init__(
        self,
        offline=True,
        probe_id=None,
        one=None,
        histology=True,
        spike_collection=None,
        remote=False,
    ) -> None:
        super().__init__()
        requested_offline = self._normalize_offline_flag(offline)
        if not requested_offline:
            logger.warning(
                "ONE/Alyx online mode is not supported in this branch; "
                "using preprocessed datapackage mode. DocDB remains available "
                "through the DocDB checkbox."
            )
        offline = True

        self._workspace = AlignmentWorkspace()
        self.app = self._workspace.app
        self.popup_manager = DesktopPopupManager()
        self.style = DesktopShellStyle.default()
        self.shell_actions = DesktopShellActions(self)
        self.offline: bool = offline
        window_setup.initialize_shell(self, offline=offline)
        self.displays = DesktopDisplays.create(
            config=desktop_display_config_from_handles(
                display_config_handles_from_main_window(self)
            ),
        )
        window_setup.install_main_layout(self, displays=self.displays)
        self.views = DesktopViews.from_handles(
            view_handles_from_main_window(self),
            displays=self.displays,
        )
        self.selection_view = self.views.selection
        self.path_view = self.views.path
        self.depth_plot_view = self.views.depth
        self.shank_screen_view = self.views.shank_screen
        self.alignment_screen_view = self.views.alignment_screen
        self.export_view = self.views.export
        self.desktop_workbench = DesktopWorkbench.create(
            app=self.app,
            parent=self,
            views=self.views,
            displays=self.displays,
            ports=desktop_workbench_ports_from_handles(
                workbench_port_handles_from_main_window(self)
            ),
        )
        self.desktop_workbench.initialize_startup_stream_state()
        self.desktop_workbench.connect_events()
        self._set_default_output_root_from_environment()

        self.desktop_workbench.initialize_region_lookup(
            lambda allen: initialize_region_lookup(self, allen)
        )

    def closeEvent(self, event) -> None:
        """Shut down desktop work before the Qt window closes."""
        workbench = getattr(self, "desktop_workbench", None)
        if workbench is not None and not workbench.shutdown():
            logger.warning("Window close ignored while a fresh load is still running")
            event.ignore()
            return
        super().closeEvent(event)

    @staticmethod
    def _normalize_offline_flag(offline: Any) -> bool:
        """Interpret legacy CLI/string offline flags."""
        if isinstance(offline, str):
            return offline.strip().lower() not in {"0", "false", "no", "off"}
        return bool(offline)

    def _show_one_unsupported(self, feature: str) -> None:
        """Report unsupported ONE/Alyx-only actions without crashing."""
        message = (
            f"{feature} requires ONE/Alyx online mode, which is not supported "
            "in this preprocessed datapackage workflow."
        )
        logger.warning(message)
        QtWidgets.QMessageBox.information(self, "Unavailable", message)

    def _set_default_output_root_from_environment(self) -> None:
        """Use an environment-provided save root as the startup default."""
        output_root = output_root_from_environment()
        if output_root is None:
            return
        if self.shell_actions.set_save_root(output_root):
            logger.info(
                "Default save root set from %s: %s",
                OUTPUT_ROOT_ENV_VAR,
                output_root,
            )
