import logging
import os
import platform
import sys

if platform.system() == "Darwin":
    if platform.release().split(".")[0] >= "20":
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

import matplotlib.pyplot as mpl  # noqa: F401  # Needed to make Qt show properly.
from PyQt5 import QtWidgets

from ephys_alignment_gui.desktop.main_window import MainWindow

logger = logging.getLogger(__name__)


def viewer(probe_id, one=None, histology=False, spike_collection=None, title=None):
    """Open or reuse an ephys alignment GUI window."""
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    av = MainWindow._get_or_create(
        probe_id=probe_id,
        one=one,
        histology=histology,
        spike_collection=spike_collection,
        title=title,
    )
    av.show()
    return av


def setup_logging(log_level=logging.INFO, log_file=None) -> None:
    """
    Setup logging configuration for the entire application.

    Parameters
    ----------
    log_level : int
        Logging level (logging.DEBUG, logging.INFO, etc.)
    log_file : Path or str, optional
        If provided, also log to this file
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_file}")

    root_logger.info("=" * 60)
    root_logger.info("Ephys Alignment GUI Starting")
    root_logger.info("=" * 60)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="IBL ephys alignment GUI for preprocessed datapackages"
    )
    parser.add_argument(
        "-o",
        "--offline",
        default=True,
        required=False,
        help="Legacy flag; ONE/Alyx online mode is not supported.",
    )
    parser.add_argument(
        "-r",
        "--remote",
        default=False,
        required=False,
        action="store_true",
        help="Remote mode",
    )
    parser.add_argument(
        "-i",
        "--insertion",
        default=None,
        required=False,
        help="Insertion mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        required=False,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        required=False,
        type=str,
        help="Path to log file (optional, logs to console by default)",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    setup_logging(log_level=log_level, log_file=args.log_file)

    logger.info(f"Arguments: {args}")

    app = QtWidgets.QApplication([])
    mainapp = MainWindow(
        offline=args.offline,
        probe_id=args.insertion,
        remote=args.remote,
    )
    mainapp.show()

    logger.info("Starting Qt event loop")
    app.exec_()


if __name__ == "__main__":
    main()
