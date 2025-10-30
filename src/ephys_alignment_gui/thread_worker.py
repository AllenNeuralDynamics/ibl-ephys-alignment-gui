from PyQt5.QtCore import QThread, QObject, pyqtSignal
import logging
import traceback

logger = logging.getLogger(__name__)


class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)  # Emit error message

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.fn()
        except Exception as e:
            # Capture and log the exception
            error_msg = f"ERROR in worker thread: {e}"
            logger.error("="*60)
            logger.error(error_msg)
            logger.error("="*60)
            logger.exception("Full traceback:")
            logger.error("="*60)

            # Emit error signal so GUI can handle it
            self.error.emit(error_msg)
        finally:
            # Always emit finished to clean up thread, even if error handling fails
            self.finished.emit()
