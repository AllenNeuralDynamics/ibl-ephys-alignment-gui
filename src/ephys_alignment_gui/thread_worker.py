from PyQt5.QtCore import QThread, QObject, pyqtSignal
import traceback

class Worker(QObject):
    """Worker for running functions in background threads with proper error handling."""

    finished = pyqtSignal()
    error = pyqtSignal(Exception, str)  # (exception, traceback_string)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Execute the function with exception handling."""
        # Import here to avoid circular dependency
        from ephys_alignment_gui.launch_gui import is_debug_mode

        try:
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit()
            return result

        except Exception as e:
            # Capture full traceback
            tb_str = traceback.format_exc()

            # Always print to console
            print("\n" + "="*60)
            print("ERROR IN WORKER THREAD")
            print("="*60)
            print(tb_str)
            print("="*60 + "\n")

            # Emit error signal for GUI handling
            self.error.emit(e, tb_str)

            if is_debug_mode():
                # Debug mode: Re-raise so VSCode can break on it
                # VSCode will catch this and show the exception in the debug console
                print("Re-raising exception for debugger...")
                raise
            else:
                # Normal mode: Don't re-raise, let GUI handle via signal
                pass
