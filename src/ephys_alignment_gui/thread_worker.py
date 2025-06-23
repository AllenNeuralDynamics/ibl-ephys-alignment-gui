from PyQt5.QtCore import QThread, QObject, pyqtSignal

class Worker(QObject):
    finished = pyqtSignal()
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()
        self.finished.emit()
