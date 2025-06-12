from PyQt5.QtCore import QThread, QObject

class Worker(QObject):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()
