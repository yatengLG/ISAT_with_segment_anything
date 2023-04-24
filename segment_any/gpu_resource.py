# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5.QtCore import QThread, pyqtSignal
import os


class GPUResource_Thread(QThread):
    message = pyqtSignal(str)

    def __init__(self):
        super(GPUResource_Thread, self).__init__()
        self.gpu_id = None
        self.callback = None

        self.keep = True
        try:
            r = os.popen('nvidia-smi -q -d MEMORY -i 0 | grep Total').readline()
            self.total = r.split(':')[-1].strip().split(' ')[0]
        except Exception as e:
            print(e)
            self.total = 'none'

    def run(self):
        while True:
            r = os.popen('nvidia-smi -q -d MEMORY -i 0 | grep Used').readline()
            used = r.split(':')[-1].strip().split(' ')[0]
            self.message.emit("cuda: {}/{}MiB".format(used, self.total))

    def __del__(self):
        self.message.emit("Ground filter thread | Wait for thread to exit.")
        self.wait()
        self.message.emit("Ground filter thread | Thread exited.")

    def set_callback(self, callback):
        self.callback = callback