# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5.QtCore import QThread, pyqtSignal
import os
import platform

osplatform = platform.system()


class GPUResource_Thread(QThread):
    message = pyqtSignal(str)

    def __init__(self):
        super(GPUResource_Thread, self).__init__()
        self.gpu_id = None
        self.callback = None

        if osplatform == 'Windows':
            self.command = 'nvidia-smi -q -d MEMORY -i 0 | findstr'
        elif osplatform == 'Linux':
            self.command = 'nvidia-smi -q -d MEMORY -i 0 | grep'
        elif osplatform == 'Darwin':
            self.command = 'nvidia-smi -q -d MEMORY -i 0 | grep'
        else:
            self.command = 'nvidia-smi -q -d MEMORY -i 0 | grep'
        try:
            r = os.popen('{} Total'.format(self.command)).readline()
            self.total = r.split(':')[-1].strip().split(' ')[0]
        except Exception as e:
            print(e)
            self.total = 'none'

    def run(self):
        while True:
            r = os.popen('{} Used'.format(self.command)).readline()
            used = r.split(':')[-1].strip().split(' ')[0]
            self.message.emit("cuda: {}/{}MiB".format(used, self.total))

    def __del__(self):
        self.message.emit("Ground filter thread | Wait for thread to exit.")
        self.wait()
        self.message.emit("Ground filter thread | Thread exited.")

    def set_callback(self, callback):
        self.callback = callback