# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5.QtCore import QThread, pyqtSignal
import os
import platform

osplatform = platform.system()


class GPUResource_Thread(QThread):
    """
    The thread for monitoring GPU0 resources.

    Arguments:
        gpu_id (int): The id of the GPU device to monitor. default 0.
    """
    message = pyqtSignal(str)

    def __init__(self, gpu_id: int=0):
        super(GPUResource_Thread, self).__init__()
        self.gpu_id = gpu_id

        if osplatform == 'Windows':
            self.command = 'nvidia-smi -q -d MEMORY -i {} | findstr'.format(self.gpu_id)
        elif osplatform == 'Linux':
            self.command = 'nvidia-smi -q -d MEMORY -i {} | grep'.format(self.gpu_id)
        elif osplatform == 'Darwin':
            self.command = 'nvidia-smi -q -d MEMORY -i {} | grep'.format(self.gpu_id)
        else:
            self.command = 'nvidia-smi -q -d MEMORY -i {} | grep'.format(self.gpu_id)
        try:
            r = os.popen('{} Total'.format(self.command)).readline()
            self.total = r.split(':')[-1].strip().split(' ')[0]
        except Exception as e:
            print(e)
            self.total = 'none'

    def run(self):
        while True:
            try:
                r = os.popen('{} Used'.format(self.command)).readline()
                used = r.split(':')[-1].strip().split(' ')[0]
                self.message.emit("cuda: {}/{}MiB".format(used, self.total))
            except:
                self.message.emit("cuda: {}/{}MiB".format('-', '-'))

    def __del__(self):
        self.message.emit("Ground filter thread | Wait for thread to exit.")
        self.wait()
        self.message.emit("Ground filter thread | Thread exited.")

