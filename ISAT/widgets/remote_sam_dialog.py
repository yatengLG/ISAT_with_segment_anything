# -*- coding: utf-8 -*-
# @Author  : LG

import os.path
import requests
from ISAT.ui.remote_sam_dialog import Ui_Dialog
from ISAT.configs import CHECKPOINT_PATH
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QValidator

class RemoteSamDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(RemoteSamDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)

        self.remote_sam_model_name = None

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.checkBox_use_remote.setEnabled(False)
        self.frame_info.setVisible(False)
        self.lineEdit_host.setValidator(IPv4Validator())

        self.pushButton_check.clicked.connect(self.check_connection)
        self.checkBox_use_remote.stateChanged.connect(self.use_remote_sam)
        self.pushButton_close.clicked.connect(self.close)

    def check_connection(self):
        host = self.lineEdit_host.text()
        port = self.lineEdit_port.text()
        try:
            response = requests.get(f"http://{host}:{port}/api/info")
            if response.status_code != 200:
                QtWidgets.QMessageBox.warning(self, "Error", f"Could not connect to remote Sam, status_code: {response.status_code}")
                return
            checkpoint = response.json()['checkpoint']
            device = response.json()['device']
            dtype = response.json()['dtype']

            try:
                model_name = os.path.split(checkpoint)[-1]

                self.frame_info.setVisible(True)
                self.label_name.setText(model_name)
                self.label_device.setText(device)
                self.label_dtype.setText(dtype)

                self.checkBox_use_remote.setEnabled(True)

            except Exception as e:
                self.frame_info.setVisible(False)
                self.checkBox_use_remote.setEnabled(False)
                QtWidgets.QMessageBox.warning(self, "Error", str(e))

        except requests.exceptions.RequestException as e:
            self.frame_info.setVisible(False)
            self.checkBox_use_remote.setEnabled(False)
            QtWidgets.QMessageBox.warning(self, "Error", str(e))

    def use_remote_sam(self, check_state):
        if check_state == QtCore.Qt.CheckState.Checked:
            model_name = self.label_name.text()
            loadl_model_path = os.path.join(CHECKPOINT_PATH, model_name)
            if not os.path.exists(loadl_model_path):
                self.mainwindow.use_remote_sam = False
                QtWidgets.QMessageBox.warning(self, "Error", f"Local model {model_name} does not exist.")
            else:
                try:
                    self.mainwindow.use_remote_sam = True
                    self.mainwindow.init_segment_anything(loadl_model_path)
                except Exception as e:
                    self.mainwindow.use_remote_sam = False
                    QtWidgets.QMessageBox.warning(self, "Error", f"Init local sam failed.\n {e}")

        else:
            self.mainwindow.use_remote_sam = False

        state = QtCore.Qt.CheckState.Checked if self.mainwindow.use_remote_sam else QtCore.Qt.CheckState.Unchecked
        self.sender().setCheckState(state)
        self.lineEdit_host.setEnabled(not self.mainwindow.use_remote_sam)
        self.lineEdit_port.setEnabled(not self.mainwindow.use_remote_sam)
        self.pushButton_check.setEnabled(not self.mainwindow.use_remote_sam)

class IPv4Validator(QValidator):
    def validate(self, input_str, pos):
        parts = input_str.split('.')
        if len(parts) > 4:
            # 返回 Invalid + 原始输入 + 光标位置
            return (QValidator.Invalid, input_str, pos)

        for i, part in enumerate(parts):
            if i < len(parts) - 1 and not part:
                return (QValidator.Invalid, input_str, pos)
            if part:
                if not part.isdigit() or len(part) > 3:
                    return (QValidator.Invalid, input_str, pos)
                num = int(part)
                if num < 0 or num > 255:
                    return (QValidator.Invalid, input_str, pos)

        if len(parts) == 4:
            if not parts[3]:
                # 允许中间状态（如 192.168.1.）
                return (QValidator.Intermediate, input_str, pos)
            elif 0 <= int(parts[3]) <= 255:
                return (QValidator.Acceptable, input_str, pos)
            else:
                return (QValidator.Invalid, input_str, pos)
        else:
            # 允许继续输入（如 192.168）
            return (QValidator.Intermediate, input_str, pos)
