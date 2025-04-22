# -*- coding: utf-8 -*-
# @Author  : LG

import os.path
import requests
from ISAT.ui.remote_sam_dialog import Ui_Dialog
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QValidator

class RemoteSamDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(RemoteSamDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)

        self.remote_sam_model_name = None

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.checkBox_use_remote.setEnabled(False)
        self.widget_info.setVisible(False)
        self.lineEdit_host.setValidator(IPv4Validator())

        self.pushButton_check.clicked.connect(self.check_connection)
        self.checkBox_use_remote.stateChanged.connect(self.use_remote_sam)

    def check_connection(self):
        host = self.lineEdit_host.text()
        port = self.lineEdit_port.text()
        try:
            response = requests.get(f"http://{host}:{port}/info")
            if response.status_code != 200:
                return
            checkpoint = response.json()['checkpoint']
            device = response.json()['device']
            dtype = response.json()['dtype']

            try:
                model_name = os.path.split(checkpoint)[-1]

                self.widget_info.setVisible(True)
                self.label_name.setText(model_name)
                self.label_device.setText(device)
                self.label_dtype.setText(dtype)

                self.checkBox_use_remote.setEnabled(True)

            except Exception as e:
                self.widget_info.setVisible(False)
                self.checkBox_use_remote.setEnabled(False)

        except requests.exceptions.RequestException as e:
            self.widget_info.setVisible(False)
            self.checkBox_use_remote.setEnabled(False)

    def use_remote_sam(self, check_state):
        if check_state == QtCore.Qt.CheckState.Checked:
            try:
                self.mainwindow.use_remote_sam = True
                self.mainwindow.init_segment_anything(self.label_name.text(), checked=True)
            except Exception as e:
                self.sender().setChecked(False)
                self.mainwindow.use_remote_sam = False
                QtWidgets.QMessageBox.warning(self, "Error", f"Init local sam failed.\n {e}")
                return
        else:
            self.mainwindow.use_remote_sam = False


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