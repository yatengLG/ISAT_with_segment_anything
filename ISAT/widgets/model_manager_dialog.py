# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from ISAT.ui.model_manager_dialog import Ui_Dialog
from ISAT.configs import CHECKPOINT_PATH
from ISAT.segment_any.model_zoo import model_dict
from urllib import request
from functools import partial
import shutil
import os

class DownloadThread(QThread):
    tag = pyqtSignal(float, float)
    def __init__(self, parent=None):
        super(DownloadThread, self).__init__(parent)
        self.url = None
        self.name = None
        self.pause = False

        self.block_size = 4096

    def setNameAndUrl(self, name, url):
        self.url = url
        self.name = name
        self.pause = False

    def run(self):
        if self.name is not None and self.url is not None:
            tmp_root = os.path.join(CHECKPOINT_PATH, 'tmp')
            if not os.path.exists(tmp_root):
                os.mkdir(tmp_root)
            print('Download {} from {}'.format(self.name,  self.url))
            # 检查缓存
            downloaded_size = 0
            download_tmp = os.path.join(tmp_root, self.name)
            if os.path.exists(download_tmp):
                with open(download_tmp, 'rb') as f:
                    downloaded_size = len(f.read())

            req = request.Request(self.url, headers={"Range": "bytes=0-"})
            try:
                response = request.urlopen(req, timeout=10)
                total_size = int(response.headers['Content-Length'])
            except Exception as e:
                print('When download {} from {}, {}'.format(self.name, self.url, e))
                return
            # 存在缓存
            if downloaded_size != 0:
                # 判断缓存文件是否下载完
                if downloaded_size > total_size:
                    self.move_from_tmp(download_tmp, os.path.join(CHECKPOINT_PATH, self.name))
                    return
                # 断点续传
                content_range = response.headers.get('Content-Range', None)
                if content_range is not None:
                    req = request.Request(self.url, headers={"Range": "bytes={}-".format(downloaded_size)})
                    response = request.urlopen(req)
                    content_range = response.headers.get('Content-Range', None)
                    if content_range is not None:
                        print('Resume downloading: ', content_range)
                else:
                    print('Not supprot resume download.')
                    downloaded_size = 0

            open_mode = 'wb' if downloaded_size == 0 else 'ab'
            with open(download_tmp, open_mode) as f:
                while True:
                    if self.pause:
                        self.tag.emit(-1, -1)
                        return
                    buffer = response.read(self.block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded_size += len(buffer)
                    self.tag.emit(downloaded_size, total_size)
            # 下载完成
            self.move_from_tmp(download_tmp, os.path.join(CHECKPOINT_PATH, self.name))

    def move_from_tmp(self, from_path, to_path):
        try:
            shutil.move(from_path, to_path)
            self.tag.emit(-2, -2)

        except Exception as e:
            print('Error when move {} to {}, {}'.format(from_path, to_path, e))

class ModelManagerDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(ModelManagerDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.download_thread_dict = {}
        self.update_gui()

        self.tableWidget.setColumnWidth(0, 200)
        self.pushButton_clear_tmp.clicked.connect(self.clear_tmp)

    def update_gui(self):
        for index, (name, info_dict) in enumerate(model_dict.items()):
            url = info_dict.get('url', '')
            memory = info_dict.get('memory', '')
            params = info_dict.get('params', '')
            name_label = QtWidgets.QLabel()
            name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            name_label.setText(name)
            # 显存占用
            params_label = QtWidgets.QLabel()
            params_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            params_label.setText(params)
            # 权重大小
            memory_label = QtWidgets.QLabel()
            memory_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            memory_label.setText(memory)
            # 下载/删除按钮
            ops_button = QtWidgets.QPushButton()
            if os.path.exists(os.path.join(CHECKPOINT_PATH, name)):
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'red')
                ops_button.setStyleSheet('QWidget {color: %s}' % 'red')
                ops_button.setText('delete')
                ops_button.clicked.connect(self.delete)
            else:
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'green')
                ops_button.setStyleSheet('QWidget {color: %s}' % 'green')
                ops_button.setText('download')
                ops_button.clicked.connect(self.download)

            self.tableWidget.insertRow(index)
            self.tableWidget.setCellWidget(index, 0, name_label)
            self.tableWidget.setCellWidget(index, 1, memory_label)
            self.tableWidget.setCellWidget(index, 2, params_label)
            self.tableWidget.setCellWidget(index, 3, ops_button)

    def download(self):
        button = self.sender()
        row = self.tableWidget.indexAt(button.pos()).row()
        name_label = self.tableWidget.cellWidget(row, 0)
        name = name_label.text()
        info_dict = model_dict.get(name, None)
        url = info_dict.get('url', None)

        if name in self.download_thread_dict:
            download_thread = self.download_thread_dict[name]
        else:
            download_thread = DownloadThread(self)
            self.download_thread_dict[name] = download_thread

        download_thread.setNameAndUrl(name, url)
        download_thread.tag.connect(partial(self.download_process, button))
        download_thread.start()

        button.setEnabled(False)
        button.clicked.connect(self.pause)
        button.setEnabled(True)

    def download_process(self, button:QtWidgets.QPushButton, downloaded_size, total_size):
        if downloaded_size == -1 and total_size == -1:
            button.setText('continue')
            button.setEnabled(False)
            button.clicked.connect(self.download)
            button.setEnabled(True)

        elif downloaded_size == -2 and total_size == -2:
            button.setText('delete')
            button.setStyleSheet('QWidget {color: %s}' % 'red')
            button.setEnabled(False)
            button.clicked.connect(self.delete)
            button.setEnabled(True)
            self.mainwindow.update_menuSAM()
        else:
            button.setText('{:.2f}% - {}/{}M'.format(downloaded_size/total_size*100, downloaded_size//1000000, total_size//1000000))


    def pause(self):
        button = self.sender()
        row = self.tableWidget.indexAt(button.pos()).row()
        name_label = self.tableWidget.cellWidget(row, 0)
        name = name_label.text()

        download_thread:DownloadThread = self.download_thread_dict[name]
        download_thread.pause = True

    def delete(self):
        button = self.sender()
        row = self.tableWidget.indexAt(button.pos()).row()
        name_label = self.tableWidget.cellWidget(row, 0)
        name = name_label.text()
        try:
            os.remove(os.path.join(CHECKPOINT_PATH, name))
            button.setText('download')
            button.setStyleSheet('QWidget {color: %s}' % 'green')
            button.setEnabled(False)
            button.clicked.connect(self.download)
            button.setEnabled(True)
        except Exception as e:
            print('Error when remove {}, {}'.format(
                os.path.join(CHECKPOINT_PATH, name), e))
        self.mainwindow.update_menuSAM()

    def clear_tmp(self):
        remove_list = []
        for name in os.listdir(os.path.join(CHECKPOINT_PATH, 'tmp')):
            tmp_file = os.path.join(CHECKPOINT_PATH, 'tmp', name)
            if name in self.download_thread_dict:
                thread = self.download_thread_dict.get(name, None)
                if thread is not None:
                    if thread.isRunning():
                        continue
            try:
                os.remove(tmp_file)
                remove_list.append(tmp_file)

            except Exception as e:
                print('Error when remove {}, {}'.format(
                    os.path.join(CHECKPOINT_PATH, name), e))
        QtWidgets.QMessageBox.information(
            self,
            'clear tmp',
            'Remove tmp: [' + '],['.join(remove_list) + ']'
        )

