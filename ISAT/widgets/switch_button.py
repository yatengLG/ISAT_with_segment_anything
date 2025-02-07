# -*- coding: utf-8 -*-
# @Author  : LG
# from https://blog.51cto.com/u_15872074/5841477

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal, QTimer, QRect, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen


class SwitchBtn(QWidget):
    #信号
    checkedChanged = pyqtSignal(bool)
    def __init__(self,parent=None):
        super(QWidget, self).__init__(parent)

        self.checked = False
        self.bgColorOff = QColor('#6E798A')
        self.bgColorOn = QColor('#70AEFF')

        self.sliderColorOff = QColor(255, 255, 255)
        self.sliderColorOn = QColor(255, 255, 255)

        self.textColorOff = QColor(255, 255, 255)
        self.textColorOn = QColor(255, 255, 255)

        self.textOff = "OFF"
        self.textOn = "ON"

        self.space = 2
        self.rectRadius = 5

        self.step = self.width() / 50
        self.startX = 0
        self.endX = 0

        self.timer = QTimer(self)  # 初始化一个定时器
        self.timer.timeout.connect(self.updateValue)  # 计时结束调用operate()方法
        self.setFont(QFont("timesnewroman", 10))

    def updateValue(self):
        if self.checked:
            if self.startX < self.endX:
                self.startX = self.startX + self.step
            else:
                self.startX = self.endX
                self.timer.stop()
        else:
            if self.startX  > self.endX:
                self.startX = self.startX - self.step
            else:
                self.startX = self.endX
                self.timer.stop()
        self.update()

    def mousePressEvent(self,event):
        self.checked = not self.checked
        #发射信号
        self.checkedChanged.emit(self.checked)
        # 每次移动的步长为宽度的50分之一
        self.step = self.width() / 50
        #状态切换改变后自动计算终点坐标
        if self.checked:
            self.endX = self.width() - self.height()
        else:
            self.endX = 0
        self.timer.start(5)

    def paintEvent(self, evt):
        #绘制准备工作, 启用反锯齿
            painter = QPainter()
            painter.begin(self)
            painter.setRenderHint(QPainter.Antialiasing)
            #绘制背景
            self.drawBg(evt, painter)
            #绘制滑块
            self.drawSlider(evt, painter)
            #绘制文字
            self.drawText(evt, painter)
            painter.end()

    def drawText(self, event, painter):
        painter.save()
        if self.checked:
            painter.setPen(self.textColorOn)
            painter.drawText(0, 0, int(self.width() / 2 + self.space * 2), int(self.height()), Qt.AlignCenter, self.textOn)
        else:
            painter.setPen(self.textColorOff)
            painter.drawText(int(self.width() / 2), 0, int(self.width() / 2 - self.space), int(self.height()), Qt.AlignCenter, self.textOff)
        painter.restore()

    def drawBg(self, event, painter):
        painter.save()
        painter.setPen(Qt.NoPen)
        if self.checked:
            painter.setBrush(self.bgColorOn)
        else:
            painter.setBrush(self.bgColorOff)
        #半径为高度的一半
        radius = self.height() / 2
        #圆的宽度为高度
        circleWidth = self.height()

        path = QPainterPath()
        path.moveTo(radius, 0)
        path.arcTo(QRectF(0, 0, circleWidth, circleWidth), 90, 180)
        path.lineTo(self.width() - radius, self.height())
        path.arcTo(QRectF(self.width() - self.height(), 0, circleWidth, circleWidth), 270, 180)
        path.lineTo(radius, 0)

        painter.drawPath(path)
        painter.restore()

    def drawSlider(self, event, painter):
        painter.save()
        if self.checked:
            painter.setBrush(self.sliderColorOn)
            painter.setPen(QPen(self.sliderColorOn, 1))
        else:
            painter.setBrush(self.sliderColorOff)

        sliderWidth = self.height() - self.space * 2
        sliderRect = QRectF(self.startX + self.space, self.space, sliderWidth, sliderWidth)
        painter.drawEllipse(sliderRect)

        painter.restore()

    def setChecked(self, checked=False):
        if self.checked != checked:
            self.mousePressEvent(None)
