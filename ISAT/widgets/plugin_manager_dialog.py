# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtCore, QtWidgets

from ISAT.ui.plugin_manager_dialog import Ui_Dialog


class PluginManagerDialog(QtWidgets.QDialog, Ui_Dialog):
    """Plugin manager interface (view only).

    Presentation of the plugins discovered by :class:`PluginManager` —
    enable/disable switches and the reload button.  Plugin discovery and
    lifecycle event dispatch live in ``PluginManager``.
    """

    def __init__(self, plugin_manager, parent=None):
        super(PluginManagerDialog, self).__init__(parent)
        self.plugin_manager = plugin_manager
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.horizontalHeader().setSectionResizeMode(
            4, QtWidgets.QHeaderView.Stretch
        )
        self.tableWidget.setColumnWidth(0, 100)
        self.tableWidget.setColumnWidth(1, 250)
        self.tableWidget.setColumnWidth(2, 150)
        self.tableWidget.setColumnWidth(3, 150)

        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_reload.clicked.connect(self.reload_plugins)

        self.update_gui()

    def reload_plugins(self):
        """Reload plugins through the manager and refresh the table."""
        self.plugin_manager.load_plugins()
        self.update_gui()

    def update_gui(self):
        self.tableWidget.setRowCount(0)
        row = 0
        for plugin_instance in self.plugin_manager.plugins:
            activate_checkbox = QtWidgets.QCheckBox()
            activate_checkbox.stateChanged.connect(
                plugin_instance.activate_state_changed
            )
            plugin_name_item = QtWidgets.QTableWidgetItem(
                plugin_instance.get_plugin_name()
            )
            plugin_author_item = QtWidgets.QTableWidgetItem(
                plugin_instance.get_plugin_author()
            )
            plugin_author_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            plugin_version_item = QtWidgets.QTableWidgetItem(
                plugin_instance.get_plugin_version()
            )
            plugin_version_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            plugin_description_item = QtWidgets.QTableWidgetItem(
                plugin_instance.get_plugin_description()
            )

            self.tableWidget.insertRow(self.tableWidget.rowCount())
            self.tableWidget.setCellWidget(row, 0, activate_checkbox)
            self.tableWidget.setItem(row, 1, plugin_name_item)
            self.tableWidget.setItem(
                row, 2, QtWidgets.QTableWidgetItem(plugin_author_item)
            )
            self.tableWidget.setItem(
                row, 3, QtWidgets.QTableWidgetItem(plugin_version_item)
            )
            self.tableWidget.setItem(
                row, 4, QtWidgets.QTableWidgetItem(plugin_description_item)
            )

            row += 1
