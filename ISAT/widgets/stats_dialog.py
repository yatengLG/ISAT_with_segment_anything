# -*- coding: utf-8 -*-
# @Author  : LG

import os

from PyQt5 import QtCore, QtWidgets

from ISAT.ui.stats_dialog import Ui_Dialog


class StatsThread(QtCore.QThread):
    """Background worker: enumerate label jsons and count annotations.

    Emits ``progress(done, total, filename)`` while parsing and
    ``finished_stats(result)`` when done.  ``result`` is a dict::

        {
            "counts": {category: {"polygon": int, "obb": int}},
            "valid_files": int,
            "skipped_files": int,
            "empty_files": int,
            "total_objects": int,
            "empty_objects": int,
        }

    A file is *valid* when it parses as JSON and carries
    ``info.description == "ISAT"``; anything else is skipped and counted.
    ``empty_files`` are valid ISAT files whose ``objects`` list is empty.
    Objects with fewer than 3 segmentation points are ignored (and counted
    as ``empty_objects``) — they are not real annotations.
    """

    progress = QtCore.pyqtSignal(int, int, str)
    finished_stats = QtCore.pyqtSignal(dict)

    def __init__(self, label_dir: str, recursive: bool = False, parent=None):
        super(StatsThread, self).__init__(parent)
        self.label_dir = label_dir
        self.recursive = recursive

    def _collect_jsons(self):
        if self.recursive:
            files = []
            for root, _dirs, names in os.walk(self.label_dir):
                for name in names:
                    if name.lower().endswith(".json"):
                        files.append(os.path.join(root, name))
            return files
        return [
            os.path.join(self.label_dir, name)
            for name in os.listdir(self.label_dir)
            if name.lower().endswith(".json")
        ]

    def run(self):
        counts = {}
        valid_files = 0
        skipped_files = 0
        empty_files = 0
        total_objects = 0
        empty_objects = 0

        try:
            files = self._collect_jsons()
        except OSError:
            files = []
        total = len(files)
        self.progress.emit(0, total, "")

        for i, path in enumerate(files):
            if self.isInterruptionRequested():
                break
            self.progress.emit(i + 1, total, os.path.basename(path))
            try:
                with open(path, "r", encoding="utf-8") as f:
                    import json as _json

                    dataset = _json.load(f)
                info = dataset.get("info", {})
                if info.get("description", "") != "ISAT":
                    skipped_files += 1
                    continue
                objects = dataset.get("objects", [])
                if len(objects) == 0:
                    # valid ISAT file but with no annotation targets
                    empty_files += 1
                    valid_files += 1
                    continue
                for obj in objects:
                    category = obj.get("category", "unknow")
                    segmentation = obj.get("segmentation", [])
                    if len(segmentation) < 3:
                        empty_objects += 1
                        continue
                    shape_type = obj.get("shape_type", "polygon")
                    total_objects += 1
                    entry = counts.setdefault(category, {"polygon": 0, "obb": 0})
                    if shape_type == "obb":
                        entry["obb"] += 1
                    else:
                        entry["polygon"] += 1
                valid_files += 1
            except Exception:
                # unreadable / corrupt / not-json: skip and count
                skipped_files += 1

        self.finished_stats.emit(
            {
                "counts": counts,
                "valid_files": valid_files,
                "skipped_files": skipped_files,
                "empty_files": empty_files,
                "total_objects": total_objects,
                "empty_objects": empty_objects,
            }
        )


class StatsDialog(QtWidgets.QDialog, Ui_Dialog):
    """Annotation statistics dialog.

    Counts polygons / OBBs per category across every ISAT label json in a
    chosen directory.  A background :class:`StatsThread` drives the progress
    bar so the UI stays responsive.  The layout lives in
    ``ISAT/ui/stats_dialog.ui`` (edit with Qt Designer).
    """

    def __init__(self, parent, mainwindow):
        super(StatsDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.thread = None
        self.setupUi(self)

        self.pushButton_browse.clicked.connect(self._browse)
        self.pushButton_start.clicked.connect(self._start)
        self.pushButton_cancel.clicked.connect(self._cancel)

        self.tableWidget.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )

    # ------------------------------------------------------------------
    #  slots
    # ------------------------------------------------------------------

    def _browse(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select label directory", self.lineEdit_dir.text()
        )
        if directory:
            self.lineEdit_dir.setText(directory)

    def _start(self):
        label_dir = self.lineEdit_dir.text().strip()
        if not label_dir:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please select a label directory first."
            )
            return
        if not os.path.isdir(label_dir):
            QtWidgets.QMessageBox.warning(
                self, "Warning", "The directory does not exist: {}".format(label_dir)
            )
            return

        self.tableWidget.setRowCount(0)
        self.progressBar.setValue(0)
        self.label_status.setText("Enumerating label files...")
        self.label_summary.setText("")
        self.pushButton_start.setEnabled(False)
        self.pushButton_cancel.setEnabled(True)

        self.thread = StatsThread(
            label_dir, recursive=self.checkBox_recursive.isChecked(), parent=self
        )
        self.thread.progress.connect(self._on_progress)
        self.thread.finished_stats.connect(self._on_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _cancel(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.requestInterruption()
            self.label_status.setText("Cancelling...")
            self.pushButton_cancel.setEnabled(False)

    def _on_progress(self, done, total, filename):
        if total > 0:
            self.progressBar.setValue(int(done * 100 / total))
        if filename:
            self.label_status.setText(
                "Parsing {} ({}/{})".format(filename, done, total)
            )
        else:
            self.label_status.setText("Found {} label files.".format(total))

    def _on_finished(self, result):
        self.progressBar.setValue(100)
        counts = result["counts"]
        valid = result["valid_files"]
        skipped = result["skipped_files"]
        empty_files = result["empty_files"]
        total_objects = result["total_objects"]
        empty_objects = result["empty_objects"]

        # fill the table: one row per category + a total row
        categories = sorted(counts.keys())
        self.tableWidget.setRowCount(len(categories) + 1)
        total_polygon = 0
        total_obb = 0
        for row, category in enumerate(categories):
            entry = counts[category]
            total_polygon += entry["polygon"]
            total_obb += entry["obb"]
            self._set_row(row, category, entry["polygon"], entry["obb"])

        # total row
        self._set_row(len(categories), "Total", total_polygon, total_obb)
        font = self.tableWidget.font()
        font.setBold(True)
        for col in range(4):
            item = self.tableWidget.item(len(categories), col)
            item.setFont(font)

        summary = (
            "Valid ISAT files: {}    Empty files (no objects): {}    "
            "Skipped: {}    Total objects: {}    "
            "Empty objects (ignored): {}".format(
                valid, empty_files, skipped, total_objects, empty_objects
            )
        )
        self.label_summary.setText(summary)
        self.label_status.setText("Statistics finished.")

        self.pushButton_start.setEnabled(True)
        self.pushButton_cancel.setEnabled(False)

    def _set_row(self, row, category, polygon_count, obb_count):
        self.tableWidget.setItem(
            row, 0, QtWidgets.QTableWidgetItem(str(category))
        )
        self.tableWidget.setItem(
            row, 1, QtWidgets.QTableWidgetItem(str(polygon_count))
        )
        self.tableWidget.setItem(
            row, 2, QtWidgets.QTableWidgetItem(str(obb_count))
        )
        self.tableWidget.setItem(
            row, 3, QtWidgets.QTableWidgetItem(str(polygon_count + obb_count))
        )
