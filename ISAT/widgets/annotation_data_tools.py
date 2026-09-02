# -*- coding: utf-8 -*-
# @Author  : LG

import json
import math
import os

from PyQt5 import QtCore, QtGui, QtWidgets
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import explain_validity

from ISAT.ui.annotation_data_tools import Ui_Dialog

# 等宽字体候选：Windows 优先，Linux 兜底（DejaVu/Liberation 为 Linux 自带）
MONO_FONT_CANDIDATES = (
    "Consolas",
    "Cascadia Mono",
    "Courier New",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Menlo",
    "Monaco",
)


class ScanThread(QtCore.QThread):
    """Background worker scanning a label directory.

    Emits ``progress(done, total, filename)`` while parsing and
    ``finished(result)`` when done.  ``result`` is a dict::

        {
            "counts": {category: {"polygon": int, "obb": int}},
            "valid_files": int,
            "skipped_files": int,
            "empty_files": int,
            "total_objects": int,
            "empty_objects": int,
            "problems": [(filename, level, obj_index, message), ...],
        }

    ``level`` is one of "Error" / "Warning" / "Info".  A file is *valid*
    when it parses as JSON and carries ``info.description == "ISAT"``.
    """

    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(dict)

    def __init__(self, label_dir: str, recursive: bool = False, parent=None):
        super(ScanThread, self).__init__(parent)
        self.label_dir = label_dir
        self.recursive = recursive

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ordinal(index):
        return "object #{}".format(index)

    @staticmethod
    def _is_rectangle(points):
        """True when 4 points form a rectangle (order-independent).

        Uses the squared edge-length spectrum: 4 equal short edges, 2 equal
        diagonals, diagonal^2 == 2 * edge^2.
        """
        if len(points) != 4:
            return False
        dists = []
        for i in range(4):
            for j in range(i + 1, 4):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dists.append(dx * dx + dy * dy)
        dists.sort()
        s = dists[0]
        if s <= 1e-9:
            return False
        if any(abs(d - s) > 1e-6 * s for d in dists[:4]):
            return False
        if abs(dists[4] - dists[5]) > 1e-6 * s:
            return False
        if abs(dists[4] - 2 * s) > 1e-6 * s:
            return False
        return True

    # ------------------------------------------------------------------
    #  scanning
    # ------------------------------------------------------------------

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

    def _scan_file(self, path):
        """Return (counts_delta, problems, is_valid, n_objects, empty_objs).

        ``empty_objs`` counts objects whose segmentation has fewer than 3
        points (empty or degenerate) inside a valid file — they are not real
        annotations and are excluded from the per-category counts.
        """
        problems = []
        counts_delta = {}
        empty_objs = 0
        valid_objs = 0
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        info = dataset.get("info", {})
        if info.get("description", "") != "ISAT":
            problems.append(
                (os.path.basename(path), "Error", -1, "Not an ISAT json.")
            )
            return counts_delta, problems, False, 0, 0
        width = info.get("width")
        height = info.get("height")
        objects = dataset.get("objects", [])
        if len(objects) == 0:
            problems.append(
                (os.path.basename(path), "Info", -1, "ISAT file with no objects.")
            )
            return counts_delta, problems, True, 0, 0

        for obj_index, obj in enumerate(objects):
            tag = self._ordinal(obj_index)
            category = obj.get("category", "unknow")
            shape_type = obj.get("shape_type", "polygon")
            segmentation = obj.get("segmentation", [])
            points = [[float(p[0]), float(p[1])] for p in segmentation]

            if len(points) == 0:
                # 空目标：对象存在但没有任何分割点
                problems.append(
                    (os.path.basename(path), "Error", obj_index,
                     "{} empty target (no segmentation points).".format(tag))
                )
                empty_objs += 1
                continue
            if len(points) < 3:
                problems.append(
                    (os.path.basename(path), "Error", obj_index,
                     "{} polygon error. Vertex < 3.".format(tag))
                )
                empty_objs += 1
                continue
            if shape_type == "obb":
                if not self._is_rectangle(points):
                    problems.append(
                        (os.path.basename(path), "Warning", obj_index,
                         "{} OBB is not a rectangle.".format(tag))
                    )
            else:
                polyon = ShapelyPolygon(points)
                if not polyon.is_valid:
                    problems.append(
                        (os.path.basename(path), "Warning", obj_index,
                         "{} polygon invalid. {}".format(tag, explain_validity(polyon)))
                    )

            if width is not None and height is not None:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                xmin, ymin = min(xs), min(ys)
                xmax, ymax = max(xs), max(ys)
                if xmin < 0 or xmax > width or ymin < 0 or ymax > height:
                    problems.append(
                        (os.path.basename(path), "Warning", obj_index,
                         "{} polygon warning. Out of the image.".format(tag))
                    )

            entry = counts_delta.setdefault(category, {"polygon": 0, "obb": 0})
            if shape_type == "obb":
                entry["obb"] += 1
            else:
                entry["polygon"] += 1
            valid_objs += 1

        return counts_delta, problems, True, valid_objs, empty_objs

    def run(self):
        counts = {}
        valid_files = 0
        skipped_files = 0
        empty_files = 0
        total_objects = 0
        empty_objects = 0
        problems = []

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
                counts_delta, file_problems, is_valid, n_objects, empty_objs = (
                    self._scan_file(path)
                )
            except Exception:
                # unreadable / corrupt / unexpected shape
                problems.append(
                    (os.path.basename(path), "Error", -1, "Broken json file.")
                )
                skipped_files += 1
                continue

            problems.extend(file_problems)
            if not is_valid:
                skipped_files += 1
                continue
            if n_objects == 0:
                empty_files += 1
                valid_files += 1
                continue
            empty_objects += empty_objs
            for category, entry in counts_delta.items():
                for key in ("polygon", "obb"):
                    counts.setdefault(category, {"polygon": 0, "obb": 0})[
                        key
                    ] += entry[key]
            total_objects += n_objects
            valid_files += 1

        self.finished.emit(
            {
                "counts": counts,
                "valid_files": valid_files,
                "skipped_files": skipped_files,
                "empty_files": empty_files,
                "total_objects": total_objects,
                "empty_objects": empty_objects,
                "problems": problems,
            }
        )


class AnnotationDataToolsDialog(QtWidgets.QDialog, Ui_Dialog):
    """Merged statistics + validator tool.

    A single directory input drives one background :class:`ScanThread`; the
    two tabs present the results differently — a per-category count table
    (Statistics) and a per-problem log (Validator).
    """

    def __init__(self, parent, mainwindow):
        super(AnnotationDataToolsDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.thread = None
        self.setupUi(self)

        self.pushButton_browse.clicked.connect(self._browse)
        self.pushButton_start.clicked.connect(self._start)
        self.pushButton_cancel.clicked.connect(self._cancel)
        self.pushButton_close.clicked.connect(self.close)

        self.tableWidget.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )

        # 问题清单使用等宽字体对齐（Windows/Linux 双平台候选）
        self.textBrowser.setFont(self._mono_font())

    @staticmethod
    def _mono_font():
        """Pick the first available monospace family, else generic fallback."""
        available = set(QtGui.QFontDatabase().families())
        for name in MONO_FONT_CANDIDATES:
            if name in available:
                font = QtGui.QFont(name)
                font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
                return font
        font = QtGui.QFont()
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        return font

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
        self.textBrowser.clear()
        self.label_summary_stats.setText("")
        self.label_summary_validator.setText("")
        self.progressBar.setValue(0)
        self.label_status.setText("Enumerating label files...")
        self.pushButton_start.setEnabled(False)
        self.pushButton_cancel.setEnabled(True)
        self.pushButton_close.setEnabled(False)

        self.thread = ScanThread(
            label_dir, recursive=self.checkBox_recursive.isChecked(), parent=self
        )
        self.thread.progress.connect(self._on_progress)
        self.thread.finished.connect(self._on_finished)
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
        self.label_status.setText("Finished.")

        # ---- statistics tab ----
        counts = result["counts"]
        categories = sorted(counts.keys())
        self.tableWidget.setRowCount(len(categories) + 1)
        total_polygon = 0
        total_obb = 0
        for row, category in enumerate(categories):
            entry = counts[category]
            total_polygon += entry["polygon"]
            total_obb += entry["obb"]
            self._set_row(row, category, entry["polygon"], entry["obb"])
        self._set_row(len(categories), "Total", total_polygon, total_obb)
        font = self.tableWidget.font()
        font.setBold(True)
        for col in range(4):
            item = self.tableWidget.item(len(categories), col)
            item.setFont(font)
        self.label_summary_stats.setText(
            "Valid ISAT files: {}    Empty files (no objects): {}    "
            "Skipped: {}    Total objects: {}    "
            "Empty objects (ignored): {}".format(
                result["valid_files"],
                result["empty_files"],
                result["skipped_files"],
                result["total_objects"],
                result["empty_objects"],
            )
        )

        # ---- validator tab ----
        problems = result["problems"]
        if problems:
            error_count = sum(1 for p in problems if p[1] == "Error")
            warning_count = sum(1 for p in problems if p[1] == "Warning")
            info_count = sum(1 for p in problems if p[1] == "Info")
            for filename, level, _idx, message in problems:
                self.textBrowser.append(
                    "{:>7} | {} | {}".format(level, filename, message)
                )
            self.label_summary_validator.setText(
                "Problems: {} Error, {} Warning, {} Info".format(
                    error_count, warning_count, info_count
                )
            )
        else:
            self.textBrowser.append("No problems found.")
            self.label_summary_validator.setText("No problems found.")

        self.pushButton_start.setEnabled(True)
        self.pushButton_cancel.setEnabled(False)
        self.pushButton_close.setEnabled(True)

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
