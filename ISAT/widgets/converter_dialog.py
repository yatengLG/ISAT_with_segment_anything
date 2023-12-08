# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from ISAT.ui.Converter_dialog import Ui_Dialog
from ISAT.scripts.isat import ISAT
from ISAT.scripts.coco import COCO
from ISAT.scripts.yolo import YOLO
from ISAT.scripts.labelme import LABELME
from ISAT.scripts.voc import VOC
from ISAT.scripts.voc_detection import VOCDetect
import os
import yaml
import imgviz


class Converter(QThread, ISAT):
    message = pyqtSignal(int, int, str)
    def __init__(self):
        super(Converter, self).__init__()
        self.cancel = False
        self.isat_json_root = None

    def run(self):
        raise NotImplementedError

    def load_from_isat(self):
        json_files = [file for file in os.listdir(self.isat_json_root) if file.endswith('.json')]
        num_json_files = len(json_files)
        for index, file in enumerate(json_files):
            if self.cancel:
                self.message.emit(-1, -1, ' ' * 18 + '| -- Cancel --')
                return

            self.message.emit(index, num_json_files,
                              '{:>8d}/{:<8d} | Loading ISAT json {}'.format(index + 1, num_json_files, file))
            try:
                anno = self._load_one_isat_json(os.path.join(self.isat_json_root, file))
                self.annos[self.remove_file_suffix(file)] = anno
            except Exception as e:
                self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))

        # load cates
        self.message.emit(-1, -1, ' ' * 18 + '| Loading cats.')
        if os.path.exists(os.path.join(self.isat_json_root, 'isat.yaml')):
            cates = []
            with open(os.path.join(self.isat_json_root, 'isat.yaml'), 'rb')as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            for label in cfg.get('label', []):
                cates.append(label.get('name'))
            self.cates = tuple(cates)
        else:
            cates = set()
            for _, anno in self.annos.items():
                for obj in anno.objs:
                    cates.add(obj.category)
            cates = list(cates)
            cates.sort()
            self.cates = tuple(cates)
        self.message.emit(-1, -1, ' ' * 18 + '| Loaded.')
        return True

    def convert_to_isat(self):
        self.message.emit(-1, -1, ' ' * 18 + '| Start convert to ISAT.')
        num_annos = len(self.annos)

        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, ' ' * 18 + '| -- Cancel --')
                return

            self.message.emit(index + 1, num_annos,
                              '{:>8d}/{:<8d} | Converting to {}'.format(index + 1, num_annos,
                                                                        name_without_suffix + 'json'))
            try:
                self._save_one_isat_json(anno,
                                         os.path.join(self.isat_json_root, '{}.json'.format(name_without_suffix)))

            except Exception as e:
                self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))

        # 类别文件
        cmap = imgviz.label_colormap()
        cates = sorted(self.cates)
        categories = []
        for index, cat in enumerate(cates):
            r, g, b = cmap[index + 1]
            categories.append({
                'name': cat if isinstance(cat, str) else str(cat),
                'color': "#{:02x}{:02x}{:02x}".format(r, g, b)
            })
        s = yaml.dump({'label': categories})
        with open(os.path.join(self.isat_json_root, 'isat.yaml'), 'w') as f:
            f.write(s)

        return True


class COCOConverter(Converter, COCO):
    def __init__(self):
        super(COCOConverter, self).__init__()
        self.coco_json_path = None
        self.isat_json_root = None
        self.keep_crowd = False
        self.coco2isat = True

    def run(self):
        if self.coco_json_path is None or self.isat_json_root is None:
            return

        if self.coco2isat:
            self.coco_to_isat()
        else:
            self.isat_to_coco()

    def coco_to_isat(self):
        self.message.emit(-1, -1, ' ' * 18 + '| -- COCO to ISAT --')

        # load from coco
        self.message.emit(-1, -1, ' ' * 18 + '| Loading from COCO.')
        try:
            self.read_from_coco(self.coco_json_path)
            self.message.emit(-1, -1, ' ' * 18 + '| Loaded.')

        except Exception as e:
            self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))
            return

        # convert to isat
        self.convert_to_isat()
        self.message.emit(-1, -1, ' ' * 18 + '| Finished.')
        return True

    def isat_to_coco(self):
        self.message.emit(-1, -1, ' ' * 18 + '| -- ISAT to COCO --')

        # load from isat
        self.load_from_isat()

        # convert to coco
        self.message.emit(-1, -1, ' ' * 18 + '| Saving COCO json to {}.'.format('', '', self.coco_json_path))
        try:
            self.save_to_coco(self.coco_json_path)
        except Exception as e:
            self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))
        self.message.emit(-1, -1, ' ' * 18 + '| Finished.')
        return True


class YOLOConverter(Converter, YOLO):
    def __init__(self):
        super(YOLOConverter, self).__init__()
        self.yolo_images_root = None
        self.yolo_txt_root = None
        self.yolo_category_file = None
        self.isat_json_root = None

        self.yolo2isat = True

    def run(self):
        if self.yolo2isat:
            if self.yolo_images_root is None or self.yolo_txt_root is None or self.isat_json_root is None:
                return
            self.yolo_to_isat()
        else:
            if self.yolo_txt_root is None or self.isat_json_root is None:
                return
            self.isat_to_yolo()

    def yolo_to_isat(self):
        self.message.emit(-1, -1, ' ' * 18 + '| -- YOLO to ISAT --')

        # load category file
        if self.yolo_category_file is not None:
            self.message.emit(-1, -1, ' ' * 18 + '| Loading YOLO categorys.')
            class_dict = {}
            with open(self.yolo_category_file, 'r') as f:
                lines = f.readlines()
                for index, line in enumerate(lines):
                    class_dict[index] = line.rstrip('\n')
                    self.message.emit(-1, -1, ' ' * 18 + '| {} : {}'.format(index, line.rstrip('\n')))
            self.message.emit(-1, -1, ' ' * 18 + '| Loaded.')
        else:
            class_dict = None

        # load from yolo
        self.message.emit(-1, -1, ' ' * 18 + '| Loading YOLO txts.')
        img_files = os.listdir(self.yolo_images_root)
        num_img_files = len(img_files)

        for index, img_name in enumerate(img_files):
            if self.cancel:
                self.message.emit(-1, -1, ' ' * 18 + '| -- Cancel --')
                return
            name_without_suffix = self.remove_file_suffix(img_name)
            txt_path = os.path.join(self.yolo_txt_root, name_without_suffix + '.txt')
            image_path = os.path.join(self.yolo_images_root, img_name)
            if not os.path.exists(txt_path):
                continue

            self.message.emit(index + 1, num_img_files,
                              '{:>8d}/{:<8d} | Loading from {}'.format(index + 1, num_img_files,
                                                                       name_without_suffix + '.txt'))
            try:
                anno = self._load_one_yolo_txt(image_path, txt_path, class_dict)
                self.annos[name_without_suffix] = anno
            except Exception as e:
                self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))

        # cates
        if class_dict is not None:
            self.cates = tuple(class_dict.values())
        else:
            class_set = set()
            for _, anno in self.annos.items():
                for obj in anno.objs:
                    class_set.add(obj.category)
            class_set = list(class_set)
            class_set.sort()
            self.cates = tuple(class_set)

        # convert to isat
        self.convert_to_isat()
        self.message.emit(-1, -1, ' ' * 18 + '| Finished.')

        return True

    def isat_to_yolo(self):
        self.message.emit(-1, -1, ' ' * 18 + '| -- ISAT to YOLO --')

        # load from isat
        self.load_from_isat()

        # save to yolo
        cates_index_dict = {cat: index for index, cat in enumerate(self.cates)}
        with open(os.path.join(self.yolo_txt_root, 'classification.txt'), 'w') as f:
            for cat in self.cates:
                f.write('{}\n'.format(cat))
        num_annos = len(self.annos)
        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, ' ' * 18 + '| -- Cancel --')
                return

            txt_path = os.path.join(self.yolo_txt_root, name_without_suffix + '.txt')
            try:
                self.message.emit(index + 1, num_annos,
                                  '{:>8d}/{:<8d} | Save yolo txt to {}'.format(index + 1, num_annos,
                                                                           name_without_suffix + '.txt'))
                self._save_one_yolo_txt(anno, txt_path, cates_index_dict)
            except Exception as e:
                self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))
        self.message.emit(-1, -1, ' ' * 18 + '| Finished.')
        return True


class LABELMEConverter(Converter, LABELME):
    def __init__(self):
        super(LABELMEConverter, self).__init__()
        self.isat_json_root = None
        self.labelme_json_root = None
        self.labelme2isat = True

    def run(self):
        if self.labelme_json_root is None or self.isat_json_root is None:
            return
        if self.labelme2isat:
            self.labelme_to_isat()
        else:
            self.isat_to_labelme()


    def labelme_to_isat(self):
        self.message.emit(-1, -1, ' ' * 18 + '| -- ISAT to ISAT --')

        # load from labelme
        self.message.emit(-1, -1, ' ' * 18 + '| Loading from LABELME.')
        json_files = [file for file in os.listdir(self.labelme_json_root) if file.endswith('.json')]
        num_json_files = len(json_files)
        for index, file in enumerate(json_files):
            if self.cancel:
                self.message.emit(-1, -1, ' ' * 18 + '| -- Cancel --')
                return

            name_without_suffix = self.remove_file_suffix(file)

            self.message.emit(index + 1, num_json_files,
                              '{:>8d}/{:<8d} | Loading from {}'.format(index + 1, num_json_files,
                                                                       name_without_suffix + '.json'))
            try:
                anno = self._load_one_labelme_json(os.path.join(self.labelme_json_root, file))
                self.annos[name_without_suffix] = anno
            except Exception as e:
                self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))

        # cats
        class_set = set()
        for _, anno in self.annos.items():
            for obj in anno.objs:
                class_set.add(obj.category)

        class_set = list(class_set)
        class_set.sort()
        self.cates = tuple(class_set)

        # save to isat
        self.convert_to_isat()
        self.message.emit(-1, -1, ' ' * 18 + '| Finished.')

        return True

    def isat_to_labelme(self):

        # load from isat
        self.load_from_isat()

        # save to labelme
        num_annos = len(self.annos.items())
        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, ' ' * 18 + '| -- Cancel --')
                return

            json_path = os.path.join(self.labelme_json_root, name_without_suffix + '.json')
            self.message.emit(index + 1, num_annos,
                              '{:>8d}/{:<8d} | Save LABELME json to {}'.format(index + 1, num_annos,
                                                                       name_without_suffix + '.json'))
            try:
                self._save_one_labelme_json(anno, json_path)
            except Exception as e:
                self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))

        self.message.emit(-1, -1, ' ' * 18 + '| Finished.')
        return True


class VOCConverter(Converter, VOC):
    def __init__(self):
        super(VOCConverter, self).__init__()
        self.isat_json_root = None
        self.voc_png_root = None

    def run(self):
        if self.isat_json_root is not None and self.voc_png_root is not None:
            self.isat_to_voc()

    def isat_to_voc(self):
        self.message.emit(-1, -1, ' ' * 18 + '| -- ISAT to VOC --')

        # load from isat
        self.load_from_isat()

        # cmap
        cmap = imgviz.label_colormap()

        if not self.is_instance:
            category_index_dict = {}
            with open(os.path.join(self.voc_png_root, 'classification.txt'), 'w') as f:
                for index, cate in enumerate(self.cates):
                    category_index_dict[cate] = index
                    f.write('{}\n'.format(cate))
        else:
            category_index_dict = None
        # save to voc
        num_annos = len(self.annos)
        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, ' ' * 18 + '| -- Cancel --')
                return

            png_path = os.path.join(self.voc_png_root, name_without_suffix + '.png')
            try:
                self.message.emit(index + 1, num_annos,
                                  '{:>8d}/{:<8d} | Save voc png to {}'.format(index + 1, num_annos,
                                                                           name_without_suffix + '.png'))
                self._save_one_voc_png(anno, png_path, cmap, category_index_dict)
            except Exception as e:
                self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))

        self.message.emit(-1, -1, ' ' * 18 + '| Finished.')
        return True


class VOCConverterForDetection(Converter, VOCDetect):
    def __init__(self):
        super(VOCConverterForDetection, self).__init__()
        self.isat_json_root = None
        self.voc_xml_root = None

    def run(self):
        if self.isat_json_root is not None and self.voc_xml_root is not None:
            self.isat_to_voc_for_detection()

    def isat_to_voc_for_detection(self):
        self.message.emit(-1, -1, ' ' * 18 + '| -- ISAT to VOC for Detection --')

        # load from isat
        self.load_from_isat()

        # save to xml
        num_annos = len(self.annos)
        for index, (name_without_suffix, anno) in enumerate(self.annos.items()):
            if self.cancel:
                self.message.emit(-1, -1, ' ' * 18 + '| -- Cancel --')
                return

            xml_path = os.path.join(self.voc_xml_root, name_without_suffix + '.xml')
            try:
                self.message.emit(index + 1, num_annos,
                                  '{:>8d}/{:<8d} | Save voc png to {}'.format(index + 1, num_annos,
                                                                           name_without_suffix + '.png'))
                self._save_one_voc_xml(anno, xml_path)

            except Exception as e:
                self.message.emit(-1, -1, ' ' * 18 + '| Error: {}.'.format(e))
        self.message.emit(-1, -1, ' ' * 18 + '| Finished.')
        return True


class ConverterDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(ConverterDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.converter = None
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.init_connect()

    def apply(self):
        self.tabWidget.setEnabled(False)
        self.pushButton_convert.setEnabled(False)
        self.progressBar.reset()
        self.textBrowser.clear()

        if self.tabWidget.currentWidget() == self.tab_COCO:
            # COCO
            if self.toolBox_coco.currentWidget() == self.toolbox_item_coco2isat:
                # coco2isat
                if self.lineEdit_coco2isat_coco_json_path.text() and self.lineEdit_coco2isat_isat_json_root.text():
                    self.converter = COCOConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.coco2isat = True
                    self.converter.coco_json_path = self.lineEdit_coco2isat_coco_json_path.text()
                    self.converter.isat_json_root = self.lineEdit_coco2isat_isat_json_root.text()
                    self.converter.keep_crowd = self.checkBox_coco2isat_keep_crowd.isChecked()

                    self.converter.run()
                else:
                    QtWidgets.QMessageBox.warning(self, '', '')

            elif self.toolBox_coco.currentWidget() == self.toolbox_item_isat2coco:
                # isat2coco
                if self.lineEdit_isat2coco_isat_json_root.text() and self.lineEdit_isat2coco_coco_json_path.text():
                    self.converter = COCOConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.coco2isat = False
                    self.converter.isat_json_root = self.lineEdit_isat2coco_isat_json_root.text()
                    self.converter.coco_json_path = self.lineEdit_isat2coco_coco_json_path.text()
                    self.converter.keep_crowd = self.checkBox_isat2coco_keep_crowd.isChecked()

                    self.converter.run()
                else:
                    QtWidgets.QMessageBox.warning(self, '', '')

        elif self.tabWidget.currentWidget() == self.tab_YOLO:
            # YOLO
            if self.toolBox_yolo.currentWidget() == self.toolbox_item_yolo2isat:
                # yolo2isat
                if self.lineEdit_yolo2isat_yolo_image_root.text() and self.lineEdit_yolo2isat_yolo_txt_root.text() and self.lineEdit_yolo2isat_isat_json_root.text():
                    self.converter = YOLOConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.yolo2isat = True
                    self.converter.yolo_images_root = self.lineEdit_yolo2isat_yolo_image_root.text()
                    self.converter.yolo_txt_root = self.lineEdit_yolo2isat_yolo_txt_root.text()
                    self.converter.isat_json_root = self.lineEdit_yolo2isat_isat_json_root.text()
                    if self.lineEdit_yolo2isat_yolo_cate_path.text():
                        self.converter.yolo_category_file = self.lineEdit_yolo2isat_yolo_cate_path.text()

                    self.converter.run()
                else:
                    QtWidgets.QMessageBox.warning(self, '', '')

            elif self.toolBox_yolo.currentWidget() == self.toolbox_item_isat2yolo:
                # isat2yolo
                if self.lineEdit_isat2yolo_isat_json_root.text() and self.lineEdit_isat2yolo_yolo_txt_root.text():
                    self.converter = YOLOConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.yolo2isat = False
                    self.converter.isat_json_root = self.lineEdit_isat2yolo_isat_json_root.text()
                    self.converter.yolo_txt_root = self.lineEdit_isat2yolo_yolo_txt_root.text()

                    self.converter.run()
                else:
                    QtWidgets.QMessageBox.warning(self, '', '')

        elif self.tabWidget.currentWidget() == self.tab_LABELME:
            # LABELME
            if self.toolBox_labelme.currentWidget() == self.toolbox_item_labelme2isat:
                # labelme2isat
                if self.lineEdit_labelme2isat_labelme_json_root.text() and self.lineEdit_labelme2isat_isat_json_root.text():
                    self.converter = LABELMEConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.labelme2isat = True
                    self.converter.labelme_json_root = self.lineEdit_labelme2isat_labelme_json_root.text()
                    self.converter.isat_json_root = self.lineEdit_labelme2isat_isat_json_root.text()

                    self.converter.run()
                else:
                    QtWidgets.QMessageBox.warning(self, '', '')

            elif self.toolBox_labelme.currentWidget() == self.toolbox_item_isat2labelme:
                # isat2labelme
                if self.lineEdit_isat2labelme_isat_json_root.text() and self.lineEdit_isat2labelme_labelme_json_root.text():
                    self.converter = LABELMEConverter()
                    self.converter.message.connect(self.print_message)
                    self.converter.labelme2isat = False
                    self.converter.isat_json_root = self.lineEdit_isat2labelme_isat_json_root.text()
                    self.converter.labelme_json_root = self.lineEdit_isat2labelme_labelme_json_root.text()

                    self.converter.run()
                else:
                    QtWidgets.QMessageBox.warning(self, '', '')

        elif self.tabWidget.currentWidget() == self.tab_VOC:
            # VOC
            if self.lineEdit_isat2voc_isat_json_root.text() and self.lineEdit_isat2voc_voc_png_root.text():
                self.converter = VOCConverter()
                self.converter.message.connect(self.print_message)
                self.converter.isat_json_root = self.lineEdit_isat2voc_isat_json_root.text()
                self.converter.voc_png_root = self.lineEdit_isat2voc_voc_png_root.text()
                self.converter.is_instance = self.checkBox_is_instance.isChecked()
                self.converter.run()
            else:
                QtWidgets.QMessageBox.warning(self, '', '')

        elif self.tabWidget.currentWidget() == self.tab_VOC_DETECTION:
            # VOC_DETECTION
            if self.lineEdit_isat2vocod_isat_json_root.text() and self.lineEdit_isat2vocod_voc_xml_root.text():
                self.converter = VOCConverterForDetection()
                self.converter.message.connect(self.print_message)
                self.converter.isat_json_root = self.lineEdit_isat2vocod_isat_json_root.text()
                self.converter.voc_xml_root = self.lineEdit_isat2vocod_voc_xml_root.text()
                self.converter.run()
            else:
                QtWidgets.QMessageBox.warning(self, '', '')

        else:
            pass
        self.tabWidget.setEnabled(True)
        self.pushButton_convert.setEnabled(True)

    def cancel(self):
        try:
            self.converter.cancel = True
        except: pass

    def open_file(self):

        # coco2isat
        if self.sender() == self.pushButton_coco2isat_coco_json_path:
            filter="json (*.json)"
            lineEdit = self.lineEdit_coco2isat_coco_json_path
        # isat2coco
        elif self.sender() == self.pushButton_isat2coco_coco_json_path:
            filter = "json (*.json)"
            lineEdit = self.lineEdit_isat2coco_coco_json_path
        elif self.sender() == self.pushButton_yolo2isat_yolo_cate_path:
            filter = "txt (*.txt)"
            lineEdit = self.lineEdit_yolo2isat_yolo_cate_path
        else:
            filter = ''
            lineEdit = None

        path, suffix = QtWidgets.QFileDialog.getOpenFileName(self, caption='Open file', filter=filter)

        if path:
            if lineEdit is not None:
                lineEdit.setText(path)
        else:
            if lineEdit is not None:
                lineEdit.setText(path)

    def save_file(self):
        path, suffix = QtWidgets.QFileDialog.getSaveFileName(self, caption='Save file', filter="json (*.json)")
        # coco2isat
        if self.sender() == self.pushButton_coco2isat_coco_json_path:
            lineEdit = self.lineEdit_coco2isat_coco_json_path
        # isat2coco
        elif self.sender() == self.pushButton_isat2coco_coco_json_path:
            lineEdit = self.lineEdit_isat2coco_coco_json_path
        else:
            lineEdit = None

        if path:
            if not path.endswith('.json'):
                path += '.json'
            if lineEdit is not None:
                lineEdit.setText(path)
        else:
            if lineEdit is not None:
                lineEdit.clear()


    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption='Open dir')
        # coco2isat
        if self.sender() == self.pushButton_coco2isat_isat_json_root:
            lineEdit = self.lineEdit_coco2isat_isat_json_root
        # isat2coco
        elif self.sender() == self.pushButton_isat2coco_isat_json_root:
            lineEdit = self.lineEdit_isat2coco_isat_json_root
        # yolo2isat
        elif self.sender() == self.pushButton_yolo2isat_yolo_image_root:
            lineEdit = self.lineEdit_yolo2isat_yolo_image_root
        elif self.sender() == self.pushButton_yolo2isat_yolo_txt_root:
            lineEdit = self.lineEdit_yolo2isat_yolo_txt_root
        elif self.sender() == self.pushButton_yolo2isat_isat_json_root:
            lineEdit = self.lineEdit_yolo2isat_isat_json_root
        # isat2yolo
        elif self.sender() == self.pushButton_isat2yolo_isat_json_root:
            lineEdit = self.lineEdit_isat2yolo_isat_json_root
        elif self.sender() == self.pushButton_isat2yolo_yolo_txt_root:
            lineEdit = self.lineEdit_isat2yolo_yolo_txt_root
        # labelme2isat
        elif self.sender() == self.pushButton_labelme2isat_labelme_json_root:
            lineEdit = self.lineEdit_labelme2isat_labelme_json_root
        elif self.sender() == self.pushButton_labelme2isat_isat_json_root:
            lineEdit = self.lineEdit_labelme2isat_isat_json_root
        # isat2labelme
        elif self.sender() == self.pushButton_isat2labelme_isat_json_root:
            lineEdit = self.lineEdit_isat2labelme_isat_json_root
        elif self.sender() == self.pushButton_isat2labelme_labelme_json_root:
            lineEdit = self.lineEdit_isat2labelme_labelme_json_root
        # isat2voc
        elif self.sender() == self.pushButton_isat2voc_isat_json_root:
            lineEdit = self.lineEdit_isat2voc_isat_json_root
        elif self.sender() == self.pushButton_isat2voc_voc_png_root:
            lineEdit = self.lineEdit_isat2voc_voc_png_root
        # isat2vocod
        elif self.sender() == self.pushButton_isat2vocod_isat_json_root:
            lineEdit = self.lineEdit_isat2vocod_isat_json_root
        elif self.sender() == self.pushButton_isat2vocod_voc_xml_root:
            lineEdit = self.lineEdit_isat2vocod_voc_xml_root
        else:
            lineEdit = None

        if dir:
            if lineEdit is not None:
                lineEdit.setText(dir)
        else:
            if lineEdit is not None:
                lineEdit.clear()

    def save_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption='Save dir')
        # coco2isat
        if self.sender() == self.pushButton_coco2isat_isat_json_root:
            lineEdit = self.lineEdit_coco2isat_isat_json_root
        # isat2coco
        elif self.sender() == self.pushButton_isat2coco_isat_json_root:
            lineEdit = self.lineEdit_isat2coco_isat_json_root
        # yolo2isat
        elif self.sender() == self.pushButton_yolo2isat_yolo_image_root:
            lineEdit = self.lineEdit_yolo2isat_yolo_image_root
        elif self.sender() == self.pushButton_yolo2isat_yolo_txt_root:
            lineEdit = self.lineEdit_yolo2isat_yolo_txt_root
        elif self.sender() == self.pushButton_yolo2isat_isat_json_root:
            lineEdit = self.lineEdit_yolo2isat_isat_json_root
        # isat2yolo
        elif self.sender() == self.pushButton_isat2yolo_isat_json_root:
            lineEdit = self.lineEdit_isat2yolo_isat_json_root
        elif self.sender() == self.pushButton_isat2yolo_yolo_txt_root:
            lineEdit = self.lineEdit_isat2yolo_yolo_txt_root
        # labelme2isat
        elif self.sender() == self.pushButton_labelme2isat_labelme_json_root:
            lineEdit = self.lineEdit_labelme2isat_labelme_json_root
        elif self.sender() == self.pushButton_labelme2isat_isat_json_root:
            lineEdit = self.lineEdit_labelme2isat_isat_json_root
        # isat2labelme
        elif self.sender() == self.pushButton_isat2labelme_isat_json_root:
            lineEdit = self.lineEdit_isat2labelme_isat_json_root
        elif self.sender() == self.pushButton_isat2labelme_labelme_json_root:
            lineEdit = self.lineEdit_isat2labelme_labelme_json_root
        # isat2voc
        elif self.sender() == self.pushButton_isat2voc_isat_json_root:
            lineEdit = self.lineEdit_isat2voc_isat_json_root
        elif self.sender() == self.pushButton_isat2voc_voc_png_root:
            lineEdit = self.lineEdit_isat2voc_voc_png_root
        # isat2vocod
        elif self.sender() == self.pushButton_isat2vocod_isat_json_root:
            lineEdit = self.lineEdit_isat2vocod_isat_json_root
        elif self.sender() == self.pushButton_isat2vocod_voc_xml_root:
            lineEdit = self.lineEdit_isat2vocod_voc_xml_root
        else:
            lineEdit = None

        if dir:
            self.save_root = dir
            if lineEdit is not None:
                lineEdit.setText(dir)
        else:
            if lineEdit is not None:
                lineEdit.clear()

    def print_message(self, index, all, message):
        if all > 0:
            self.progressBar.setMaximum(all)
        if index > 0:
            self.progressBar.setValue(index)
        if message:
            self.textBrowser.append(message)

    def init_connect(self):
        self.pushButton_convert.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)
        # coco2isat
        self.pushButton_coco2isat_coco_json_path.clicked.connect(self.open_file)
        self.pushButton_coco2isat_isat_json_root.clicked.connect(self.save_dir)
        # isat2coco
        self.pushButton_isat2coco_isat_json_root.clicked.connect(self.open_dir)
        self.pushButton_isat2coco_coco_json_path.clicked.connect(self.save_file)
        # yolo2isat
        self.pushButton_yolo2isat_yolo_image_root.clicked.connect(self.open_dir)
        self.pushButton_yolo2isat_yolo_txt_root.clicked.connect(self.open_dir)
        self.pushButton_yolo2isat_yolo_cate_path.clicked.connect(self.open_file)
        self.pushButton_yolo2isat_isat_json_root.clicked.connect(self.save_dir)
        # isat2yolo
        self.pushButton_isat2yolo_isat_json_root.clicked.connect(self.open_dir)
        self.pushButton_isat2yolo_yolo_txt_root.clicked.connect(self.save_dir)
        # labelme2isat
        self.pushButton_labelme2isat_labelme_json_root.clicked.connect(self.open_dir)
        self.pushButton_labelme2isat_isat_json_root.clicked.connect(self.save_dir)
        # isat2labelme
        self.pushButton_isat2labelme_isat_json_root.clicked.connect(self.open_dir)
        self.pushButton_isat2labelme_labelme_json_root.clicked.connect(self.save_dir)
        # voc
        self.pushButton_isat2voc_isat_json_root.clicked.connect(self.open_dir)
        self.pushButton_isat2voc_voc_png_root.clicked.connect(self.save_dir)
        # voc-object-detection
        self.pushButton_isat2vocod_isat_json_root.clicked.connect(self.open_dir)
        self.pushButton_isat2vocod_voc_xml_root.clicked.connect(self.save_dir)

