# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ui.MainWindow import Ui_MainWindow
from widgets.setting_dialog import SettingDialog
from widgets.category_choice_dialog import CategoryChoiceDialog
from widgets.category_edit_dialog import CategoryEditDialog
from widgets.labels_dock_widget import LabelsDockWidget
from widgets.files_dock_widget import FilesDockWidget
from widgets.info_dock_widget import InfoDockWidget
from widgets.right_button_menu import RightButtonMenu
from widgets.shortcut_dialog import ShortcutDialog
from widgets.about_dialog import AboutDialog
from widgets.ISAT_to_VOC_dialog import ISATtoVOCDialog
from widgets.ISAT_to_COCO_dialog import ISATtoCOCODialog
from widgets.ISAT_to_LABELME_dialog import ISATtoLabelMeDialog
from widgets.COCO_to_ISAT_dialog import COCOtoISATDialog
from widgets.canvas import AnnotationScene, AnnotationView
from configs import STATUSMode, MAPMode, load_config, save_config, CONFIG_FILE, DEFAULT_CONFIG_FILE
from annotation import Object, Annotation
from widgets.polygon import Polygon
import os
from PIL import Image
import functools
import imgviz
from segment_any.segment_any import SegAny
from segment_any.gpu_resource import GPUResource_Thread, osplatform
import icons_rc


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.init_ui()
        self.image_root: str = None
        self.label_root:str = None

        self.files_list: list = []
        self.current_index = None
        self.current_file_index: int = None

        self.config_file = CONFIG_FILE if os.path.exists(CONFIG_FILE) else DEFAULT_CONFIG_FILE
        self.saved = True
        self.can_be_annotated = True
        self.load_finished = False
        self.polygons:list = []

        self.png_palette = None # 图像拥有调色盘，说明是单通道的标注png文件
        self.instance_cmap = imgviz.label_colormap()
        self.map_mode = MAPMode.LABEL
        # 标注目标
        self.current_label:Annotation = None

        self.reload_cfg()

        self.init_connect()
        self.reset_action()
        self.init_segment_anything()

    def init_segment_anything(self):
        if os.path.exists('./segment_any/sam_vit_h_4b8939.pth'):
            self.statusbar.showMessage('Find the checkpoint named {}.'.format('sam_vit_h_4b8939.pth'))
            self.segany = SegAny('./segment_any/sam_vit_h_4b8939.pth')
            self.use_segment_anything = True
        elif os.path.exists('./segment_any/sam_vit_l_0b3195.pth'):
            self.statusbar.showMessage('Find the checkpoint named {}.'.format('sam_vit_l_0b3195.pth'))
            self.segany = SegAny('./segment_any/sam_vit_l_0b3195.pth')
            self.use_segment_anything = True
        elif os.path.exists('./segment_any/sam_vit_b_01ec64.pth'):
            self.statusbar.showMessage('Find the checkpoint named {}.'.format('sam_vit_b_01ec64.pth'))
            self.segany = SegAny('./segment_any/sam_vit_b_01ec64.pth')
            self.use_segment_anything = True
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'The checkpoint of [Segment anything] not existed. If you want use quick annotate, please download from {}'.format('https://github.com/facebookresearch/segment-anything#model-checkpoints'))
            self.use_segment_anything = False

        if self.use_segment_anything:
            if self.segany.device != 'cpu':
                self.gpu_resource_thread = GPUResource_Thread()
                self.gpu_resource_thread.message.connect(self.labelGPUResource.setText)
                self.gpu_resource_thread.start()
            else:
                self.labelGPUResource.setText('cpu')
        else:
            self.labelGPUResource.setText('segment anything unused.')

    def init_ui(self):
        #
        self.setting_dialog = SettingDialog(parent=self, mainwindow=self)

        self.labels_dock_widget = LabelsDockWidget(mainwindow=self)
        self.labels_dock.setWidget(self.labels_dock_widget)

        self.files_dock_widget = FilesDockWidget(mainwindow=self)
        self.files_dock.setWidget(self.files_dock_widget)

        self.info_dock_widget = InfoDockWidget(mainwindow=self)
        self.info_dock.setWidget(self.info_dock_widget)

        self.scene = AnnotationScene(mainwindow=self)
        self.category_choice_widget = CategoryChoiceDialog(self, mainwindow=self, scene=self.scene)
        self.category_edit_widget = CategoryEditDialog(self, self, self.scene)

        self.ISAT_to_VOC_dialog = ISATtoVOCDialog(self, mainwindow=self)
        self.ISAT_to_COCO_dialog = ISATtoCOCODialog(self, mainwindow=self)
        self.ISAT_to_LABELME_dialog = ISATtoLabelMeDialog(self, mainwindow=self)
        self.COCO_to_ISAT_dialog = COCOtoISATDialog(self, mainwindow=self)

        self.view = AnnotationView(parent=self)
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)

        self.right_button_menu = RightButtonMenu(mainwindow=self)
        self.right_button_menu.addAction(self.actionEdit)
        self.right_button_menu.addAction(self.actionTo_top)
        self.right_button_menu.addAction(self.actionTo_bottom)

        self.shortcut_dialog = ShortcutDialog(self)
        self.about_dialog = AboutDialog(self)

        self.labelGPUResource = QtWidgets.QLabel('')
        self.labelGPUResource.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelGPUResource.setFixedWidth(180)
        self.statusbar.addPermanentWidget(self.labelGPUResource)

        self.labelCoord = QtWidgets.QLabel('')
        self.labelCoord.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelCoord.setFixedWidth(150)
        self.statusbar.addPermanentWidget(self.labelCoord)

        self.labelData = QtWidgets.QLabel('')
        self.labelData.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelData.setFixedWidth(150)
        self.statusbar.addPermanentWidget(self.labelData)

        self.trans = QtCore.QTranslator()

    def translate(self, language='zh'):
        if language == 'zh':
            self.trans.load('ui/zh_CN')
        else:
            self.trans.load('ui/en')
        self.actionChinese.setChecked(language=='zh')
        self.actionEnglish.setChecked(language=='en')
        _app = QtWidgets.QApplication.instance()
        _app.installTranslator(self.trans)
        self.retranslateUi(self)
        self.info_dock_widget.retranslateUi(self.info_dock_widget)
        self.labels_dock_widget.retranslateUi(self.labels_dock_widget)
        self.files_dock_widget.retranslateUi(self.files_dock_widget)
        self.category_choice_widget.retranslateUi(self.category_choice_widget)
        self.category_edit_widget.retranslateUi(self.category_edit_widget)
        self.setting_dialog.retranslateUi(self.setting_dialog)
        self.about_dialog.retranslateUi(self.about_dialog)
        self.shortcut_dialog.retranslateUi(self.shortcut_dialog)
        self.ISAT_to_VOC_dialog.retranslateUi(self.ISAT_to_VOC_dialog)
        self.ISAT_to_COCO_dialog.retranslateUi(self.ISAT_to_COCO_dialog)
        self.ISAT_to_LABELME_dialog.retranslateUi(self.ISAT_to_LABELME_dialog)
        self.COCO_to_ISAT_dialog.retranslateUi(self.COCO_to_ISAT_dialog)

    def translate_to_chinese(self):
        self.translate('zh')
        self.cfg['language'] = 'zh'

    def translate_to_english(self):
        self.translate('en')
        self.cfg['language'] = 'en'

    def reload_cfg(self):
        self.cfg = load_config(self.config_file)
        label_dict_list = self.cfg.get('label', [])
        d = {}
        for label_dict in label_dict_list:
            category = label_dict.get('name', 'unknow')
            color = label_dict.get('color', '#000000')
            d[category] = color
        self.category_color_dict = d

        if self.current_index is not None:
            self.show_image(self.current_index)

        language = self.cfg.get('language', 'en')
        self.translate(language)

    def set_saved_state(self, is_saved:bool):
        self.saved = is_saved
        if self.files_list is not None and self.current_index is not None:

            if is_saved:
                self.setWindowTitle(self.current_label.label_path)
            else:
                self.setWindowTitle('*{}'.format(self.current_label.label_path))

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dir:
            self.files_list.clear()
            self.files_dock_widget.listWidget.clear()

            files = []
            suffixs = tuple(['{}'.format(fmt.data().decode('ascii').lower()) for fmt in QtGui.QImageReader.supportedImageFormats()])
            for f in os.listdir(dir):
                if f.lower().endswith(suffixs):
                    # f = os.path.join(dir, f)
                    files.append(f)
            files = sorted(files)
            self.files_list = files

            self.files_dock_widget.update_widget()

        self.current_index = 0

        self.image_root = dir
        self.actionOpen_dir.setStatusTip("Image root: {}".format(self.image_root))
        if self.label_root is None:
            self.label_root = dir
            self.actionSave_dir.setStatusTip("Label root: {}".format(self.label_root))

        self.show_image(self.current_index)

    def save_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dir:
            self.label_root = dir
            self.actionSave_dir.setStatusTip("Label root: {}".format(self.label_root))

        # 刷新图片
        if self.current_index is not None:
            self.show_image(self.current_index)

    def save(self):
        if self.current_label is None:
            return
        self.current_label.objects.clear()
        for polygon in self.polygons:
            object = polygon.to_object()
            self.current_label.objects.append(object)

        self.current_label.note = self.info_dock_widget.lineEdit_note.text()
        self.current_label.save_annotation()
        # 保存标注文件的同时保存一份isat配置文件
        self.save_cfg(os.path.join(self.label_root, 'isat.yaml'))
        self.set_saved_state(True)

    def show_image(self, index:int):
        self.reset_action()
        self.current_label = None
        self.load_finished = False
        self.saved = True
        if not -1 < index < len(self.files_list):
            self.scene.clear()
            self.scene.setSceneRect(QtCore.QRectF())
            return
        try:
            self.polygons.clear()
            self.labels_dock_widget.listWidget.clear()
            self.scene.cancel_draw()
            file_path = os.path.join(self.image_root, self.files_list[index])
            image_data = Image.open(file_path)

            self.png_palette = image_data.getpalette()
            if self.png_palette is not None and file_path.endswith('.png'):
                self.statusbar.showMessage('This is a label file.')
                self.can_be_annotated = False

            else:
                self.can_be_annotated = True

            if self.can_be_annotated:
                self.actionSegment_anything.setEnabled(self.use_segment_anything)
                self.actionPolygon.setEnabled(True)
                self.actionSave.setEnabled(True)
                self.actionBit_map.setEnabled(True)
                self.actionBackspace.setEnabled(True)
                self.actionFinish.setEnabled(True)
                self.actionCancel.setEnabled(True)
                self.actionVisible.setEnabled(True)
            else:
                self.actionSegment_anything.setEnabled(False)
                self.actionPolygon.setEnabled(False)
                self.actionSave.setEnabled(False)
                self.actionBit_map.setEnabled(False)
                self.actionBackspace.setEnabled(False)
                self.actionFinish.setEnabled(False)
                self.actionCancel.setEnabled(False)
                self.actionVisible.setEnabled(False)

            self.scene.load_image(file_path)
            self.view.zoomfit()

            # load label
            if self.can_be_annotated:
                _, name = os.path.split(file_path)
                label_path = os.path.join(self.label_root, '.'.join(name.split('.')[:-1]) + '.json')
                self.current_label = Annotation(file_path, label_path)
                # 载入数据
                self.current_label.load_annotation()

                for object in self.current_label.objects:
                    polygon = Polygon()
                    self.scene.addItem(polygon)
                    polygon.load_object(object)
                    self.polygons.append(polygon)

            if self.current_label is not None:
                self.setWindowTitle('{}'.format(self.current_label.label_path))
            else:
                self.setWindowTitle('{}'.format(file_path))

            self.labels_dock_widget.update_listwidget()
            self.info_dock_widget.update_widget()
            self.files_dock_widget.set_select(index)
            self.current_index = index
            self.files_dock_widget.label_current.setText('{}'.format(self.current_index+1))
            self.load_finished = True

        except Exception as e:
            print(e)
        finally:
            if self.current_index > 0:
                self.actionPrev.setEnabled(True)
            else:
                self.actionPrev.setEnabled(False)

            if self.current_index < len(self.files_list) - 1:
                self.actionNext.setEnabled(True)
            else:
                self.actionNext.setEnabled(False)

    def prev_image(self):
        if self.scene.mode != STATUSMode.VIEW:
            return
        if self.current_index is None:
            return
        if not self.saved:
            result = QtWidgets.QMessageBox.question(self, 'Warning', 'Proceed without saved?', QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                return
        self.current_index = self.current_index - 1
        if self.current_index < 0:
            self.current_index = 0
            QtWidgets.QMessageBox.warning(self, 'Warning', 'This is the first picture.')
        else:
            self.show_image(self.current_index)

    def next_image(self):
        if self.scene.mode != STATUSMode.VIEW:
            return
        if self.current_index is None:
            return
        if not self.saved:
            result = QtWidgets.QMessageBox.question(self, 'Warning', 'Proceed without saved?', QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                return
        self.current_index = self.current_index + 1
        if self.current_index > len(self.files_list)-1:
            self.current_index = len(self.files_list)-1
            QtWidgets.QMessageBox.warning(self, 'Warning', 'This is the last picture.')
        else:
            self.show_image(self.current_index)

    def jump_to(self):
        index = self.files_dock_widget.lineEdit_jump.text()
        if index:
            if not index.isdigit():
                if index in self.files_list:
                    index = self.files_list.index(index)+1
                else:
                    QtWidgets.QMessageBox.warning(self, 'Warning', 'Don`t exist image named: {}'.format(index))
                    self.files_dock_widget.lineEdit_jump.clear()
                    return
            index = int(index)-1
            if 0 <= index < len(self.files_list):
                self.show_image(index)
                self.files_dock_widget.lineEdit_jump.clear()
            else:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'Index must be in [1, {}].'.format(len(self.files_list)))
                self.files_dock_widget.lineEdit_jump.clear()
                self.files_dock_widget.lineEdit_jump.clearFocus()
                return

    def cancel_draw(self):
        self.scene.cancel_draw()

    def setting(self):
        self.setting_dialog.load_cfg()
        self.setting_dialog.show()

    def add_new_object(self, category, group, segmentation, area, layer, bbox):
        if self.current_label is None:
            return
        object = Object(category=category, group=group, segmentation=segmentation, area=area, layer=layer, bbox=bbox)
        self.current_label.objects.append(object)

    def delete_object(self, index:int):
        if 0 <= index < len(self.current_label.objects):
            del self.current_label.objects[index]

    def change_bit_map(self):
        self.set_labels_visible(True)
        if self.scene.mode == STATUSMode.CREATE:
            self.scene.cancel_draw()
        if self.map_mode == MAPMode.LABEL:
            # to semantic
            for polygon in self.polygons:
                polygon.setEnabled(False)
                for vertex in polygon.vertexs:
                    vertex.setVisible(False)
                polygon.change_color(QtGui.QColor(self.category_color_dict.get(polygon.category, '#000000')))
                polygon.color.setAlpha(255)
                polygon.setBrush(polygon.color)
            self.labels_dock_widget.listWidget.setEnabled(False)
            self.labels_dock_widget.checkBox_visible.setEnabled(False)
            self.actionSegment_anything.setEnabled(False)
            self.actionPolygon.setEnabled(False)
            self.actionVisible.setEnabled(False)
            self.map_mode = MAPMode.SEMANTIC
            semantic_icon = QtGui.QIcon()
            semantic_icon.addPixmap(QtGui.QPixmap(":/icon/icons/semantic.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionBit_map.setIcon(semantic_icon)

        elif self.map_mode == MAPMode.SEMANTIC:
            # to instance
            for polygon in self.polygons:
                polygon.setEnabled(False)
                for vertex in polygon.vertexs:
                    vertex.setVisible(False)
                if polygon.group != '':
                    rgb = self.instance_cmap[int(polygon.group)]
                else:
                    rgb = self.instance_cmap[0]
                polygon.change_color(QtGui.QColor(rgb[0], rgb[1], rgb[2], 255))
                polygon.color.setAlpha(255)
                polygon.setBrush(polygon.color)
            self.labels_dock_widget.listWidget.setEnabled(False)
            self.labels_dock_widget.checkBox_visible.setEnabled(False)
            self.actionSegment_anything.setEnabled(False)
            self.actionPolygon.setEnabled(False)
            self.actionVisible.setEnabled(False)
            self.map_mode = MAPMode.INSTANCE
            instance_icon = QtGui.QIcon()
            instance_icon.addPixmap(QtGui.QPixmap(":/icon/icons/instance.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionBit_map.setIcon(instance_icon)

        elif self.map_mode == MAPMode.INSTANCE:
            # to label
            for polygon in self.polygons:
                polygon.setEnabled(True)
                for vertex in polygon.vertexs:
                    # vertex.setEnabled(True)
                    vertex.setVisible(polygon.isVisible())
                polygon.change_color(QtGui.QColor(self.category_color_dict.get(polygon.category, '#000000')))
                polygon.color.setAlpha(polygon.nohover_alpha)
                polygon.setBrush(polygon.color)
            self.labels_dock_widget.listWidget.setEnabled(True)
            self.labels_dock_widget.checkBox_visible.setEnabled(True)
            self.actionSegment_anything.setEnabled(self.use_segment_anything)
            self.actionPolygon.setEnabled(True)
            self.actionVisible.setEnabled(True)
            self.map_mode = MAPMode.LABEL
            label_icon = QtGui.QIcon()
            label_icon.addPixmap(QtGui.QPixmap(":/icon/icons/照片_pic.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionBit_map.setIcon(label_icon)
        else:
            pass

    def set_labels_visible(self, visible=None):
        if visible is None:
            visible = not self.labels_dock_widget.checkBox_visible.isChecked()
        self.labels_dock_widget.checkBox_visible.setChecked(visible)
        self.labels_dock_widget.set_all_polygon_visible(visible)

    def ISAT_to_VOC(self):
        self.ISAT_to_VOC_dialog.reset_gui()
        self.ISAT_to_VOC_dialog.show()

    def ISAT_to_COCO(self):
        self.ISAT_to_COCO_dialog.reset_gui()
        self.ISAT_to_COCO_dialog.show()

    def ISAT_to_LABELME(self):
        self.ISAT_to_LABELME_dialog.reset_gui()
        self.ISAT_to_LABELME_dialog.show()

    def COCO_to_ISAT(self):
        self.COCO_to_ISAT_dialog.reset_gui()
        self.COCO_to_ISAT_dialog.show()

    def help(self):
        self.shortcut_dialog.show()

    def about(self):
        self.about_dialog.show()

    def save_cfg(self, config_file):
        save_config(self.cfg, config_file)

    def exit(self):
        self.save_cfg(self.config_file)
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        self.exit()

    def init_connect(self):
        self.actionOpen_dir.triggered.connect(self.open_dir)
        self.actionSave_dir.triggered.connect(self.save_dir)
        self.actionPrev.triggered.connect(self.prev_image)
        self.actionNext.triggered.connect(self.next_image)
        self.actionSetting.triggered.connect(self.setting)
        self.actionExit.triggered.connect(self.exit)

        self.actionSegment_anything.triggered.connect(self.scene.start_segment_anything)
        self.actionPolygon.triggered.connect(self.scene.start_draw_polygon)
        self.actionCancel.triggered.connect(self.scene.cancel_draw)
        self.actionBackspace.triggered.connect(self.scene.backspace)
        self.actionFinish.triggered.connect(self.scene.finish_draw)
        self.actionEdit.triggered.connect(self.scene.edit_polygon)
        self.actionDelete.triggered.connect(self.scene.delete_selected_graph)
        self.actionSave.triggered.connect(self.save)
        self.actionTo_top.triggered.connect(self.scene.move_polygon_to_top)
        self.actionTo_bottom.triggered.connect(self.scene.move_polygon_to_bottom)

        self.actionZoom_in.triggered.connect(self.view.zoom_in)
        self.actionZoom_out.triggered.connect(self.view.zoom_out)
        self.actionFit_wiondow.triggered.connect(self.view.zoomfit)
        self.actionBit_map.triggered.connect(self.change_bit_map)
        self.actionVisible.triggered.connect(functools.partial(self.set_labels_visible, None))

        self.actionToVOC.triggered.connect(self.ISAT_to_VOC)
        self.actionToCOCO.triggered.connect(self.ISAT_to_COCO)
        self.actionTo_LabelMe.triggered.connect(self.ISAT_to_LABELME)
        self.actionFromCOCO.triggered.connect(self.COCO_to_ISAT)

        self.actionShortcut.triggered.connect(self.help)
        self.actionAbout.triggered.connect(self.about)

        self.actionChinese.triggered.connect(self.translate_to_chinese)
        self.actionEnglish.triggered.connect(self.translate_to_english)

        self.labels_dock_widget.listWidget.doubleClicked.connect(self.scene.edit_polygon)

    def reset_action(self):
        self.actionPrev.setEnabled(False)
        self.actionNext.setEnabled(False)
        self.actionSegment_anything.setEnabled(False)
        self.actionPolygon.setEnabled(False)
        self.actionEdit.setEnabled(False)
        self.actionDelete.setEnabled(False)
        self.actionSave.setEnabled(False)
        self.actionTo_top.setEnabled(False)
        self.actionTo_bottom.setEnabled(False)
        self.actionBit_map.setChecked(False)
        self.actionBit_map.setEnabled(False)
