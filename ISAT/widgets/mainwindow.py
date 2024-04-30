# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ISAT.ui.MainWindow import Ui_MainWindow
from ISAT.widgets.setting_dialog import SettingDialog
from ISAT.widgets.category_choice_dialog import CategoryChoiceDialog
from ISAT.widgets.category_edit_dialog import CategoryEditDialog
from ISAT.widgets.category_dock_widget import CategoriesDockWidget
from ISAT.widgets.annos_dock_widget import AnnosDockWidget
from ISAT.widgets.files_dock_widget import FilesDockWidget
from ISAT.widgets.info_dock_widget import InfoDockWidget
from ISAT.widgets.right_button_menu import RightButtonMenu
from ISAT.widgets.shortcut_dialog import ShortcutDialog
from ISAT.widgets.about_dialog import AboutDialog
from ISAT.widgets.converter_dialog import ConverterDialog
from ISAT.widgets.auto_segment_dialog import AutoSegmentDialog
from ISAT.widgets.model_manager_dialog import ModelManagerDialog
from ISAT.widgets.canvas import AnnotationScene, AnnotationView
from ISAT.configs import STATUSMode, MAPMode, load_config, save_config, CONFIG_FILE, SOFTWARE_CONFIG_FILE, CHECKPOINT_PATH, ISAT_ROOT
from ISAT.annotation import Object, Annotation
from ISAT.widgets.polygon import Polygon, PromptPoint
import os
from PIL import Image
import functools
import imgviz
from ISAT.segment_any.segment_any import SegAny
from ISAT.segment_any.gpu_resource import GPUResource_Thread, osplatform
import ISAT.ui.icons_rc
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import torch
import cv2  # 调整图像饱和度

class SegAnyThread(QThread):
    tag = pyqtSignal(int, int, str)
    def __init__(self, mainwindow):
        super(SegAnyThread, self).__init__()
        self.mainwindow = mainwindow
        self.results_dict = {}
        self.index = None

    @torch.no_grad()
    def sam_encoder(self, image):
        torch.cuda.empty_cache()

        input_image = self.mainwindow.segany.predictor_with_point_prompt.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.mainwindow.segany.predictor_with_point_prompt.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        original_size = image.shape[:2]
        input_size = tuple(input_image_torch.shape[-2:])

        input_image = self.mainwindow.segany.predictor_with_point_prompt.model.preprocess(input_image_torch)
        features = self.mainwindow.segany.predictor_with_point_prompt.model.image_encoder(input_image)
        return features, original_size, input_size

    def run(self):
        if self.index is not None:

            # 需要缓存特征的图像索引，可以自行更改缓存策略
            indexs = [self.index]
            if self.index + 1 < len(self.mainwindow.files_list):
                indexs += [self.index + 1]
            if self.index - 1 > -1:
                indexs += [self.index - 1]

            # 先删除不需要的旧特征
            features_ks = list(self.results_dict.keys())
            for k in features_ks:
                if k not in indexs:
                    try:
                        del self.results_dict[k]
                        self.tag.emit(k, 0, '')  # 删除
                    except:
                        pass

            for index in indexs:
                if index not in self.results_dict:
                    self.tag.emit(index, 2, '')    # 进行

                    image_path = os.path.join(self.mainwindow.image_root, self.mainwindow.files_list[index])
                    self.results_dict[index] = {}
                    image_data = np.array(Image.open(image_path))
                    try:
                        features, original_size, input_size = self.sam_encoder(image_data)
                    except Exception as e:
                        self.tag.emit(index, 3, '{}'.format(e))  # error
                        del self.results_dict[index]
                        continue

                    self.results_dict[index]['features'] = features
                    self.results_dict[index]['original_size'] = original_size
                    self.results_dict[index]['input_size'] = input_size

                    self.tag.emit(index, 1, '')    # 完成

                    torch.cuda.empty_cache()
                else:
                    self.tag.emit(index, 1, '')


class InitSegAnyThread(QThread):
    tag = pyqtSignal(bool)
    def __init__(self, mainwindow):
        super(InitSegAnyThread, self).__init__()
        self.mainwindow = mainwindow
        self.model_path:str = None

    def run(self):
        if self.model_path is not None:
            try:
                self.mainwindow.segany = SegAny(self.model_path, self.mainwindow.cfg['software']['use_bfloat16'])
                self.tag.emit(True)
            except Exception as e:
                print(e)
                self.tag.emit(False)
        else:
            self.tag.emit(False)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.image_root: str = None
        self.label_root:str = None

        self.files_list: list = []
        self.current_index = None
        self.current_file_index: int = None

        self.current_label = '__background__'
        self.current_group = 1

        self.config_file = CONFIG_FILE
        self.software_config_file = SOFTWARE_CONFIG_FILE

        self.saved = True
        self.can_be_annotated = True
        self.load_finished = False
        self.polygons:list = []

        self.png_palette = None # 图像拥有调色盘，说明是单通道的标注png文件
        self.instance_cmap = imgviz.label_colormap()
        self.map_mode = MAPMode.LABEL
        # 标注目标
        self.current_label:Annotation = None
        self.use_segment_anything = False
        self.gpu_resource_thread = None

        # 新增 手动/自动 group选择
        self.group_select_mode = 'auto'
        self.init_ui()
        self.reload_cfg()

        self.init_connect()
        self.reset_action()

        # sam初始化线程，大模型加载较慢
        self.init_segany_thread = InitSegAnyThread(self)
        self.init_segany_thread.tag.connect(self.init_sam_finish)

    def init_segment_anything(self, model_name=None):
        if not self.saved:
            result = QtWidgets.QMessageBox.question(self, 'Warning', 'Proceed without saved?', QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                if isinstance(self.sender(), QtWidgets.QAction):
                    self.sender().setChecked(False)
                return
        if model_name is None:
            if self.use_segment_anything:
                model_name = os.path.split(self.segany.checkpoint)[-1]
            else:
                return
        # 等待sam线程完成
        try:
            self.seganythread.wait()
            self.seganythread.results_dict.clear()
        except:
            # if isinstance(self.sender(), QtWidgets.QAction):
            #     self.sender().setChecked(False)
            pass

        if model_name == '':
            self.use_segment_anything = False
            return
        model_path = os.path.join(CHECKPOINT_PATH, model_name)
        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The checkpoint of [Segment anything] not existed. If you want use quick annotate, please download from {}'.format(
                                              'https://github.com/facebookresearch/segment-anything#model-checkpoints'))

            self.use_segment_anything = False
            return

        self.model_manager_dialog.waiting_icon.setVisible(True)
        self.model_manager_dialog.setEnabled(False)
        self.init_segany_thread.model_path = model_path
        self.init_segany_thread.start()

    def init_sam_finish(self, tag:bool):

        if tag:
            self.use_segment_anything = True
            if self.use_segment_anything:
                if self.segany.device != 'cpu':
                    if self.gpu_resource_thread is None:
                        self.gpu_resource_thread = GPUResource_Thread()
                        self.gpu_resource_thread.message.connect(self.labelGPUResource.setText)
                        self.gpu_resource_thread.start()
                else:
                    self.labelGPUResource.setText('cpu')
            else:
                self.labelGPUResource.setText('segment anything unused.')

            self.seganythread = SegAnyThread(self)
            self.seganythread.tag.connect(self.sam_encoder_finish)
            self.seganythread.start()

            if self.current_index is not None:
                self.show_image(self.current_index)
        else:
            self.use_segment_anything = False

            for row in range(self.model_manager_dialog.tableWidget.rowCount()):
                used_checkbox = self.model_manager_dialog.tableWidget.cellWidget(row, 0)
                if used_checkbox.isEnabled():
                    used_checkbox.setChecked(False)
            self.pushButton_Model_manage.setStatusTip('')

        self.model_manager_dialog.waiting_icon.setVisible(False)
        self.model_manager_dialog.setEnabled(True)
        self.model_manager_dialog.update_gui()

    def sam_encoder_finish(self, index:int, state:int, message:str):
        if state == 1:  # 识别完
            # 如果当前图片刚识别完，需刷新segany状态
            if self.current_index == index:
                self.SeganyEnabled()

        # 图片识别状态刷新
        if state == 1: color = '#00FF00'
        elif state == 0: color = '#999999'
        elif state == 2: color = '#FFFF00'
        elif state == 3:
            color = '#999999'
            if index == self.current_index:
                QtWidgets.QMessageBox.warning(self, 'warning','SAM not support the image: {}\nError: {}'.format(self.files_list[index], message))

        else: color = '#999999'

        if index == self.current_index:
            self.files_dock_widget.label_current_state.setStyleSheet("background-color: {};".format(color))
        elif index == self.current_index - 1:
            self.files_dock_widget.label_prev_state.setStyleSheet("background-color: {};".format(color))
        elif index == self.current_index + 1:
            self.files_dock_widget.label_next_state.setStyleSheet("background-color: {};".format(color))
        else:
            pass

        # item = self.files_dock_widget.listWidget_files.item(index)
        # widget = self.files_dock_widget.listWidget_files.itemWidget(item)
        # if widget is not None:
        #     state_color = widget.findChild(QtWidgets.QLabel, 'state_color')
        #     state_color.setStyleSheet("background-color: {};".format(color))

    def SeganyEnabled(self):
        """
        segany激活
        判断当前图片是否缓存特征图，如果存在特征图，设置segany参数，并开放半自动标注
        :return:
        """
        if not self.use_segment_anything:
            self.pushButton_Segment_anything.setEnabled(False)
            return

        results = self.seganythread.results_dict.get(self.current_index, {})
        features = results.get('features', None)
        original_size = results.get('original_size', None)
        input_size = results.get('input_size', None)

        if features is not None and original_size is not None and input_size is not None:
            if self.segany.model_source == 'sam_hq':
                features, interm_features = features
                self.segany.predictor_with_point_prompt.interm_features = interm_features
            self.segany.predictor_with_point_prompt.features = features
            self.segany.predictor_with_point_prompt.original_size = original_size
            self.segany.predictor_with_point_prompt.input_size = input_size
            self.segany.predictor_with_point_prompt.is_image_set = True
            self.pushButton_Segment_anything.setEnabled(True)
        else:
            self.segany.predictor_with_point_prompt.reset_image()
            self.pushButton_Segment_anything.setEnabled(False)

    def init_ui(self):
        self.m_flag = False
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        #
        self.categories_dock_widget = CategoriesDockWidget(mainwindow=self)
        self.annos_dock_widget = AnnosDockWidget(mainwindow=self)
        self.files_dock_widget = FilesDockWidget(mainwindow=self)
        self.info_dock_widget = InfoDockWidget(mainwindow=self)

        self.scene = AnnotationScene(mainwindow=self)
        self.category_choice_widget = CategoryChoiceDialog(self, mainwindow=self, scene=self.scene)
        self.category_edit_widget = CategoryEditDialog(self, self, self.scene)

        self.view = AnnotationView(parent=self)
        self.view.setScene(self.scene)

        # 整体可调整区域, 横向布局
        self.splitter.addWidget(self.categories_dock_widget)
        self.splitter.addWidget(self.view)
        # 可调整区域右侧
        right_widget = QtWidgets.QWidget()
        right_widget_layout = QtWidgets.QVBoxLayout()
        right_widget_layout.setContentsMargins(1, 1, 1, 1)
        right_widget.setLayout(right_widget_layout)
        right_widget_layout.addWidget(self.info_dock_widget)
        right_widget_layout.addWidget(self.annos_dock_widget)
        right_widget_layout.addWidget(self.files_dock_widget)
        self.splitter.addWidget(right_widget)

        self.splitter.setSizes([250, 780, 250])

        # 新增 group 选择 快捷键
        self.next_group_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Tab"), self)
        self.prev_group_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("`"), self)
        self.next_group_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self.prev_group_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        # 新增手动/自动 选择group
        self.next_group_shortcut.activated.connect(self.annos_dock_widget.go_to_next_group)
        self.prev_group_shortcut.activated.connect(self.annos_dock_widget.go_to_prev_group)           

        # 右键菜单
        self.right_button_menu = RightButtonMenu(mainwindow=self)
        self.right_button_menu.addAction(self.actionEdit)
        self.right_button_menu.addAction(self.actionTo_top)
        self.right_button_menu.addAction(self.actionTo_bottom)
        self.right_button_menu.addAction(self.actionDelete)

        self.Converter_dialog = ConverterDialog(self, mainwindow=self)
        self.auto_segment_dialog = AutoSegmentDialog(self, self)
        self.setting_dialog = SettingDialog(parent=self, mainwindow=self)
        self.model_manager_dialog = ModelManagerDialog(self, self)
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

    def update_menuSAM(self):
        #
        self.menuSAM_model.clear()
        self.menuSAM_model.addAction(self.actionModel_manage)
        model_names = sorted(
            [pth for pth in os.listdir(CHECKPOINT_PATH) if pth.endswith('.pth') or pth.endswith('.pt')])
        self.pths_actions = {}
        for model_name in model_names:
            action = QtWidgets.QAction(self)
            action.setObjectName("actionZoom_in")
            action.triggered.connect(functools.partial(self.init_segment_anything, model_name))
            action.setText("{}".format(model_name))
            action.setCheckable(True)

            self.pths_actions[model_name] = action
            self.menuSAM_model.addAction(action)

    def translate(self, language:str=None):
        if language is None:
            if self.cfg['software']['language'] == 'zh':
                language = 'en'
            elif self.cfg['software']['language'] == 'en':
                language = 'zh'
        if language == 'zh':
            self.trans.load(os.path.join(ISAT_ROOT, 'ui/zh_CN'))
        else:
            self.trans.load(os.path.join(ISAT_ROOT, 'ui/en'))

        self.cfg['software']['language'] = language
        self.save_software_cfg()

        # 刷新图标
        label_icon = QtGui.QIcon()
        icon = ':/icons/icons/english_32.svg' if language == 'zh' else ':/icons/icons/chinese_32.svg'
        label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Translate.setIcon(label_icon)

        _app = QtWidgets.QApplication.instance()
        _app.installTranslator(self.trans)
        self.retranslateUi(self)
        self.info_dock_widget.retranslateUi(self.info_dock_widget)
        self.annos_dock_widget.retranslateUi(self.annos_dock_widget)
        self.files_dock_widget.retranslateUi(self.files_dock_widget)
        self.category_choice_widget.retranslateUi(self.category_choice_widget)
        self.category_edit_widget.retranslateUi(self.category_edit_widget)
        self.setting_dialog.retranslateUi(self.setting_dialog)
        self.model_manager_dialog.retranslateUi(self.model_manager_dialog)
        self.about_dialog.retranslateUi(self.about_dialog)
        self.shortcut_dialog.retranslateUi(self.shortcut_dialog)
        self.Converter_dialog.retranslateUi(self.Converter_dialog)
        self.auto_segment_dialog.retranslateUi(self.auto_segment_dialog)

    def translate_to_chinese(self):
        self.translate('zh')


    def translate_to_english(self):
        self.translate('en')


    def reload_cfg(self):
        # 软件配置
        self.cfg = load_config(self.software_config_file)
        if self.cfg is None:
            self.cfg = {}
        software_cfg = self.cfg.get('software', {})
        self.cfg['software'] = software_cfg

        language = software_cfg.get('language', 'en')
        self.cfg['software']['language'] = language
        self.translate(language)

        contour_mode = software_cfg.get('contour_mode', 'max_only')
        self.cfg['software']['contour_mode'] = contour_mode
        self.change_contour_mode(contour_mode)

        mask_alpha = software_cfg.get('mask_alpha', 0.5)
        self.cfg['software']['mask_alpha'] = mask_alpha
        self.slider_mask_aplha.setValue(int(mask_alpha*10))

        vertex_size = software_cfg.get('vertex_size', 2)
        self.cfg['software']['vertex_size'] = int(vertex_size)
        self.slider_vertex_size.setValue(vertex_size)

        show_prompt = software_cfg.get('show_prompt', False)
        self.cfg['software']['show_prompt'] = bool(show_prompt)
        label_icon = QtGui.QIcon()
        icon = ':/icons/icons/circular_solid_32.svg' if self.cfg['software']['show_prompt'] else ':/icons/icons/circular_dash_32.svg'
        label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_show_prompt.setIcon(label_icon)

        show_edge = software_cfg.get('show_edge', True)
        self.cfg['software']['show_edge'] = bool(show_edge)
        label_icon = QtGui.QIcon()
        icon = ':/icons/icons/edge_solid_32.svg' if self.cfg['software']['show_edge'] else ':/icons/icons/edge_dash_32.svg'
        label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_show_edge.setIcon(label_icon)

        use_polydp = software_cfg.get('use_polydp', True)
        self.cfg['software']['use_polydp'] = bool(use_polydp)
        label_icon = QtGui.QIcon()
        icon = ':/icons/icons/circular_with_four_point_32.svg' if self.cfg['software']['use_polydp'] else ':/icons/icons/circular_with_eight_point_32.svg'
        label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_use_polydp.setIcon(label_icon)

        use_bfloat16 = software_cfg.get('use_bfloat16', False)
        self.cfg['software']['use_bfloat16'] = bool(use_bfloat16)
        self.model_manager_dialog.update_gui()

        # 类别
        self.cfg.update(load_config(self.config_file))
        label_dict_list = self.cfg.get('label', [])
        if len(label_dict_list) < 1 or label_dict_list[0].get('name', 'unknow') != '__background__':
            label_dict_list.insert(0, {'color': '#000000', 'name': '__background__'})

        d = {}
        for label_dict in label_dict_list:
            category = label_dict.get('name', 'unknow')
            color = label_dict.get('color', '#000000')
            d[category] = color
        self.category_color_dict = d

        self.categories_dock_widget.update_widget()

        if self.current_index is not None:
            self.show_image(self.current_index)

    def set_saved_state(self, is_saved:bool):
        self.saved = is_saved
        if self.files_list is not None and self.current_index is not None:

            if is_saved:
                self.label_Title.setText(self.current_label.label_path)
            else:
                self.label_Title.setText('*{}'.format(self.current_label.label_path))

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if not dir:
            return

        # 等待sam线程退出，并清空特征缓存
        if self.use_segment_anything:
            self.seganythread.wait()
            self.seganythread.results_dict.clear()

        self.files_list.clear()
        self.files_dock_widget.listWidget_files.clear()

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
        self.pushButton_Open_dir.setStatusTip("Image root: {}".format(self.image_root))

        self.label_root = dir
        self.pushButton_Save_dir.setStatusTip("Label root: {}".format(self.label_root))

        if os.path.exists(os.path.join(dir, 'isat.yaml')):
            # load setting yaml
            self.config_file = os.path.join(dir, 'isat.yaml')
            self.reload_cfg()

        self.show_image(self.current_index)

    def save_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if not dir:
            return

        self.label_root = dir
        self.pushButton_Save_dir.setStatusTip("Label root: {}".format(self.label_root))
        # load setting yaml
        if os.path.exists(os.path.join(dir, 'isat.yaml')):
            self.config_file = os.path.join(dir, 'isat.yaml')
            self.reload_cfg()
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

    def update_group_display(self):
        self.categories_dock_widget.lineEdit_currentGroup.setText(str(self.current_group))

    def show_image(self, index:int, zoomfit:bool=True):
        self.reset_action()
        self.change_bit_map_to_label()
        self.annos_dock_widget.comboBox_group_select.setCurrentIndex(0)
        # 
        self.files_dock_widget.label_prev_state.setStyleSheet("background-color: {};".format('#999999'))
        self.files_dock_widget.label_current_state.setStyleSheet("background-color: {};".format('#999999'))
        self.files_dock_widget.label_next_state.setStyleSheet("background-color: {};".format('#999999'))

        self.current_label = None
        self.load_finished = False
        self.saved = True
        if not -1 < index < len(self.files_list):
            self.scene.clear()
            self.scene.setSceneRect(QtCore.QRectF())
            return
        try:
            self.polygons.clear()
            self.annos_dock_widget.listWidget_annos.clear()
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
                self.pushButton_Polygon.setEnabled(True)
                self.pushButton_Save.setEnabled(True)
                self.pushButton_Bit_map.setEnabled(True)
                self.pushButton_Backspace.setEnabled(True)
                self.pushButton_Finish.setEnabled(True)
                self.pushButton_Cancel.setEnabled(True)
                self.pushButton_Visible.setEnabled(True)
            else:
                self.pushButton_Polygon.setEnabled(False)
                self.pushButton_Save.setEnabled(False)
                self.pushButton_Bit_map.setEnabled(False)
                self.pushButton_Backspace.setEnabled(False)
                self.pushButton_Finish.setEnabled(False)
                self.pushButton_Cancel.setEnabled(False)
                self.pushButton_Visible.setEnabled(False)

            self.scene.load_image(file_path)

            if self.use_segment_anything and self.can_be_annotated:
                self.segany.reset_image()
                self.seganythread.index = index
                self.seganythread.start()
                self.SeganyEnabled()
            if zoomfit:
                self.view.zoomfit()

            # load label
            if self.can_be_annotated:
                self.current_group = 1
                _, name = os.path.split(file_path)
                label_path = os.path.join(self.label_root, '.'.join(name.split('.')[:-1]) + '.json')
                self.current_label = Annotation(file_path, label_path)
                # 载入数据
                self.current_label.load_annotation()

                for object in self.current_label.objects:
                    try:
                        group = int(object.group)
                        # 新增 手动/自动 group选择
                        if self.group_select_mode == 'auto':
                            self.current_group = group + 1 if group >= self.current_group else self.current_group
                        elif self.group_select_mode == 'manual':
                            self.current_group = 1
                        self.update_group_display()
                    except Exception as e:
                        pass
                    polygon = Polygon()
                    self.scene.addItem(polygon)
                    polygon.load_object(object)
                    self.polygons.append(polygon)

            if self.current_label is not None:
                self.label_Title.setText('{}'.format(self.current_label.label_path))
            else:
                self.label_Title.setText('{}'.format(file_path))

            self.annos_dock_widget.update_listwidget()
            self.info_dock_widget.update_widget()
            self.files_dock_widget.set_select(index)
            self.current_index = index
            self.files_dock_widget.label_current.setText('{}'.format(self.current_index+1))
            self.load_finished = True

        except Exception as e:
            print(e)
        finally:
            if self.current_index > 0:
                self.pushButton_Prev.setEnabled(True)
            else:
                self.pushButton_Prev.setEnabled(False)

            if self.current_index < len(self.files_list) - 1:
                self.pushButton_Next.setEnabled(True)
            else:
                self.pushButton_Next.setEnabled(False)

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

    def change_bit_map_to_semantic(self):
        # to semantic
        for polygon in self.polygons:
            polygon.setEnabled(False)
            for vertex in polygon.vertexs:
                vertex.setVisible(False)
            polygon.change_color(QtGui.QColor(self.category_color_dict.get(polygon.category, '#000000')))
            polygon.color.setAlpha(255)
            polygon.setBrush(polygon.color)
        self.annos_dock_widget.listWidget_annos.setEnabled(False)
        self.annos_dock_widget.checkBox_visible.setEnabled(False)
        self.pushButton_Segment_anything.setEnabled(False)
        self.pushButton_Polygon.setEnabled(False)
        self.pushButton_Visible.setEnabled(False)
        self.map_mode = MAPMode.SEMANTIC
        semantic_icon = QtGui.QIcon()
        semantic_icon.addPixmap(QtGui.QPixmap(":/icon/icons/semantic.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Bit_map.setIcon(semantic_icon)

    def change_bit_map_to_instance(self):
        # to instance
        for polygon in self.polygons:
            polygon.setEnabled(False)
            for vertex in polygon.vertexs:
                vertex.setVisible(False)
            if polygon.group != '':
                index = int(polygon.group)
                index = index % self.instance_cmap.shape[0]
                rgb = self.instance_cmap[index]
            else:
                rgb = self.instance_cmap[0]
            polygon.change_color(QtGui.QColor(rgb[0], rgb[1], rgb[2], 255))
            polygon.color.setAlpha(255)
            polygon.setBrush(polygon.color)
        self.annos_dock_widget.listWidget_annos.setEnabled(False)
        self.annos_dock_widget.checkBox_visible.setEnabled(False)
        self.pushButton_Segment_anything.setEnabled(False)
        self.pushButton_Polygon.setEnabled(False)
        self.pushButton_Visible.setEnabled(False)
        self.map_mode = MAPMode.INSTANCE
        instance_icon = QtGui.QIcon()
        instance_icon.addPixmap(QtGui.QPixmap(":/icon/icons/instance.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Bit_map.setIcon(instance_icon)

    def change_bit_map_to_label(self):
        # to label
        for polygon in self.polygons:
            polygon.setEnabled(True)
            for vertex in polygon.vertexs:
                # vertex.setEnabled(True)
                vertex.setVisible(polygon.isVisible())
            polygon.change_color(QtGui.QColor(self.category_color_dict.get(polygon.category, '#000000')))
            polygon.color.setAlpha(polygon.nohover_alpha)
            polygon.setBrush(polygon.color)
        self.annos_dock_widget.listWidget_annos.setEnabled(True)
        self.annos_dock_widget.checkBox_visible.setEnabled(True)
        self.SeganyEnabled()
        self.pushButton_Polygon.setEnabled(True)
        self.pushButton_Visible.setEnabled(True)
        self.map_mode = MAPMode.LABEL
        label_icon = QtGui.QIcon()
        label_icon.addPixmap(QtGui.QPixmap(":/icon/icons/照片_pic.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Bit_map.setIcon(label_icon)

    def change_bit_map(self):
        self.set_labels_visible(True)
        if self.scene.mode == STATUSMode.CREATE:
            self.scene.cancel_draw()
        if self.map_mode == MAPMode.LABEL:
            self.change_bit_map_to_semantic()

        elif self.map_mode == MAPMode.SEMANTIC:
            self.change_bit_map_to_instance()

        elif self.map_mode == MAPMode.INSTANCE:
            self.change_bit_map_to_label()
        else:
            pass

    def set_labels_visible(self, visible=None):
        if visible is None:
            visible = not self.annos_dock_widget.checkBox_visible.isChecked()
        self.annos_dock_widget.set_all_polygon_visible(visible)

    def model_manage(self):
        if not self.saved:
            result = QtWidgets.QMessageBox.question(self, 'Warning', 'Proceed without saved?', QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                return
        self.model_manager_dialog.show()

    def change_bfloat16_state(self, use: bool):
        self.cfg['software']['use_bfloat16'] = use
        self.init_segment_anything()
        self.model_manager_dialog.update_gui()
        self.save_software_cfg()

    def change_contour_mode(self, contour_mode:str=None):
        if contour_mode is None:
            if self.cfg['software']['contour_mode'] == 'max_only':
                contour_mode = 'external'
            elif self.cfg['software']['contour_mode'] == 'external':
                contour_mode = 'all'
            elif self.cfg['software']['contour_mode'] == 'all':
                contour_mode = 'max_only'

        if contour_mode == 'max_only':
            self.scene.change_contour_mode_to_save_max_only()

            label_icon = QtGui.QIcon()
            icon = ':/icons/icons/contour_only_max_32.svg'
            label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.pushButton_Contour_mode.setIcon(label_icon)

        elif contour_mode == 'external':
            self.scene.change_contour_mode_to_save_external()
            label_icon = QtGui.QIcon()
            icon = ':/icons/icons/contour_external_32.svg'
            label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.pushButton_Contour_mode.setIcon(label_icon)

        elif contour_mode == 'all':
            self.scene.change_contour_mode_to_save_all()
            label_icon = QtGui.QIcon()
            icon = ':/icons/icons/contour_all_32.svg'
            label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.pushButton_Contour_mode.setIcon(label_icon)
        else:
            pass
        self.cfg['software']['contour_mode'] = contour_mode
        self.save_software_cfg()

    def change_mask_aplha(self):
        value = self.slider_mask_aplha.value() / 10
        self.scene.mask_alpha = value
        self.scene.update_mask()
        self.cfg['software']['mask_alpha'] = value
        self.save_software_cfg()

    def change_vertex_size(self):
        value = self.slider_vertex_size.value()
        self.cfg['software']['vertex_size'] = value
        self.save_software_cfg()
        if self.current_index is not None:
            self.show_image(self.current_index, zoomfit=False)

    def change_edge_state(self, state:bool=None):
        if state is None:
            state = self.cfg.get('software', {}).get('show_edge', True)
            state = not state
        self.cfg['software']['show_edge'] = state
        self.save_software_cfg()
        #
        label_icon = QtGui.QIcon()
        icon = ':/icons/icons/edge_solid_32.svg' if state else ':/icons/icons/edge_dash_32.svg'
        label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_show_edge.setIcon(label_icon)

        if self.current_index is not None:
            self.show_image(self.current_index, zoomfit=False)

    def change_approx_polygon_state(self, state:bool=None):  # 是否使用多边形拟合，来减少多边形顶点
        if state is None:
            state = self.cfg.get('software', {}).get('use_polydp', True)
            state = not state
        self.cfg['software']['use_polydp'] = state
        self.save_software_cfg()
        #
        label_icon = QtGui.QIcon()
        icon = ':/icons/icons/circular_with_four_point_32.svg' if state else ':/icons/icons/circular_with_eight_point_32.svg'
        label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_use_polydp.setIcon(label_icon)

        if self.current_index is not None:
            self.show_image(self.current_index, zoomfit=False)

    def change_saturation(self, value):  # 调整图像饱和度
        if self.scene.image_data is not None:
            saturation_scale = value / 100.0
            hsv_image = cv2.cvtColor(self.scene.image_data, cv2.COLOR_RGB2HSV)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255)
            image_hsv = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
            height, width, channels = self.scene.image_data.shape
            pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(image_hsv.data, width, height, channels * width, QtGui.QImage.Format_RGB888))
            self.scene.image_item.setPixmap(pixmap)
        else:
            print('Image data not loaded in AnnotationScene')

    def change_prompt_visiable(self, state:bool=None):
        if state is None:
            state = self.cfg.get('software', {}).get('show_prompt', True)
            state = not state
        self.cfg['software']['show_prompt'] = state
        self.save_software_cfg()
        #
        label_icon = QtGui.QIcon()
        icon = ':/icons/icons/circular_solid_32.svg' if state else ':/icons/icons/circular_dash_32.svg'
        label_icon.addPixmap(QtGui.QPixmap(icon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_show_prompt.setIcon(label_icon)

        for item in self.scene.items():
            if isinstance(item, PromptPoint):
                item.setVisible(state)
        # if self.current_index is not None:
        #     self.show_image(self.current_index)

    def converter(self):
        self.Converter_dialog.show()

    def auto_segment(self):
        if self.use_segment_anything:
            self.auto_segment_dialog.show()
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Choice a sam model before auto segment.')

    def help(self):
        self.shortcut_dialog.show()

    def about(self):
        self.about_dialog.show()

    def save_cfg(self, config_file):
        # 只保存类别配置
        cfg = {'label': self.cfg.get('label', [])}
        save_config(cfg, config_file)

    def save_software_cfg(self):
        save_config(self.cfg, self.software_config_file)

    def exit(self):
        # 保存类别配置
        self.save_cfg(self.config_file)
        # 保存软件配置
        self.save_software_cfg()
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        self.exit()

    def minimized(self):
        self.showMinimized()

    def maximized(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def init_connect(self):
        self.pushButton_Open_dir.pressed.connect(self.open_dir)
        self.pushButton_Save_dir.pressed.connect(self.save_dir)
        self.pushButton_Prev.pressed.connect(self.prev_image)
        self.pushButton_Next.pressed.connect(self.next_image)
        self.pushButton_Setting.pressed.connect(self.setting)
        self.pushButton_Exit.pressed.connect(self.exit)

        self.pushButton_Segment_anything.pressed.connect(self.scene.start_segment_anything)
        self.pushButton_Polygon.pressed.connect(self.scene.start_draw_polygon)
        self.pushButton_Cancel.pressed.connect(self.scene.cancel_draw)
        self.pushButton_Backspace.pressed.connect(self.scene.backspace)
        self.pushButton_Finish.pressed.connect(self.scene.finish_draw)
        self.pushButton_Edit.pressed.connect(self.scene.edit_polygon)
        self.pushButton_Delete.pressed.connect(self.scene.delete_selected_graph)
        self.pushButton_Save.pressed.connect(self.save)
        self.pushButton_To_top.pressed.connect(self.scene.move_polygon_to_top)
        self.pushButton_To_bottom.pressed.connect(self.scene.move_polygon_to_bottom)

        self.actionEdit.triggered.connect(self.scene.edit_polygon)
        self.actionTo_top.triggered.connect(self.scene.move_polygon_to_top)
        self.actionTo_bottom.triggered.connect(self.scene.move_polygon_to_bottom)
        self.actionDelete.triggered.connect(self.scene.delete_selected_graph)

        self.pushButton_Zoom_in.pressed.connect(self.view.zoom_in)
        self.pushButton_Zoom_out.pressed.connect(self.view.zoom_out)
        self.pushButton_Fit_wiondow.pressed.connect(self.view.zoomfit)
        self.pushButton_Bit_map.pressed.connect(self.change_bit_map)
        self.pushButton_Visible.pressed.connect(functools.partial(self.set_labels_visible, None))

        self.pushButton_Model_manage.pressed.connect(self.model_manage)
        self.pushButton_Model_manage.setStatusTip('')

        self.pushButton_Converter.pressed.connect(self.converter)
        self.pushButton_Auto_segment.pressed.connect(self.auto_segment)

        self.pushButton_Shortcut.pressed.connect(self.help)
        self.pushButton_About.pressed.connect(self.about)

        self.pushButton_Translate.pressed.connect(self.translate)

        self.pushButton_Minimized.pressed.connect(self.minimized)
        self.pushButton_Maximized.pressed.connect(self.maximized)
        self.pushButton_Exit.pressed.connect(self.exit)
        self.annos_dock_widget.listWidget_annos.doubleClicked.connect(self.scene.edit_polygon)

        self.slider_mask_aplha.valueChanged.connect(self.change_mask_aplha)
        self.slider_vertex_size.valueChanged.connect(self.change_vertex_size)
        self.slider_image_saturation.valueChanged.connect(self.change_saturation)

        self.pushButton_show_edge.pressed.connect(self.change_edge_state)
        self.pushButton_show_prompt.pressed.connect(self.change_prompt_visiable)
        self.pushButton_use_polydp.pressed.connect(self.change_approx_polygon_state)
        self.pushButton_Contour_mode.pressed.connect(self.change_contour_mode)

    def reset_action(self):
        self.pushButton_Prev.setEnabled(False)
        self.pushButton_Next.setEnabled(False)
        self.pushButton_Segment_anything.setEnabled(False)
        self.pushButton_Polygon.setEnabled(False)
        self.pushButton_Edit.setEnabled(False)
        self.pushButton_Delete.setEnabled(False)
        self.pushButton_Save.setEnabled(False)
        self.pushButton_To_top.setEnabled(False)
        self.pushButton_To_bottom.setEnabled(False)
        self.pushButton_Bit_map.setChecked(False)
        self.pushButton_Bit_map.setEnabled(False)

        self.actionEdit.setEnabled(False)
        self.actionTo_top.setEnabled(False)
        self.actionTo_bottom.setEnabled(False)
        self.actionDelete.setEnabled(False)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if QtCore.Qt.LeftButton and self.m_flag and QMouseEvent.pos().y() <= 40:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseDoubleClickEvent(self, QMouseEvent:QtGui.QMouseEvent):
        if QMouseEvent.pos().y() <= 40:
            self.maximized()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

