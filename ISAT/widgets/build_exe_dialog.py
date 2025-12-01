# -*- coding: utf-8 -*-
# @Author  : LG

import os
import sys
import subprocess
from PyQt5 import QtCore, QtWidgets, QtGui


class BuildExeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("打包为EXE / Build EXE")
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.resize(700, 500)

        self.setup_ui()
        self.check_environment()

    def setup_ui(self):
        """设置界面"""
        layout = QtWidgets.QVBoxLayout(self)

        # 标题
        title_label = QtWidgets.QLabel("将ISAT打包为独立的EXE可执行文件")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)

        # 环境检查区域
        env_group = QtWidgets.QGroupBox("环境检查 / Environment Check")
        env_layout = QtWidgets.QVBoxLayout()

        self.python_label = QtWidgets.QLabel("Python: 检查中...")
        self.pyinstaller_label = QtWidgets.QLabel("PyInstaller: 检查中...")
        self.spec_label = QtWidgets.QLabel("build_exe.spec: 检查中...")

        env_layout.addWidget(self.python_label)
        env_layout.addWidget(self.pyinstaller_label)
        env_layout.addWidget(self.spec_label)

        env_group.setLayout(env_layout)
        layout.addWidget(env_group)

        # 选项区域
        options_group = QtWidgets.QGroupBox("打包选项 / Build Options")
        options_layout = QtWidgets.QVBoxLayout()

        self.clean_checkbox = QtWidgets.QCheckBox("清理之前的构建 / Clean previous builds")
        self.clean_checkbox.setChecked(True)
        options_layout.addWidget(self.clean_checkbox)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # 输出区域
        output_group = QtWidgets.QGroupBox("打包日志 / Build Log")
        output_layout = QtWidgets.QVBoxLayout()

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas, Monaco, monospace; font-size: 10px;")
        output_layout.addWidget(self.log_text)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定模式
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # 按钮区域
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self.start_button = QtWidgets.QPushButton("开始打包 / Start Build")
        self.start_button.setMinimumWidth(150)
        self.start_button.clicked.connect(self.start_build)
        button_layout.addWidget(self.start_button)

        self.close_button = QtWidgets.QPushButton("关闭 / Close")
        self.close_button.setMinimumWidth(100)
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def check_environment(self):
        """检查打包环境"""
        self.log_text.append("=== 环境检查 / Environment Check ===\n")

        # 检查Python
        try:
            python_version = sys.version.split()[0]
            self.python_label.setText(f"✓ Python: {python_version}")
            self.python_label.setStyleSheet("color: green;")
            self.log_text.append(f"✓ Python {python_version}")
        except Exception as e:
            self.python_label.setText(f"✗ Python: 未找到")
            self.python_label.setStyleSheet("color: red;")
            self.log_text.append(f"✗ Python 未找到: {e}")

        # 检查PyInstaller
        pyinstaller_found = False
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "pyinstaller"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # 提取版本号
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':')[1].strip()
                        self.pyinstaller_label.setText(f"✓ PyInstaller: {version}")
                        self.pyinstaller_label.setStyleSheet("color: green;")
                        self.log_text.append(f"✓ PyInstaller {version}")
                        pyinstaller_found = True
                        break

                # 如果没有找到版本信息，说明格式不对
                if not pyinstaller_found:
                    self.pyinstaller_label.setText("✗ PyInstaller: 未安装")
                    self.pyinstaller_label.setStyleSheet("color: orange;")
                    self.log_text.append("⚠ PyInstaller 未安装，将自动安装")
            else:
                self.pyinstaller_label.setText("✗ PyInstaller: 未安装")
                self.pyinstaller_label.setStyleSheet("color: orange;")
                self.log_text.append("⚠ PyInstaller 未安装，将自动安装")
        except Exception as e:
            self.pyinstaller_label.setText(f"✗ PyInstaller: 检查失败")
            self.pyinstaller_label.setStyleSheet("color: orange;")
            self.log_text.append(f"⚠ PyInstaller 检查失败: {e}")

        # 检查spec文件
        project_root = self.get_project_root()
        spec_file = os.path.join(project_root, "build_exe.spec")
        if os.path.exists(spec_file):
            self.spec_label.setText(f"✓ build_exe.spec: 已找到")
            self.spec_label.setStyleSheet("color: green;")
            self.log_text.append(f"✓ build_exe.spec 已找到")
        else:
            self.spec_label.setText(f"✗ build_exe.spec: 未找到")
            self.spec_label.setStyleSheet("color: red;")
            self.log_text.append(f"✗ build_exe.spec 未找到")
            self.start_button.setEnabled(False)

        self.log_text.append("\n准备就绪 / Ready to build\n")

    def get_project_root(self):
        """获取项目根目录"""
        # ISAT模块所在目录的父目录
        isat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.dirname(isat_dir)

    def start_build(self):
        """开始打包"""
        self.start_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.progress_bar.show()
        self.log_text.append("\n=== 开始打包 / Start Building ===")
        self.log_text.append("⏱️  预计需要时间：15-20分钟 / Estimated time: 15-20 minutes")
        self.log_text.append("📝 请耐心等待，可以最小化窗口继续其他工作 / Please be patient\n")

        # 在后台线程中执行打包
        self.build_thread = BuildThread(
            project_root=self.get_project_root(),
            clean_build=self.clean_checkbox.isChecked()
        )
        self.build_thread.log_signal.connect(self.append_log)
        self.build_thread.finished_signal.connect(self.build_finished)
        self.build_thread.start()

    def append_log(self, text):
        """追加日志"""
        self.log_text.append(text)
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def build_finished(self, success, message):
        """打包完成"""
        self.progress_bar.hide()
        self.start_button.setEnabled(True)
        self.close_button.setEnabled(True)

        if success:
            self.log_text.append(f"\n✓ 打包成功！/ Build succeeded!")
            self.log_text.append(f"输出位置 / Output: {message}")
            QtWidgets.QMessageBox.information(
                self,
                "打包成功 / Success",
                f"打包已完成！\n\n输出位置:\n{message}"
            )
        else:
            self.log_text.append(f"\n✗ 打包失败 / Build failed: {message}")
            QtWidgets.QMessageBox.warning(
                self,
                "打包失败 / Failed",
                f"打包失败:\n{message}"
            )


class BuildThread(QtCore.QThread):
    """打包线程"""
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(bool, str)

    def __init__(self, project_root, clean_build):
        super().__init__()
        self.project_root = project_root
        self.clean_build = clean_build

    def run(self):
        """执行打包"""
        try:
            # 切换到项目目录
            os.chdir(self.project_root)

            # 1. 检查并安装依赖
            self.log_signal.emit("[1/4] 检查依赖 / Checking dependencies...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "pyinstaller"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.log_signal.emit("安装 PyInstaller / Installing PyInstaller...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "pyinstaller"],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    self.finished_signal.emit(False, "PyInstaller 安装失败")
                    return

            # 2. 清理旧文件
            if self.clean_build:
                self.log_signal.emit("\n[2/4] 清理旧文件 / Cleaning old builds...")
                import shutil
                for dir_name in ['build', 'dist']:
                    dir_path = os.path.join(self.project_root, dir_name)
                    if os.path.exists(dir_path):
                        try:
                            # 使用ignore_errors=True避免权限问题卡住
                            shutil.rmtree(dir_path, ignore_errors=True)
                            self.log_signal.emit(f"✓ 已删除 {dir_name}/")
                        except Exception as e:
                            self.log_signal.emit(f"⚠ 无法删除 {dir_name}/: {e}")
                            # 即使删除失败也继续
                            continue
            else:
                self.log_signal.emit("\n[2/4] 跳过清理 / Skip cleaning")

            # 3. 运行PyInstaller
            self.log_signal.emit("\n[3/4] 执行打包 / Running PyInstaller...")
            self.log_signal.emit("⏱️  这是最耗时的步骤，需要约15-20分钟 / This is the longest step, takes 15-20 minutes")
            self.log_signal.emit("💡 提示：可以最小化此窗口，打包会在后台继续进行 / Tip: You can minimize this window\n")

            process = subprocess.Popen(
                [sys.executable, "-m", "PyInstaller", "build_exe.spec", "--clean"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.project_root
            )

            # 实时输出日志
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    # 只显示重要信息
                    if any(keyword in line for keyword in ['INFO:', 'WARNING:', 'ERROR:', 'Building', 'Analyzing']):
                        self.log_signal.emit(line)

            process.wait()

            if process.returncode != 0:
                self.finished_signal.emit(False, f"PyInstaller 执行失败 (exit code: {process.returncode})")
                return

            # 4. 验证输出
            self.log_signal.emit("\n[4/4] 验证输出 / Verifying output...")
            dist_dir = os.path.join(self.project_root, "dist", "ISAT_SAM")
            exe_file = os.path.join(dist_dir, "ISAT_SAM.exe")

            if os.path.exists(exe_file):
                file_size = os.path.getsize(exe_file) / (1024 * 1024)  # MB
                self.log_signal.emit(f"✓ EXE 文件已生成: {file_size:.1f} MB")
                self.finished_signal.emit(True, dist_dir)
            else:
                self.finished_signal.emit(False, "未找到生成的EXE文件")

        except Exception as e:
            self.finished_signal.emit(False, str(e))
