@echo off
chcp 65001
echo ===============================================
echo ISAT_with_segment_anything 打包为EXE工具
echo ===============================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo [步骤 1/5] 检查并安装项目依赖...
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
if %errorlevel% neq 0 (
    echo [警告] 依赖安装可能存在问题，但继续执行...
)

echo.
echo [步骤 2/5] 安装PyInstaller...
pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
if %errorlevel% neq 0 (
    echo [错误] PyInstaller安装失败
    pause
    exit /b 1
)

echo.
echo [步骤 3/5] 清理之前的构建文件...
if exist "build" (
    echo 删除 build 目录...
    rmdir /s /q build
)
if exist "dist" (
    echo 删除 dist 目录...
    rmdir /s /q dist
)

echo.
echo [步骤 4/5] 开始打包（这可能需要5-15分钟，请耐心等待）...
echo 使用spec文件打包...
pyinstaller build_exe.spec --clean
if %errorlevel% neq 0 (
    echo.
    echo [错误] 打包失败！请检查错误信息。
    echo.
    echo 常见问题解决方案:
    echo 1. 确保所有依赖都已正确安装
    echo 2. 尝试手动运行: python main.py 确认程序可以正常运行
    echo 3. 检查是否有杀毒软件干扰
    echo 4. 查看上方的错误信息，可能需要添加缺失的隐藏导入
    pause
    exit /b 1
)

echo.
echo [步骤 5/5] 验证打包结果...
if exist "dist\ISAT_SAM\ISAT_SAM.exe" (
    echo.
    echo ===============================================
    echo [成功] 打包完成！
    echo ===============================================
    echo.
    echo 可执行文件位置: dist\ISAT_SAM\ISAT_SAM.exe
    echo.
    echo 提示:
    echo 1. 整个 dist\ISAT_SAM 文件夹都是必需的，不要只复制exe文件
    echo 2. 首次运行可能较慢（需要解压临时文件）
    echo 3. 如果遇到问题，可以在命令行运行exe查看错误信息
    echo 4. dist\ISAT_SAM 文件夹可以打包成zip分发给其他用户
    echo.
    echo 是否立即运行测试? (Y/N)
    set /p choice=请选择:
    if /i "%choice%"=="Y" (
        echo 启动程序...
        start "" "dist\ISAT_SAM\ISAT_SAM.exe"
    )
) else (
    echo [错误] 未找到生成的exe文件
    pause
    exit /b 1
)

echo.
pause
