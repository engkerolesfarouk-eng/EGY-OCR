@echo off
rem Builds dist\EGY-OCR\EGY-OCR.exe - copy that whole folder to any Windows PC, no Python needed.
rem Uses the "py" launcher so pip and PyInstaller run on the same Python.
cd /d "%~dp0"
py -m pip install --user -r requirements.txt pyinstaller || goto :fail
py -c "import PySide6, cv2, numpy, onnxruntime, pypdfium2, reportlab, pyclipper; print('libraries OK')" || goto :fail
py -m PyInstaller --noconfirm --clean --windowed --name EGY-OCR --icon "arocr\assets\icon.ico" ^
  --add-data "models;models" ^
  --add-data "arocr\data;arocr\data" ^
  --add-data "arocr\assets;arocr\assets" ^
  --collect-all onnxruntime --collect-all pypdfium2 ^
  app.py || goto :fail
echo.
echo Done. Run: dist\EGY-OCR\EGY-OCR.exe   (NOT the one in the build folder)
pause
exit /b 0
:fail
echo.
echo Build failed - see the messages above.
pause
exit /b 1
