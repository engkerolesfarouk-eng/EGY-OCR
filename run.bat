@echo off
rem Drag PDF/image files onto this file. Results go to an "out" folder next to it.
cd /d "%~dp0"
python -m arocr %* -o out
pause
