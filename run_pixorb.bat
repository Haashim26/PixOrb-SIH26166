@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo PixOrb virtual environment not found.
  echo Create it with: py -3.11 -m venv .venv
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python app.py
