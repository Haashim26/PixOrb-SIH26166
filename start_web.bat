@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo PixOrb virtual environment not found.
  echo Run setup_windows.ps1 first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install -r requirements-web.txt
python -m uvicorn web.main:app --host 127.0.0.1 --port 7860
