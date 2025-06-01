@echo off
REM Create virtual environment
python -m venv venv

REM Activate virtual environment
REM If running in PowerShell and activation fails due to policy, set execution policy temporarily
if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" (
	call venv\Scripts\activate.bat 2>nul || (
		powershell -Command "Set-ExecutionPolicy RemoteSigned -Scope Process -Force"
		call venv\Scripts\activate.bat
	)
) else (
	call venv\Scripts\activate.bat
)

REM Install requirements
pip install -r requirements.txt