@echo off
rem BSLM showcase: double click to run the from-scratch v1 model in a window.
rem Uses the project venv; llama-server starts on the CPU, first load takes a few seconds.
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m bslm.showcase
) else (
  start "" pythonw -m bslm.showcase
)
