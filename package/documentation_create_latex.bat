@echo off
REM adapt next line to include Scripts folder of your Python distribution
SET PATH=%PATH%;C:\Program Files\Python39\Scripts
cd "..\docs"
rmdir /s /q "_build"
call make.bat html
call make.bat latex
pause
