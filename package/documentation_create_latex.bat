@echo off
REM adapt next line to include Scripts folder of your Python distribution
SET PATH=%PATH%;C:\Users\thero\Documents\Visual Studio 2019\Python Scripts
cd "..\docs"
rmdir /s /q "_build"
call make.bat html
call make.bat latex
pause
