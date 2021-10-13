@ECHO OFF

REM create package for Gpufit, assumes everything is compiled

if "%1" == "" (
	echo specify build base path
	goto end
)

if "%2" == "" (
	echo specify version
	goto end
)

if "%3" == "" (
	echo specify source base path
	goto end
)

REM date and time from https://stackoverflow.com/a/30343827/1536976

@SETLOCAL ENABLEDELAYEDEXPANSION

@REM Use WMIC to retrieve date and time
@echo off
FOR /F "skip=1 tokens=1-6" %%A IN ('WMIC Path Win32_LocalTime Get Day^,Hour^,Minute^,Month^,Second^,Year /Format:table') DO (
    IF NOT "%%~F"=="" (
        SET /A SortDate = 10000 * %%F + 100 * %%D + %%A
        set YEAR=!SortDate:~0,4!
        set MON=!SortDate:~4,2!
        set DAY=!SortDate:~6,2!
        @REM Add 1000000 so as to force a prepended 0 if hours less than 10
        SET /A SortTime = 1000000 + 10000 * %%B + 100 * %%C + %%E
        set HOUR=!SortTime:~1,2!
        set MIN=!SortTime:~3,2!
        set SEC=!SortTime:~5,2!
    )
)

set DATECODE=!YEAR!!MON!!DAY!!HOUR!!MIN!
echo %DATECODE%

REM define paths

set BUILD_BASE=%1
set VERSION=%2
set SOURCE_BASE=%3

set OUTPUT_NAME=Gpufit_%VERSION%_cublas_build%DATECODE%
set ROOT_INSTALL=%BUILD_BASE%\%OUTPUT_NAME%
set OUTPUT_ZIP=%BUILD_BASE%\%OUTPUT_NAME%.zip

set PERFORMANCE_TEST_INSTALL=%ROOT_INSTALL%\gpufit_performance_test
set PYTHON_INSTALL=%ROOT_INSTALL%\python
set x64_MATLAB_INSTALL=%ROOT_INSTALL%\matlab
set x64_JAVA_INSTALL=%ROOT_INSTALL%\java
set SDK_INSTALL_ROOT=%ROOT_INSTALL%\sdk

set x64_BUILD=%BUILD_BASE%\VC16x64-11.4_cublas\RelWithDebInfo
set x64_BUILD_LIB=%BUILD_BASE%\VC16x64-11.4_cublas\Gpufit\RelWithDebInfo
set x64_PYTHON_BUILD=%x64_BUILD%\pyGpufit\dist
set x64_MATLAB_BUILD=%x64_BUILD%\matlab
set x64_JAVA_BUILD=%x64_BUILD%\java

set EXAMPLES_SOURCE=%SOURCE_BASE%\examples
set PYTHON_SOURCE=%SOURCE_BASE%\Gpufit\python
set MATLAB_SOURCE=%SOURCE_BASE%\Gpufit\matlab
set SDK_README_SOURCE=%SOURCE_BASE%\package\sdk_readme.txt

set MANUAL_SOURCE=%SOURCE_BASE%\docs\_build\latex\Gpufit.pdf
set MANUAL_INSTALL=%ROOT_INSTALL%\Gpufit_%VERSION%_Manual.pdf

for /r %x64_BUILD% %%a in (*cublas*) do set CUBLAS_DLL=%%~dpnxa
if not defined CUBLAS_DLL (
echo cuBLAS DLL not existent
)

REM clean up (if necessary)

if exist "%ROOT_INSTALL%" rmdir /s /q "%ROOT_INSTALL%"
if exist "%OUTPUT_ZIP%" del "%OUTPUT_ZIP%"

REM create root folder

echo create root directory
mkdir "%ROOT_INSTALL%"

REM copy main readme (is markdown, written as txt) and license

copy "%SOURCE_BASE%\README.md" "%ROOT_INSTALL%\README.txt"
copy "%SOURCE_BASE%\LICENSE.txt" "%ROOT_INSTALL%"

REM copy manual

if not exist "%MANUAL_SOURCE%" (
	echo file %MANUAL_SOURCE% required, does not exist
	goto end
)
copy "%MANUAL_SOURCE%" "%MANUAL_INSTALL%"

REM copy performance test

echo collect performance test application
mkdir "%PERFORMANCE_TEST_INSTALL%"
copy "%EXAMPLES_SOURCE%\Gpufit_Cpufit_Performance_Comparison_readme.txt" "%PERFORMANCE_TEST_INSTALL%\README.txt"

mkdir "%PERFORMANCE_TEST_INSTALL%\win64"
copy "%x64_BUILD%\Gpufit_Cpufit_Performance_Comparison.exe" "%PERFORMANCE_TEST_INSTALL%\win64"
copy "%x64_BUILD%\Gpufit.dll" "%PERFORMANCE_TEST_INSTALL%\win64"
copy "%x64_BUILD%\Cpufit.dll" "%PERFORMANCE_TEST_INSTALL%\win64"
if defined CUBLAS_DLL (
copy "%CUBLAS_DLL%" "%PERFORMANCE_TEST_INSTALL%\win64"
)

REM copy Python packages

echo collect python
mkdir "%PYTHON_INSTALL%"
copy "%x64_PYTHON_BUILD%\pyGpufit-%VERSION%-py2.py3-none-any.whl" "%PYTHON_INSTALL%\pyGpufit-%VERSION%-py2.py3-none-win_amd64.whl"
copy "%PYTHON_SOURCE%\README.txt" "%PYTHON_INSTALL%"
xcopy "%PYTHON_SOURCE%\examples" "%PYTHON_INSTALL%\examples" /i /q

REM copy Matlab 64 bit

echo collect matlab64
mkdir "%x64_MATLAB_INSTALL%"
xcopy "%x64_MATLAB_BUILD%" "%x64_MATLAB_INSTALL%" /q
xcopy "%MATLAB_SOURCE%\examples" "%x64_MATLAB_INSTALL%\examples" /i /q

REM copy Java 64 bit

echo collect java64
mkdir "%x64_JAVA_INSTALL%"
xcopy "%x64_JAVA_BUILD%" "%x64_JAVA_INSTALL%" /q /s

REM copy SDK_INSTALL_ROOT

echo collect SDK
mkdir "%SDK_INSTALL_ROOT%"
copy "%SDK_README_SOURCE%" "%SDK_INSTALL_ROOT%\README.txt"

mkdir "%SDK_INSTALL_ROOT%\include"
copy "%SOURCE_BASE%\Gpufit\gpufit.h" "%SDK_INSTALL_ROOT%\include"

mkdir "%SDK_INSTALL_ROOT%\win64"
copy "%x64_BUILD%\Gpufit.dll" "%SDK_INSTALL_ROOT%\win64"
copy "%x64_BUILD_LIB%\Gpufit.lib" "%SDK_INSTALL_ROOT%\win64"
if defined CUBLAS_DLL (
copy "%CUBLAS_DLL%" "%SDK_INSTALL_ROOT%\win64"
)

REM zip content of temp folder with 7-Zip if availabe

set ZIP=C:\Program Files\7-Zip\7z.exe

if not exist "%ZIP%" (
	echo 7-Zip not installed, zip manually
	goto end
) ELSE (
	echo zip result
	"%ZIP%" a -y -r -mem=AES256 "%OUTPUT_ZIP%" "%ROOT_INSTALL%%" > nul
)

:end
PAUSE