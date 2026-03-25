@echo off
echo Compiling FP-SAN NSS with MSVC...
cl /EHsc /MD /O2 /std:c++17 src\nss_core.cpp src\main.cpp /Fe:nss_engine.exe
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED. Ensure you are running this from the "x64 Native Tools Command Prompt for VS".
    exit /b %ERRORLEVEL%
)
echo.
echo ================================================
echo Build Successful! Run nss_engine.exe to test.
echo ================================================
