@echo off
echo Compiling FP-SAN NSS with MSVC...
cl.exe /EHsc /O2 /I.\include src\nss_core.cpp src\main.cpp /Fe:nss_engine.exe

if %errorlevel% neq 0 (
    echo.
    echo BUILD FAILED. Ensure you are running this from the "x64 Native Tools Command Prompt for VS".
    exit /b %errorlevel%
)

echo.
echo Build Successful! Running nss_engine.exe...
echo.
.\nss_engine.exe
