@echo off
chcp 65001 > nul
setlocal enableDelayedExpansion

REM Set script directory and stay in the root folder
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if we can run Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause & exit /b 1
)

REM No arguments - show menu
if "%~1"=="" goto :menu

REM Process arguments - using the launcher script

if /i "%~1"=="-demo" (python scripts/demo.py & pause & exit /b 0)
if /i "%~1"=="-stats" (
    python run_cli.py stats
    pause & exit /b 0
)
if /i "%~1"=="-history" (
    python run_cli.py history
    pause & exit /b 0
)
if /i "%~1"=="-help" (
    python run_cli.py --help
    pause & exit /b 0
)
if /i "%~1"=="-batch" (
    if "%~2"=="" (echo Usage: ai -batch file.json & pause & exit /b 1)
    python run_cli.py batch "%~2"
    pause & exit /b 0
)
if /i "%~1"=="-threshold" (
    if "%~2"=="" (echo Usage: ai -threshold 0.8 & pause & exit /b 1)
    python run_cli.py set-threshold --threshold %~2
    pause & exit /b 0
)
if /i "%~1"=="-feedback" (
    if "%~4"=="" (echo Usage: ai -feedback "question" "Yes/No" "reason" & pause & exit /b 1)
    python run_cli.py feedback "%~2" "%~3" "%~4"
    pause & exit /b 0
)

REM Default: single decision if no other command matches
echo Analyzing: %*
echo.
python run_cli.py decide "%*"
echo.
pause & exit /b 0

:menu
cls
echo.
@echo off
echo.

python print_logo.py


                                                         



echo.
echo  Quick Actions:
echo    1  Quick Decision    ^|  5  Show Demo
echo    2  Batch Process     ^|  6  Set Threshold
echo    3  View Stats        ^|  7  Debug ^& Feedback Mode
echo    4  View History
echo    0  Exit
echo.
set /p choice="Select option (0-7): "

if "!choice!"=="1" goto :quick
if "!choice!"=="2" goto :batch
if "!choice!"=="3" (
    python run_cli.py stats
    echo. & pause & goto :menu
)
if "!choice!"=="4" (
    python run_cli.py history --limit 10
    echo. & pause & goto :menu
)
if "!choice!"=="5" (python scripts/demo.py & pause & goto :menu)
if "!choice!"=="6" goto :threshold
if "!choice!"=="7" goto :debug
if "!choice!"=="0" exit /b 0

echo Invalid choice & timeout /t 1 >nul & goto :menu

:quick
echo.
set /p question="Enter your question: "
if "!question!"=="" (echo Question cannot be empty & pause & goto :menu)
echo.
echo Analyzing: !question!
echo.
python run_cli.py decide "!question!"
echo.
pause & goto :menu

:batch
echo.
set /p filepath="Enter file path (relative to project root): "
if "!filepath!"=="" (echo File path cannot be empty & pause & goto :menu)
echo Processing: !filepath!
python run_cli.py batch "!filepath!"
echo. & pause & goto :menu

:threshold
echo.
set /p threshold="Enter threshold (0.0-1.0): "
if "!threshold!"=="" (echo Threshold cannot be empty & pause & goto :menu)
python run_cli.py set-threshold --threshold !threshold!
echo. & pause & goto :menu

:debug
cls
echo.
echo  +===========================================================+
echo  ^|                DEBUG AND FEEDBACK MODE                  ^|
echo  +===========================================================+
echo.
echo  This mode helps you test, correct, and record AI responses
echo  to improve future model performance.
echo.
echo  Common test cases:
echo    - "Is the sky blue?" (should be Yes)
echo    - "2 + 2 = 5" (should be No)
echo.
set /p debug_question="Enter question to debug: "
if "!debug_question!"=="" (echo Question cannot be empty & pause & goto :menu)

echo.
echo [Debug Feedback]
echo ================
:get_answer
set /p expected="What should the correct answer be? (Yes/No): "
if /i not "!expected!"=="Yes" if /i not "!expected!"=="No" (
    echo Please enter 'Yes' or 'No'.
    goto :get_answer
)

set /p feedback="Why is this the correct answer?: "
if "!feedback!"=="" (
    echo Reason cannot be empty. Please provide a brief explanation.
    set /p feedback="Why is this the correct answer?: "
)
if "!feedback!"=="" (echo Reason cannot be empty, aborting. & pause & goto :menu)

REM Call the new feedback command
python run_cli.py feedback "!debug_question!" "!expected!" "!feedback!"

echo.
pause & goto :menu