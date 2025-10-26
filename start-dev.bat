@echo off
echo Starting Crop Counting AI Development Environment...

echo.
echo Starting Backend (Flask API)...
start "Backend" cmd /k "cd /d backend && python app.py"

echo.
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak

echo.
echo Starting Frontend (React)...
start "Frontend" cmd /k "cd /d frontend && npm start"

echo.
echo Both services starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul