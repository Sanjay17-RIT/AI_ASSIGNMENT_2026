@echo off
echo ======================================================================
echo Launching Next-Gen Glassmorphic TypeScript Frontend & API Gateway
echo Candidate: SAI SANJAY A (USN: 1MS25AD053)
echo Course: Artificial Intelligence and Computational Systems (25 A D 35)
echo ======================================================================

echo.
echo [1/2] Starting Python FastAPI Gateway on http://127.0.0.1:8000 ...
start "AI API Gateway (FastAPI)" cmd /k "python api_server.py"

timeout /t 2 /nobreak >nul

echo [2/2] Starting TypeScript Vite Frontend ...
cd frontend
start "Next-Gen Glassmorphic Frontend" cmd /k "npm run dev"

echo.
echo System started successfully!
echo - API Docs: http://127.0.0.1:8000/docs
echo - Web App:  http://localhost:5173
echo ======================================================================
