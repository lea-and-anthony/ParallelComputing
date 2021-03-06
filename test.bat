@echo off
echo ------------------------------- ORIGINAL VERSION ------------------------------
cd src_original
cmd /Q /C echo y^> "%temp%\answer.tmp" ^& (.\test.bat ^< "%temp%\answer.tmp") ^& del "%temp%\answer.tmp"
echo.
echo ------------------------------- MODIFIED VERSION ------------------------------
cd ..\src_modified
cmd /Q /C echo y^> "%temp%\answer.tmp" ^& (.\test.bat ^< "%temp%\answer.tmp") ^& del "%temp%\answer.tmp"
echo.
echo --------------------------------- CUDA VERSION --------------------------------
cd ..\src_cuda
cmd /Q /C echo y^> "%temp%\answer.tmp" ^& (.\test.bat ^< "%temp%\answer.tmp") ^& del "%temp%\answer.tmp"
echo.
echo ---------------------------- CUDA OPTIMIZED VERSION ---------------------------
cd ..\src_cuda_optimized
cmd /Q /C echo y^> "%temp%\answer.tmp" ^& (.\test.bat ^< "%temp%\answer.tmp") ^& del "%temp%\answer.tmp"
echo.
echo ------------------------ CUDA OPTIMIZED VERSION NO HOST -----------------------
cd ..\src_cuda_optimized_nohost
cmd /Q /C echo y^> "%temp%\answer.tmp" ^& (.\test.bat ^< "%temp%\answer.tmp") ^& del "%temp%\answer.tmp"
echo.
cd ..\