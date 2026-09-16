@echo off
REM Configure and build with the MSVC toolset from VS 18 BuildTools.
REM CMake has no generator for this VS version, so drive Ninja directly.
REM
REM MSVC 14.50 is newer than any nvcc here accepts, so pin 14.44 (= VS 2022
REM 17.14). nvcc only ever compiles kernel code in this project, which is what
REM keeps that constraint cheap. CUDA is optional: without nvcc the CPU
REM reference and its whole validation suite still build and run.
REM
REM NOTE: 13.1 is the only toolkit on this machine with a compiler, and its
REM runtime needs driver r580+. The installed driver is 576.52 (CUDA 12.9), so
REM the kernels COMPILE here but cannot RUN until the driver is updated. See
REM `rodsim gpu-parity` for the runtime diagnostic.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul || exit /b 1
set PATH=C:\Program Files\CMake\bin;%PATH%
set "NVCC=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe"
if exist "%NVCC%" (
  cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="%NVCC%" || exit /b 1
) else (
  cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
)
cmake --build "%~dp0build" %* || exit /b 1
