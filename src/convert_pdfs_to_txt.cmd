@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
if [%1]==[] goto usage
if [%2]==[] goto usage
if not [%3]==[] goto usage

for %%i in (%1\*.pdf) do (
  set filename=%%~ni
  echo pdftotext -y 80 -H 650 -W 1000 -nopgbrk -eol unix %%i %2\!filename!.txt
  pdftotext -y 80 -H 650 -W 1000 -nopgbrk -eol unix %%i %2\!filename!.txt
)
goto end

:usage
echo Error: Missing or extraneous arguments
echo Usage: %0 source_dir target_dir

:end
