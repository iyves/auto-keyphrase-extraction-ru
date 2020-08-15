@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
if [%1]==[] goto usage
if [%2]==[] goto usage
if not [%3]==[] goto usage

for /d %%D in (%1\*) do (
  echo convert_pdfs_to_txt.cmd %1\%%~nxD %2\%%~nxD
  convert_pdfs_to_txt.cmd %1\%%~nxD %2\%%~nxD
)
goto end

:usage
echo Error: Missing or extraneous arguments
echo Usage: %0 source_dir target_dir

:end
PAUSE
