@echo off
set datetime=%date% %time%
echo [VINCREATIONZ] Starting Git Sync at %datetime%
echo ------------------------------------------

:: Step 1: Stage all changes (new files, edits, and deletions)
git add .

:: Step 2: Commit with a timestamped message
git commit -m "Auto-sync: %datetime%"

:: Step 3: Push to the cloud
:: We use 'main' here based on your previous terminal output
git push origin main

echo ------------------------------------------
echo [SUCCESS] Your local alpha is now live on GitHub.
pause