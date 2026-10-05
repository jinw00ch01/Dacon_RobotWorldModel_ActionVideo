param([switch]$RemoveProjectFolder)
# Removes what scripts/bootstrap_pro360.ps1 installed on the Pro 360 (the Pro is no longer used):
#   - stops and unregisters the scheduled task WM-Exchange-pro360 (the hidden PowerShell supervisor)
#   - shuts Syncthing down
#   - moves C:\Dacon\WM_Runtime and C:\Dacon\WM_Exchange to the Recycle Bin (recoverable)
#   - with -RemoveProjectFolder, also moves C:\Dacon\RobotWorldModel_ActionVideo to the Recycle Bin
# Paste the one-line command from the project thread into Windows PowerShell on the Pro 360.
$ErrorActionPreference = 'Continue'
if ($env:COMPUTERNAME -eq 'DESKTOP-C78VQOR') { throw 'This is the Ultra (DESKTOP-C78VQOR). Run this only on the Pro 360.' }
$runtime = 'C:\Dacon\WM_Runtime'
$state = Join-Path $runtime 'pro360'
if (Test-Path -LiteralPath $state) { 'Requested by user: uninstall' | Set-Content -LiteralPath (Join-Path $state 'STOP') -Encoding ascii }
foreach ($task in @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object { $_.TaskName -like 'WM-*' })) {
    Stop-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false
    Write-Host "Removed scheduled task $($task.TaskName)"
}
Start-Sleep -Seconds 5
$leftover = @(Get-CimInstance Win32_Process -Filter "Name='syncthing.exe' OR Name='powershell.exe'" |
    Where-Object { $_.CommandLine -and ($_.CommandLine -match 'WM_Runtime' -or $_.CommandLine -match 'start_exchange\.ps1') -and $_.ProcessId -ne $PID })
foreach ($process in $leftover) { Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue }
if ($leftover.Count) { Write-Host "Stopped $($leftover.Count) leftover Syncthing/supervisor process(es)"; Start-Sleep -Seconds 2 }

Add-Type -AssemblyName Microsoft.VisualBasic
$targets = @($runtime, 'C:\Dacon\WM_Exchange')
if ($RemoveProjectFolder) { $targets += 'C:\Dacon\RobotWorldModel_ActionVideo' }
foreach ($path in $targets) {
    if (Test-Path -LiteralPath $path) {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($path, 'OnlyErrorDialogs', 'SendToRecycleBin')
        Write-Host "Moved to the Recycle Bin: $path"
    }
}
Write-Host 'Done: nothing from this project runs on the Pro 360 any more.' -ForegroundColor Green
