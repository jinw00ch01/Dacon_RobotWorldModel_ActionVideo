param(
    [string]$RuntimeRoot = 'C:\Dacon\WM_Runtime',
    [switch]$NoTask
)
# One-time, idempotent setup of the job runner on the Ultra (no administrator rights):
#   1. configs/local-node.json (git-ignored, absolute paths)
#   2. Task Scheduler task WM-Jobs-ultra5060: `pythonw -m wm_ops serve` at logon, restarted if it dies.
#      pythonw has no console, and every child gets CREATE_NO_WINDOW, so no window ever opens.
#      The runner launches queued jobs (one GPU job at a time) and fast-forwards a clean main checkout.
#      It never starts Claude.
#   3. Removes the old WM-Exchange-* (Syncthing) task if it is still registered.
# State lives outside %LOCALAPPDATA% on purpose: shells inside the Claude/Codex Store apps virtualize it.
$ErrorActionPreference = 'Continue'
$role = 'ultra5060'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot ".venv-$role\Scripts\python.exe"
$pythonw = Join-Path $projectRoot ".venv-$role\Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $python)) {
    & (Join-Path $PSScriptRoot 'setup_env.ps1') -Role $role
    if (-not (Test-Path -LiteralPath $python)) { throw 'Role venv setup failed; see the output above.' }
}
$stateRoot = Join-Path $RuntimeRoot $role
New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null

# 1. Local node config. Only keys the runner uses are kept (older Syncthing keys are dropped).
$configPath = Join-Path $projectRoot 'configs\local-node.json'
$gitExe = $null
if (Test-Path -LiteralPath $configPath) { $gitExe = (Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).git_exe }
$nodeConfig = [ordered]@{
    version = 1; role = $role; project_root = $projectRoot; python = $python; state_root = $stateRoot
    auto_git = $true; poll_seconds = 15
}
if ($gitExe) { $nodeConfig['git_exe'] = $gitExe }
$nodeConfig | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $configPath -Encoding UTF8
"# Role of this PC: $role`r`n@.claude/roles/$role.md" | Set-Content -LiteralPath (Join-Path $projectRoot 'CLAUDE.local.md') -Encoding UTF8
Remove-Item -LiteralPath (Join-Path $stateRoot 'STOP') -ErrorAction SilentlyContinue

# 2./3. Windowless runner task; drop the old Syncthing exchange task
foreach ($old in @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object { $_.TaskName -like 'WM-Exchange-*' })) {
    Stop-ScheduledTask -TaskName $old.TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $old.TaskName -Confirm:$false
    Write-Output "Removed old task $($old.TaskName)."
}
$taskName = "WM-Jobs-$role"
if (-not $NoTask) {
    $user = "$env:USERDOMAIN\$env:USERNAME"
    $action = New-ScheduledTaskAction -Execute $pythonw -WorkingDirectory $projectRoot -Argument "-m wm_ops --config `"$configPath`" serve"
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    Start-ScheduledTask -TaskName $taskName
    Write-Output "Scheduled task $taskName registered and started (windowless)."
}
Start-Sleep -Seconds 20
& $python -m wm_ops --config $configPath status
