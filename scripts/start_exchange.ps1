param([Parameter(Mandatory=$true)][string]$Config)
# Supervisor run by the WM-Exchange-<role> scheduled task: keeps Syncthing alive and runs one
# `python -m wm_ops tick` per poll until a STOP file appears. Never starts Claude.
$ErrorActionPreference = 'Continue'
$cfg = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$mutex = New-Object Threading.Mutex($false, ('Local\WMExchange-' + $cfg.role))
try { $acquired = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $acquired = $true }
if (-not $acquired) { $mutex.Dispose(); exit 0 }

function Write-Log([string]$Message) {
    ((Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') + ' ' + $Message) | Add-Content -LiteralPath (Join-Path $cfg.state_root 'supervisor.log') -Encoding utf8
}

function Find-Syncthing {
    # Adopt a Syncthing still running for this home instead of starting a duplicate (AFDA lesson).
    try {
        $all = @(Get-CimInstance Win32_Process -Filter "Name='syncthing.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($cfg.syncthing_home) })
        $ids = @($all | ForEach-Object { $_.ProcessId })
        $monitor = $all | Where-Object { $ids -notcontains $_.ParentProcessId } | Select-Object -First 1
        if ($null -eq $monitor) { return $null }
        return Get-Process -Id $monitor.ProcessId -ErrorAction SilentlyContinue
    } catch { return $null }
}

function Test-Exited($Process) {
    if ($null -eq $Process) { return $true }
    try { return $Process.HasExited } catch { return $false }
}

function Start-Syncthing {
    $found = Find-Syncthing
    if ($null -ne $found) { Write-Log ('adopted syncthing pid ' + $found.Id); return $found }
    $serveArgs = @('serve', '--home', ('"' + $cfg.syncthing_home + '"'), '--no-browser', '--no-upgrade',
                   '--log-file', ('"' + (Join-Path $cfg.state_root 'syncthing.log') + '"'))
    Write-Log 'starting syncthing'
    return Start-Process -FilePath $cfg.syncthing_exe -ArgumentList $serveArgs -WindowStyle Hidden -PassThru
}

try {
    Set-Location -LiteralPath $cfg.project_root
    New-Item -ItemType Directory -Path $cfg.state_root -Force | Out-Null
    $stopFile = Join-Path $cfg.state_root 'STOP'
    if (Test-Path -LiteralPath $stopFile) { return }
    $PID | Set-Content -LiteralPath (Join-Path $cfg.state_root 'supervisor.pid') -Encoding ascii
    $tickLog = Join-Path $cfg.state_root 'tick.log'
    $syncthing = Start-Syncthing
    while (-not (Test-Path -LiteralPath $stopFile)) {
        if (Test-Exited $syncthing) { $syncthing = Start-Syncthing }
        $line = (& $cfg.python -m wm_ops --config $Config tick 2>&1 | ForEach-Object { "$_" }) -join ' '
        if ($LASTEXITCODE -ne 0 -or $line -match '"(imported|launched_jobs|lost_jobs)": \[\s*"' -or $line -match '"errors": \[\s*"') {
            ((Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') + ' ' + $line) | Add-Content -LiteralPath $tickLog -Encoding utf8
        }
        Start-Sleep -Seconds $cfg.poll_seconds
    }
    Write-Log 'STOP file found; exiting'
} finally {
    if (-not (Test-Exited $syncthing)) {
        & $cfg.python -m wm_ops --config $Config syncthing shutdown *> $null
    }
    $mutex.ReleaseMutex(); $mutex.Dispose()
}
