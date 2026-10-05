param(
    [Parameter(Mandatory=$true)][ValidateSet('ultra5060','pro360')][string]$Role,
    [string]$RuntimeRoot = 'C:\Dacon\WM_Runtime',
    [string]$ExchangeRoot = 'C:\Dacon\WM_Exchange',
    [switch]$SkipTrainVideos,
    [switch]$NoTask,
    [string]$JoinToken = '',
    [string]$GitExe = ''
)
# One-time, idempotent node setup (no administrator rights):
#   1. Syncthing v2.1.5 (pinned, SHA-256 checked) with its own home under C:\Dacon\WM_Runtime\<role>
#   2. configs/local-node.json (git-ignored); the peer device ID comes from configs/nodes.json
#   3. Task Scheduler task WM-Exchange-<role>: keeps Syncthing alive and runs `python -m wm_ops tick`
#      (packet import, detached jobs, git fast-forward, peer pairing). It never starts Claude.
#   4. -JoinToken: announce this PC with the peer's one-time join code so the peer accepts it without git push
# State lives outside %LOCALAPPDATA% on purpose: shells inside the Claude/Codex Store apps virtualize it.
$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot ".venv-$Role\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    & (Join-Path $PSScriptRoot 'setup_env.ps1') -Role $Role
    if (-not (Test-Path -LiteralPath $python)) { throw 'Role venv setup failed; see the output above.' }
}
$stateRoot = Join-Path $RuntimeRoot $Role
$syncthingHome = Join-Path $stateRoot 'syncthing'
$binRoot = Join-Path $stateRoot 'bin'
New-Item -ItemType Directory -Path $stateRoot, $syncthingHome, $binRoot, $ExchangeRoot -Force | Out-Null

# 1. Syncthing binary (same pinned release the AFDA project verified on these laptops)
$version = 'v2.1.5'
$zipName = "syncthing-windows-amd64-$version.zip"
$zipPath = Join-Path $binRoot $zipName
$expectedSha = '39571e4d0900c2a2cab14c0b170f49751340a869e49734ccc8079d9b98a7974b'
if (-not (Test-Path -LiteralPath $zipPath)) {
    $partial = $zipPath + '.partial'
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/syncthing/syncthing/releases/download/$version/$zipName" -OutFile $partial
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSha) { Remove-Item -LiteralPath $partial; throw 'Syncthing archive SHA-256 mismatch.' }
    Move-Item -LiteralPath $partial -Destination $zipPath
}
if ((Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSha) { throw 'Existing Syncthing archive changed.' }
$syncthingExe = Join-Path $binRoot "syncthing-windows-amd64-$version\syncthing.exe"
if (-not (Test-Path -LiteralPath $syncthingExe)) { Expand-Archive -LiteralPath $zipPath -DestinationPath $binRoot }

$guiPort = if ($Role -eq 'ultra5060') { 8395 } else { 8396 }
$syncthingConfig = Join-Path $syncthingHome 'config.xml'
if (-not (Test-Path -LiteralPath $syncthingConfig)) {
    & $syncthingExe generate --home $syncthingHome --no-port-probing | Out-Null
    if (-not (Test-Path -LiteralPath $syncthingConfig)) { throw 'Syncthing identity generation failed.' }
    [xml]$xml = Get-Content -LiteralPath $syncthingConfig -Raw
    $xml.configuration.gui.address = "127.0.0.1:$guiPort"
    $xml.configuration.options.startBrowser = 'false'
    foreach ($folder in @($xml.configuration.folder)) { if ($null -ne $folder) { [void]$xml.configuration.RemoveChild($folder) } }
    $xml.Save($syncthingConfig)
}

# 2. Local node config (absolute paths, git-ignored). Keys written by earlier runs (peer ID, join codes) are kept.
$configPath = Join-Path $projectRoot 'configs\local-node.json'
$peerRole = if ($Role -eq 'ultra5060') { 'pro360' } else { 'ultra5060' }
$peerId = $null
$nodesPath = Join-Path $projectRoot 'configs\nodes.json'
if (Test-Path -LiteralPath $nodesPath) { $peerId = (Get-Content -LiteralPath $nodesPath -Raw | ConvertFrom-Json).devices.$peerRole }
$nodeConfig = [ordered]@{}
if (Test-Path -LiteralPath $configPath) {
    (Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $nodeConfig[$_.Name] = $_.Value }
}
$values = [ordered]@{
    version = 1; role = $Role; project_root = $projectRoot; python = $python
    exchange_root = $ExchangeRoot; state_root = $stateRoot
    syncthing_home = $syncthingHome; syncthing_exe = $syncthingExe; listen_port = 22010
    data_root = (Join-Path $projectRoot 'open'); share_data = $true; skip_train_videos = [bool]$SkipTrainVideos
    auto_git = $true; poll_seconds = 15
}
foreach ($key in $values.Keys) { $nodeConfig[$key] = $values[$key] }
if (-not $nodeConfig['peer_device_id'] -and $peerId) { $nodeConfig['peer_device_id'] = $peerId }
if ($JoinToken) { $nodeConfig['join_token'] = $JoinToken }
if ($GitExe) { $nodeConfig['git_exe'] = $GitExe }
$nodeConfig | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $configPath -Encoding UTF8
New-Item -ItemType Directory -Path (Join-Path $projectRoot 'open') -Force | Out-Null
# Role memory for every Claude session opened in this folder on this PC (git-ignored)
"# Role of this PC: $Role`r`n@.claude/roles/$Role.md" | Set-Content -LiteralPath (Join-Path $projectRoot 'CLAUDE.local.md') -Encoding UTF8
Remove-Item -LiteralPath (Join-Path $stateRoot 'STOP') -ErrorAction SilentlyContinue

# 3. Exchange supervisor as a per-user scheduled task (starts at logon, restarts when it dies)
$taskName = "WM-Exchange-$Role"
$launch = Join-Path $PSScriptRoot 'start_exchange.ps1'
if (-not $NoTask) {
    $user = "$env:USERDOMAIN\$env:USERNAME"
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $projectRoot `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launch`" -Config `"$configPath`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    Start-ScheduledTask -TaskName $taskName
    Write-Output "Scheduled task $taskName registered and started."
}

# 4. Configure devices and folders once Syncthing answers
$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    $out = & $python -m wm_ops --config $configPath syncthing configure 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ready) { Write-Output $out; throw "Syncthing did not answer on 127.0.0.1:$guiPort; see $stateRoot\syncthing.log" }
Write-Output $out
& $python -m wm_ops --config $configPath register-device
Write-Output "Setup of $Role complete. Peer from configs/nodes.json: $(if ($peerId) { $peerId } else { 'none yet' })"
