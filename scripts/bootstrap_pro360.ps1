param(
    [Parameter(Mandatory=$true)][string]$JoinToken,
    [string]$Target = 'C:\Dacon\RobotWorldModel_ActionVideo',
    [string]$Repo = 'https://github.com/jinw00ch01/Dacon_RobotWorldModel_ActionVideo.git',
    [switch]$SkipTrainVideos,
    [switch]$FullData
)
# One-paste setup of the Pro 360 node. Paste the command from the project thread into a normal
# Windows PowerShell window on the Pro 360. No administrator rights and no GitHub login needed:
#   1. git (portable MinGit 2.56.0 if git is missing, SHA-256 checked)
#   2. Python 3.12 (per-user python.org 3.12.10 if missing, SHA-256 checked, PATH untouched)
#   3. clone or update the public repo into C:\Dacon\RobotWorldModel_ActionVideo
#   4. scripts\setup_env.ps1 -Role pro360   (CPU torch and tools in .venv-pro360)
#   5. scripts\setup_node.ps1 -Role pro360  (Syncthing, task WM-Exchange-pro360, join code)
#   6. wait until the Ultra accepts this laptop; data and packet folders then start syncing
# Re-running is safe: every step skips what is already done.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$tools = 'C:\Dacon\WM_Runtime\tools'
New-Item -ItemType Directory -Path $tools -Force | Out-Null
function Step([string]$Message) { Write-Host ''; Write-Host "== $Message" -ForegroundColor Cyan }
function Fail([string]$Message) { Write-Host "FAILED: $Message" -ForegroundColor Red; throw $Message }
function Get-Verified([string]$Url, [string]$Path, [string]$Sha) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile ($Path + '.partial')
        Move-Item -LiteralPath ($Path + '.partial') -Destination $Path -Force
    }
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha) {
        Remove-Item -LiteralPath $Path -Force
        Fail "SHA-256 mismatch for $Url"
    }
}

if ($JoinToken -cnotmatch '^[A-HJ-NP-Z2-9]{8}$') { Fail 'JoinToken must be the 8-character join code from the project thread.' }
if ($env:COMPUTERNAME -eq 'DESKTOP-C78VQOR') { Fail 'This is the Ultra (DESKTOP-C78VQOR). Run this on the Pro 360.' }

Step 'git'
$git = (Get-Command git -ErrorAction SilentlyContinue).Source
if (-not $git) {
    $mingit = Join-Path $tools 'mingit'
    $git = Join-Path $mingit 'cmd\git.exe'
    if (-not (Test-Path -LiteralPath $git)) {
        $zip = Join-Path $tools 'MinGit-2.56.0-64-bit.zip'
        Get-Verified 'https://github.com/git-for-windows/git/releases/download/v2.56.0.windows.1/MinGit-2.56.0-64-bit.zip' $zip '064b440ff870ed5198527e8f3a92cdf5bd2fd0fedf5e718af95e3fdaddeff718'
        Expand-Archive -LiteralPath $zip -DestinationPath $mingit -Force
    }
}
if (-not (Test-Path -LiteralPath $git)) { Fail 'git is not available' }
Write-Host "git: $git"

Step 'Python 3.12'
$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $found = & py -3.12 -c 'import sys; print(sys.executable)' 2>$null
    if ($LASTEXITCODE -eq 0 -and $found) { $python = "$found".Trim() }
}
if (-not $python) {
    foreach ($candidate in @((Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'), 'C:\Program Files\Python312\python.exe', (Join-Path $tools 'python312\python.exe'))) {
        if (Test-Path -LiteralPath $candidate) { $python = $candidate; break }
    }
}
if (-not $python) {
    $installer = Join-Path $tools 'python-3.12.10-amd64.exe'
    Get-Verified 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' $installer '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb'
    $pythonDir = Join-Path $tools 'python312'
    Start-Process -FilePath $installer -Wait -ArgumentList @('/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0',
        'Include_test=0', 'Shortcuts=0', 'AssociateFiles=0', "TargetDir=$pythonDir")
    $python = Join-Path $pythonDir 'python.exe'
}
if (-not (Test-Path -LiteralPath $python)) { Fail 'Python 3.12 is not available' }
Write-Host "python: $python"

Step "repository -> $Target"
if (-not (Test-Path -LiteralPath (Join-Path $Target '.git'))) {
    New-Item -ItemType Directory -Path $Target -Force | Out-Null
    & $git -C $Target init -q 2>$null
    & $git -C $Target remote add origin $Repo 2>$null
}
& $git -C $Target -c credential.interactive=never fetch -q origin main
if ($LASTEXITCODE -ne 0) { Fail 'git fetch failed (check the internet connection)' }
$dirty = & $git -C $Target status --porcelain
if ($dirty) {
    Write-Host 'Local changes found; keeping them and skipping the update.' -ForegroundColor Yellow
} else {
    & $git -C $Target checkout -q -B main origin/main
    if ($LASTEXITCODE -ne 0) { Fail "git checkout failed; move files that conflict with the repository out of $Target and re-run" }
}
& $git -C $Target branch -q --set-upstream-to=origin/main main 2>$null
Write-Host ('commit: ' + (& $git -C $Target rev-parse --short HEAD))

Set-Location -LiteralPath $Target
Step 'Python environment .venv-pro360 (CPU torch; first run downloads about 1 GB)'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Target 'scripts\setup_env.ps1') -Role pro360 -PythonExecutable $python
if ($LASTEXITCODE -ne 0) { Fail 'scripts\setup_env.ps1 failed (see the messages above)' }

$freeGb = [math]::Floor((Get-PSDrive -Name $Target.Substring(0, 1)).Free / 1GB)
$skip = $SkipTrainVideos -or ((-not $FullData) -and $freeGb -lt 25)
Step ("exchange service (free disk {0} GB; training videos {1})" -f $freeGb, $(if ($skip) { 'skipped' } else { 'synced' }))
$nodeArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Target 'scripts\setup_node.ps1'), '-Role', 'pro360',
              '-JoinToken', $JoinToken, '-GitExe', $git)
if ($skip) { $nodeArgs += '-SkipTrainVideos' }
& powershell.exe @nodeArgs
if ($LASTEXITCODE -ne 0) { Fail 'scripts\setup_node.ps1 failed (see the messages above)' }

Step 'waiting for the Ultra to accept this laptop (up to 10 minutes; the Ultra must be on)'
$venvPython = Join-Path $Target '.venv-pro360\Scripts\python.exe'
$paired = $false
for ($i = 0; $i -lt 60 -and -not $paired; $i++) {
    $status = (& $venvPython -m wm_ops status 2>$null) -join "`n"
    if ($status -match '"peer_connected":\s*true') { $paired = $true; break }
    Start-Sleep -Seconds 10
}
if ($paired) {
    Write-Host 'Done: the Pro 360 is paired with the Ultra. Data and packet folders are syncing in the background.' -ForegroundColor Green
} else {
    Write-Host 'Setup finished, but the Ultra has not connected yet. Leave both laptops on; pairing completes on its own.' -ForegroundColor Yellow
    Write-Host 'If Windows asks whether to allow Syncthing through the firewall, allow it on private networks.'
}
& $venvPython -m wm_ops register-device
