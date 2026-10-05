param(
    [Parameter(Mandatory=$true)][ValidateSet('ultra5060','pro360')][string]$Role,
    [switch]$Kit,
    [string]$PythonExecutable = ''
)
# Creates the role venv (.venv-<role>) and, with -Kit, the submission-kit venv (.venv-kit).
# Idempotent: a re-run installs only what is missing. No administrator rights needed.
# Windows PowerShell 5.1 turns native stderr (pip warnings, expected import errors) into errors,
# so this script checks $LASTEXITCODE explicitly instead of using ErrorActionPreference=Stop.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

function Resolve-BasePython {
    if ($PythonExecutable) { return $PythonExecutable }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $exe = & py -3.12 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return "$exe".Trim() }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) { return $python.Source }
    throw 'Python 3.12 x64 not found. Install it from python.org first.'
}

function New-Venv([string]$Path) {
    $py = Join-Path $Path 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $py)) {
        & (Resolve-BasePython) -m venv $Path
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed: $Path" }
    }
    & $py -c 'import sys; assert sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32, sys.version'
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 x64 is required.' }
    & $py -m pip install --upgrade pip --timeout 120 --retries 5 --disable-pip-version-check -q
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    return $py
}

function Install-Torch([string]$Py, [string]$Torch, [string]$Vision, [string]$Index) {
    $have = & $Py -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)' 2>$null
    if ($LASTEXITCODE -eq 0 -and $have -and "$have".StartsWith($Torch)) { Write-Output "torch already installed: $have"; return }
    & $Py -m pip install "torch==$Torch" "torchvision==$Vision" --index-url $Index --timeout 120 --retries 5 --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw "PyTorch $Torch installation failed." }
}

$index = if ($Role -eq 'ultra5060') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }

Write-Output "== role venv .venv-$Role"
$py = New-Venv (Join-Path $root ".venv-$Role")
Install-Torch $py '2.8.0' '0.23.0' $index
& $py -m pip install -r (Join-Path $root 'requirements\common.txt') --timeout 120 --retries 5 --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Common dependency installation failed.' }
& $py -m pip check
& $py -m pip freeze | Set-Content -LiteralPath (Join-Path $root "requirements\lock-$Role.txt") -Encoding utf8

if ($Kit) {
    # The official submission kit pins torch 2.7.1 / torchvision 0.22.1; keep it isolated from model code (Rule 7).
    Write-Output '== submission kit venv .venv-kit'
    $kitRequirements = Join-Path $root 'open\submission_kit\requirements.txt'
    if (-not (Test-Path -LiteralPath $kitRequirements)) { throw "Missing $kitRequirements (unzip open.zip first)." }
    $kpy = New-Venv (Join-Path $root '.venv-kit')
    Install-Torch $kpy '2.7.1' '0.22.1' $index
    & $kpy -m pip install -r $kitRequirements --timeout 120 --retries 5 --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw 'Submission kit dependency installation failed.' }
    & $kpy -m pip freeze | Set-Content -LiteralPath (Join-Path $root 'requirements\lock-kit.txt') -Encoding utf8
}
Write-Output 'setup_env done'
