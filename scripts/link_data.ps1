param([string]$DataRoot = '')
# Makes ./open in this checkout (for example an agent's git worktree) point at the competition data of
# the main checkout through a directory junction: no copy, no administrator rights. Safe to re-run.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$link = Join-Path $root 'open'
if (Test-Path -LiteralPath (Join-Path $link 'data')) { Write-Output "open is available: $link"; exit 0 }
if (-not $DataRoot) {
    $common = & git -C $root rev-parse --path-format=absolute --git-common-dir 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $common) { throw 'Not inside a git checkout; pass -DataRoot <folder that contains data\>.' }
    $DataRoot = Join-Path (Split-Path -Parent ("$common".Trim())) 'open'
}
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot 'data'))) { throw "No competition data under $DataRoot (unzip open.zip there first)." }
if (Test-Path -LiteralPath $link) {
    if (@(Get-ChildItem -LiteralPath $link -Force).Count -ne 0) { throw "$link exists and is not the data folder; move it away first." }
    Remove-Item -LiteralPath $link
}
New-Item -ItemType Junction -Path $link -Target $DataRoot | Out-Null
Write-Output "linked $link -> $DataRoot"
