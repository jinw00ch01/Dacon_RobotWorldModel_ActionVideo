param([Parameter(Mandatory=$true)][ValidatePattern('^[a-z0-9-]+$')][string]$Name)
# Gives one agent thread its own git clone under work\agents\<name> (git-ignored by the main folder).
# The clone shares the main folder's venvs, node config and competition data (open junction), so two
# agents on this PC never edit the same working tree. Safe to re-run.
$ErrorActionPreference = 'Continue'
$main = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $main "work\agents\$Name"
$origin = "$(& git -C $main remote get-url origin)".Trim()
if (-not $origin) { throw "No origin remote in $main" }
if (-not (Test-Path -LiteralPath (Join-Path $dest '.git'))) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
    & git clone -q $origin $dest
    if ($LASTEXITCODE -ne 0) { throw "git clone failed: $origin" }
} else {
    & git -C $dest fetch -q origin
}
$link = Join-Path $dest 'open'
if (-not (Test-Path -LiteralPath (Join-Path $link 'data'))) {
    if (Test-Path -LiteralPath $link) { throw "$link exists and is not the data folder" }
    New-Item -ItemType Junction -Path $link -Target (Join-Path $main 'open') | Out-Null
}
$role = if ($Name -eq 'verifier') { 'verifier' } else { 'ultra5060' }
"# Agent checkout: $Name`r`n@.claude/roles/$role.md" | Set-Content -LiteralPath (Join-Path $dest 'CLAUDE.local.md') -Encoding UTF8
Write-Output "checkout: $dest"
Write-Output "branch:   $(& git -C $dest branch --show-current)"
Write-Output "python:   $(Join-Path $main '.venv-ultra5060\Scripts\python.exe')"
