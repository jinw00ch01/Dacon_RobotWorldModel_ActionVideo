param([switch]$Unregister)
# Stops the WM exchange supervisor (and Syncthing) on this PC. Run setup_node.ps1 again to resume.
$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content -LiteralPath (Join-Path $projectRoot 'configs\local-node.json') -Raw | ConvertFrom-Json
'Requested by user' | Set-Content -LiteralPath (Join-Path $cfg.state_root 'STOP') -Encoding ascii
if ($Unregister) { Unregister-ScheduledTask -TaskName ('WM-Exchange-' + $cfg.role) -Confirm:$false -ErrorAction SilentlyContinue }
Write-Output 'Stop requested; the supervisor exits within one poll and shuts Syncthing down.'
