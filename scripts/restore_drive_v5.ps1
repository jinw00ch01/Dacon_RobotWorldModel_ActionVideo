# MyDrive/dacon_wm 복원 + v5 잠재 올리기 (JINWOO 가 실행: Claude 는 대회 데이터·잠재를 Drive 로 옮기지 않는다)
# 실행: powershell -ExecutionPolicy Bypass -File C:\Dacon\RobotWorldModel_ActionVideo\scripts\restore_drive_v5.ps1
# 올라가는 것 (약 28GB): 변환한 Cosmos 가중치, Bridge 행동 임베더, 빈 텍스트 임베딩, v2_long 16k adapter, 행동 통계,
# 잠재 캐시 두 개. 이미 있는 파일은 robocopy 가 건너뛴다. 끝난 뒤 Google Drive 업로드가 다 끝나면 Colab v5 를 실행한다.
$dst = Join-Path ((Get-ChildItem G:\ -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'Colab Notebooks') } | Select-Object -First 1).FullName) 'dacon_wm'
$sh = 'C:\Dacon\WM_Shared'
Write-Host "대상: $dst"
robocopy "$sh\cosmos_ac\transformer" "$dst\cosmos_ac\transformer" /E /NP /NFL /NDL
robocopy "$sh\cosmos_ac" "$dst\cosmos_ac" action_embedder_bridge.pt empty_text_embeddings.pt /NP /NFL /NDL
robocopy "$sh\cosmos_ac\runs\v2_long" "$dst\cosmos_ac\runs\v2_long" adapter_016000.pt /NP /NFL /NDL
robocopy 'C:\Dacon\RobotWorldModel_ActionVideo\open\data\train' "$dst" so100_action_statistics.json /NP /NFL /NDL
robocopy "$sh\latents_240x320" "$dst\latents_240x320" /E /NP /NFL /NDL
robocopy "$sh\latents_240x320_extra" "$dst\latents_240x320_extra" /E /NP /NFL /NDL
Write-Host "복사 끝. Google Drive 앱에서 업로드가 끝났는지 확인한 뒤 Colab 에서 colab/train_cosmos_ac_v5.ipynb 를 실행하세요."
