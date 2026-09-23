param(
  [string]$Video='D:\Temp\AI Engineer test.mp4',
  [string]$Model='outputs\openvino_quyt\model.xml',
  [string]$OutputDir='outputs\cpu_final'
)
Set-Location (Resolve-Path "$PSScriptRoot\..")
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'
$dir=$OutputDir
Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$args=@(
  'scripts\run_subtitle_pipeline.py',
  $Video,
  '--output',"$dir\out.mp4",
  '--encode-preset','ultrafast',
  '--detector','ppocrv5_mobile',
  '--device','cpu',
  '--cpu-threads','0',
  '--batch-size','96',
  '--decode-prefetch-batches','4',
  '--roi-bottom-fraction','.45',
  '--ppocr-cpu-engine','openvino',
  '--ppocr-openvino-model',$Model,
  '--ppocr-openvino-streams','0',
  '--ppocr-openvino-fuse-preprocess',
  '--ppocr-adaptive-gating',
  '--ppocr-gate-max-skip-frames','1',
  '--ppocr-gate-change-threshold','.015',
  '--ppocr-gate-bright-net-threshold','.0075',
  '--ppocr-thresh','.30',
  '--ppocr-box-thresh','.50',
  '--high-score','.84',
  '--low-score','.50',
  '--export-coordinates',"$dir\out.json"
)
& '.\.venv_cpu\Scripts\python.exe' @args *> "$dir\run.log"
$code=$LASTEXITCODE
Get-Content "$dir\run.log" -Tail 110
exit $code
