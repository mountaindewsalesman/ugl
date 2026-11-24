# Start a supervised training run as its own Windows process, detached from whatever
# terminal or editor launched it (so closing or trimming that session cannot kill it).
#   powershell -File tools/train_detached.ps1 -Log data/build/train_base.log -Args "tools/train_parser_tf.py --model google/electra-base-discriminator ..."
param([Parameter(Mandatory)][string]$Log, [Parameter(Mandatory)][string]$Args)
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$p = Start-Process -FilePath $py -WorkingDirectory $root -WindowStyle Hidden -PassThru `
     -ArgumentList "tools/supervise.py --log $Log -- $Args"
$p.PriorityClass = "BelowNormal"   # the desktop stays responsive; the GPU does the work anyway
"started supervisor pid $($p.Id); log $Log"
