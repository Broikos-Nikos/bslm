# Overnight pretraining with automatic resume. Stops when the trainer prints "final step".
$env:PYTHONIOENCODING = "utf-8"
Set-Location "C:\xampp\htdocs\playground\bslm"
$name = "72m-fineweb"
$log = "pretrain\runs\$name\console.log"
New-Item -ItemType Directory -Force "pretrain\runs\$name" | Out-Null
for ($i = 1; $i -le 30; $i++) {
  "=== launch $i  $(Get-Date -Format s) ===" | Out-File -Append -Encoding utf8 $log
  & ".venv\Scripts\python.exe" -m pretrain.train --name $name --size 72m --tokens 2.4e9 --micro 8 --resume --eval_every 100 --ckpt_every 150 2>&1 | Out-File -Append -Encoding utf8 $log
  if (Select-String -Path "pretrain\runs\$name\log.txt" -Pattern "^final step" -Quiet) { "=== done ===" | Out-File -Append -Encoding utf8 $log; break }
  Start-Sleep -Seconds 30
}
