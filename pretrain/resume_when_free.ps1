# Waits until the other local GPU workload (queue on port 8188) has nothing
# running or pending for two consecutive minutes, then relaunches the given
# training loop script, which resumes from the last checkpoint.
#   powershell -File resume_when_free.ps1 <run-name> <loop-script>
#   e.g.  resume_when_free.ps1 72m-fineweb C:\xampp\htdocs\playground\bslm\pretrain\day72m.ps1
param(
  [string]$Name = "56m-fineweb",
  [string]$Loop = "C:\xampp\htdocs\playground\bslm\pretrain\overnight.ps1"
)
$log = "C:\xampp\htdocs\playground\bslm\pretrain\runs\$Name\console.log"
"=== resume_when_free waiting $(Get-Date -Format s) ===" | Out-File -Append -Encoding utf8 $log
$idle = 0
while ($true) {
  Start-Sleep -Seconds 60
  try {
    $q = Invoke-RestMethod -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:8188/queue"
    $busy = (@($q.queue_running).Count + @($q.queue_pending).Count)
  } catch { $busy = 0 }      # nothing listening counts as free
  if ($busy -eq 0) { $idle++ } else { $idle = 0 }
  if ($idle -ge 2) { break }
}
"=== card free, relaunching $Loop $(Get-Date -Format s) ===" | Out-File -Append -Encoding utf8 $log
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File $Loop" -WindowStyle Hidden
