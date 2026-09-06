#!/usr/bin/env bash
# From the local machine: pick an EU datacenter with H100 SXM stock, create a
# 20 GB network volume there (reused if one named bslm exists) and a Secure
# Cloud H100 pod on it, wait for SSH, print the pod json.
#   bash pretrain/remote/create_pod.sh [pod-name]
set -uo pipefail
NAME="${1:-bslm-72m}"
R="${RUNPODCTL:-$HOME/.bslm/bin/runpodctl.exe}"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(python -c "import json,os;print(json.load(open(os.path.expanduser('~/.bslm/runpod.json')))['api_key'])")}"
echo "balance: $($R user 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin).get('clientBalance'))")"

DC=$($R datacenter list 2>/dev/null | python -c "
import json,sys
d=json.load(sys.stdin); rows=d if isinstance(d,list) else d.get('data',d)
stock={}
for dc in rows:
    if not isinstance(dc,dict): continue
    for g in dc.get('gpuAvailability') or []:
        if isinstance(g,dict) and g.get('displayName')=='H100 SXM':
            stock[dc['id']]=g.get('stockStatus') or 'ok'
for pref in ['EU-FR-1','EUR-NO-2','EUR-IS-3','EU-NL-1']:
    if stock.get(pref) in ('ok','High','Medium'): print(pref); break
else:
    for pref in ['EU-FR-1','EUR-NO-2','EUR-IS-3','EU-NL-1']:
        if pref in stock: print(pref); break
")
echo "datacenter: $DC"
[ -z "$DC" ] && { echo "no EU datacenter with H100 SXM"; exit 1; }

VOL=$($R network-volume list 2>/dev/null | python -c "
import json,sys
d=json.load(sys.stdin); rows=d if isinstance(d,list) else d.get('data',d)
for v in rows:
    if isinstance(v,dict) and v.get('name')=='bslm' and v.get('dataCenterId')=='$DC': print(v['id']); break
")
if [ -z "$VOL" ]; then
  VOL=$($R network-volume create --name bslm --size 20 --data-center-id "$DC" 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin).get('id',''))")
  echo "volume created: $VOL"
else
  echo "volume reused: $VOL"
fi
[ -z "$VOL" ] && { echo "volume creation failed"; exit 1; }

TPL=$($R template list --type official 2>/dev/null | python -c "
import json,sys
d=json.load(sys.stdin); rows=d if isinstance(d,list) else d.get('data',d)
best=''
for t in rows:
    if not isinstance(t,dict): continue
    n=(t.get('name') or '').lower(); i=t.get('id') or ''
    if 'pytorch' in n or 'torch' in i:
        best=i;
        if '2.8' in n or '2.9' in n or 'cu12' in n: break
print(best)
")
[ -z "$TPL" ] && TPL="runpod-torch-v21"
echo "template: $TPL"

$R pod create --template-id "$TPL" --gpu-id "NVIDIA H100 80GB HBM3" --cloud-type SECURE \
  --container-disk-in-gb 30 --network-volume-id "$VOL" --data-center-ids "$DC" \
  --name "$NAME" --ports "22/tcp" --wait --wait-timeout 15m 2>&1 | tee "$HOME/.bslm/last_pod.json"
