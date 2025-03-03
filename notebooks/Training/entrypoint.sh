#!/bin/bash
set -euxo pipefail
handle_error() {
  sleep 600 # give time to debug
}
trap handle_error ERR
cleanup() {
    if [ -n "${RUNPOD_POD_ID+x}" ]; then
	    runpodctl remove pod $RUNPOD_POD_ID
    fi
}
trap cleanup EXIT
if [ -n "${RUNPOD_POD_ID+x}" ]; then
    (sleep 86400 && cleanup) & # max runtime of 24 hours
    python -c 'import torch; torch.rand(1, device="cuda:0")'
	cd ~
else
    cd /data
fi
apt update
apt install unzip git -y
git clone https://github.com/Fro116/RecommenderSystem.git
curl https://rclone.org/install.sh | bash
mkdir -p ~/.config/rclone/
echo "
[r2]
type = s3
provider = Cloudflare
access_key_id = $R2_ACCESS_KEY_ID
secret_access_key = $R2_SECRET_ACCESS_KEY
endpoint = https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com
" > ~/.config/rclone/rclone.conf
rclone --retries=10 -Pv copy r2:rsys/secrets secrets
mv secrets RecommenderSystem/
mkdir -p RecommenderSystem/data/training
pip install h5py hdf5plugin msgpack pandas scipy tqdm
cd RecommenderSystem/notebooks/Training/
python transformer.py --download
cmd="torchrun --standalone --nproc_per_node=8 transformer.py"
$cmd || (sleep 10 && $cmd) || (sleep 60 && $cmd)
python bagofwords.py --datadir ../../data/training --download
for m in 0 1; do
for metric in rating watch plantowatch drop; do
    cmd="torchrun --standalone --nproc_per_node=8 bagofwords.py --datadir ../../data/training --medium $m --metric $metric"
    # sleep to cleanup resources after torchrun exits
    $cmd || (sleep 10 && $cmd) || (sleep 60 && $cmd)
done
done
