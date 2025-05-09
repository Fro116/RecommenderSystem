#!/bin/bash
set -euxo pipefail
cd /data
apt update && apt install git curl unzip -y
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
pip install scipy h5py hdf5plugin msgpack torchao torchtune
cd RecommenderSystem/notebooks/Training/
python transformer.py --datadir ../../data/training --download
cmd="torchrun --standalone --nproc_per_node=8 transformer.py --datadir ../../data/training"
$cmd || (sleep 10 && $cmd) || (sleep 60 && $cmd)
