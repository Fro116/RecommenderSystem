#!/bin/bash
{{ENVVARS}}
git clone https://github.com/Fro116/RecommenderSystem.git
sudo apt update
sudo apt install -y python3.12-venv build-essential python3-dev
curl https://rclone.org/install.sh | sudo bash
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
python3 -m venv venv
source venv/bin/activate
pip install torch==2.13.0 pandas==3.0.3 scipy==1.18.0 h5py==3.16.0 hdf5plugin==7.0.0 msgpack==1.2.1 torchao==0.17.0 tqdm==4.68.4 flash-attn-4==4.0.0b22
cd RecommenderSystem/notebooks/Training/
python transformer.py --datadir ../../data/training --download 0 1
torchrun --standalone --nproc_per_node=8 transformer.py --datadir ../../data/training
