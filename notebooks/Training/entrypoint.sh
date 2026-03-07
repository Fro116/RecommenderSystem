#!/bin/bash
{{ENVVARS}}
source /venv/main/bin/activate
rm -rf RecommenderSystem
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
deactivate
python3 -m venv /venv/torch
source /venv/torch/bin/activate
pip install --upgrade pip
pip install torch==2.9.1 torchao==0.15.0 flash-attn-4==4.0.0b4 h5py==3.16.0 hdf5plugin==6.0.0 pandas==3.0.1 scipy==1.17.1 tqdm==4.67.3
cd RecommenderSystem/notebooks/Training/
python transformer.py --datadir ../../data/training --download 0 1 --prod
torchrun --standalone --nproc_per_node={{NUM_GPUS}} transformer.py --datadir ../../data/training --prod
deactivate
python3 -m venv venv/vastai
source venv/vastai/bin/activate
pip install vastai
vastai stop instance $CONTAINER_ID
deactivate
