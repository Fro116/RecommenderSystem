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
python3 -m venv venv
deactivate
source venv/bin/activate
pip install torch==2.13.0 pandas==3.0.3 scipy==1.18.0 h5py==3.16.0 hdf5plugin==7.0.0 msgpack==1.2.1 torchao==0.17.0 tqdm==4.68.4 flash-attn-4==4.0.0b22
cd RecommenderSystem/notebooks/Training/
if timeout --kill-after=30 180 torchrun --standalone --nproc_per_node={{NUM_GPUS}} hardware_check.py; then
    python transformer.py --datadir ../../data/training --download 0 1 --prod
    torchrun --standalone --nproc_per_node={{NUM_GPUS}} transformer.py --datadir ../../data/training --prod
fi
pip install vastai
vastai stop instance $CONTAINER_ID
