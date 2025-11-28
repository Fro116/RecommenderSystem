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
# cuda 12.8 packages
if [ ! -f "venv" ]; then
    python3 -m venv venv
fi
deactivate
source venv/bin/activate
pip install torch==2.8.0 pandas==2.3.2 scipy==1.16.2 h5py==3.14.0 hdf5plugin==5.1.0 msgpack==1.1.1 torchao==0.13.0 torchtune==0.6.1
cd RecommenderSystem/notebooks/Training/
python transformer.py --datadir ../../data/training --download 0 1 --prod
torchrun --standalone --nproc_per_node=8 transformer.py --datadir ../../data/training --modeltype masked --prod
sleep 60
torchrun --standalone --nproc_per_node=8 transformer.py --datadir ../../data/training --modeltype causal --prod

pip install vastai
vastai stop instance $CONTAINER_ID
