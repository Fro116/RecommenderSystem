#!/bin/bash
set -euxo pipefail
cd /data
apt update && apt install git curl unzip -y
# sfcompute: enable RDMA via InfiniBand
apt-get update && apt-get install -y wget sudo && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*
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
NODE=`hostname | tr '-' ' ' | awk '{print $(NF)}'`
python transformer.py --datadir ../../data/training --download $NODE $NUM_NODES
# wait for all nodes to finish downloading
MASTER_ADDR=pretrain-0.pretrain-svc
MASTER_PORT=29500
if [ "$NODE" = "0" ]; then
  echo "This is the master node ($(hostname)). Proceeding..."
else
  echo "This is a worker node ($(hostname)). Waiting for $MASTER_ADDR:$MASTER_PORT..."
  while ! (exec 3<>/dev/tcp/$MASTER_ADDR/$MASTER_PORT) 2>/dev/null; do
    echo "Master not available yet, sleeping..."
    sleep 1
  done
  echo "Master is up. Proceeding..."
fi
torchrun --nnodes $NUM_NODES --nproc_per_node=8 --rdzv-backend c10d --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT transformer.py --datadir ../../data/training --modeltype ranking
