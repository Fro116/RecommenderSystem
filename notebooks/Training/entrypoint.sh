#!/bin/bash
set -euxo pipefail
cd ~
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
rclone --retries=10 -Pv copy r2:rsys/environment/training environment
mv environment RecommenderSystem/
mkdir -p RecommenderSystem/data/training
pip install filelock h5py hdf5plugin msgpack pandas scipy tqdm
cd RecommenderSystem/notebooks/Training/
python bagofwords.py --datadir ../../data/training --medium 0 --metric rating --device cuda:0 &> ../../data/training.0.watch.log &
python bagofwords.py --datadir ../../data/training --medium 0 --metric watch --device cuda:1 &> ../../data/training.0.watch.log &
python bagofwords.py --datadir ../../data/training --medium 0 --metric plantowatch --device cuda:2 &> ../../data/training.0.plantowatch.log &
python bagofwords.py --datadir ../../data/training --medium 0 --metric drop --device cuda:3 &> ../../data/training.0.drop.log &
python bagofwords.py --datadir ../../data/training --medium 1 --metric rating --device cuda:4 &> ../../data/training.1.rating.log &
python bagofwords.py --datadir ../../data/training --medium 1 --metric watch --device cuda:5 &> ../../data/training.1.watch.log &
python bagofwords.py --datadir ../../data/training --medium 1 --metric plantowatch --device cuda:6 &> ../../data/training.1.plantowatch.log &
python bagofwords.py --datadir ../../data/training --medium 1 --metric drop --device cuda:7 &> ../../data/training.1.drop.log &
wait
if [[ -v RUNPOD_POD_ID ]]; then
    runpodctl remove pod $RUNPOD_POD_ID
fi    
