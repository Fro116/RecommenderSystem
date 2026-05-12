import os
import logging
import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
from concurrent.futures import ThreadPoolExecutor

def stringhash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def extract_bytes(fn: str) -> bytes:
    try:
        with Image.open(fn) as img:
            # flatten pixels in column major order
            byte_array = bytearray()
            for frame in ImageSequence.Iterator(img):
                img_array = np.asarray(frame)
                if img_array.ndim == 3:
                    transposed = img_array.transpose((1, 0, 2))
                elif img_array.ndim == 2:
                    transposed = img_array.transpose((1, 0))
                else:
                    transposed = img_array
                byte_array.extend(transposed.tobytes())
            return bytes(byte_array)
    except Exception as e:
        logging.error(f"could not extract bytes for {fn}: {e}")
        with open(fn, 'rb') as f:
            return f.read()

def add_imagehashes(datadir: str):
    csv_path = os.path.join(datadir, "images.csv")
    df = pd.read_csv(csv_path, low_memory=False)
    df["imagehash"] = ""
    logging.getLogger().setLevel(logging.ERROR)

    def process_row(idx_row):
        idx, row = idx_row
        if not row['saved']:
            return idx, None
        fn = os.path.join(datadir, "images", str(row['filename']))
        assert os.path.exists(fn), f"File not found: {fn}"
        return idx, stringhash(extract_bytes(fn))

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_row, df.iterrows())
        for idx, hash_val in results:
            if hash_val is not None:
                df.at[idx, "imagehash"] = hash_val
    df.to_csv(csv_path, index=False)


add_imagehashes("../../../data/import/images")
