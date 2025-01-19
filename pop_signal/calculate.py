import torch
import numpy as np
import glob, pickle
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from PIL import Image
from torchvision import transforms
from fashion_clip.fashion_clip import FashionCLIP
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 


def calculate_POP(image_dir, num=None, batch_size=256):
    item_num = image_dir.split("/")[-1]
    fclip = pickle.load(open("fclip_img.pkl", "rb"))
    target = fclip[item_num] #[,512]
    target = np.array(target).reshape(1, -1)

    """cleaned images -> cosine Similarity 계산"""

    model = FashionCLIP('fashion-clip')
    weeks = glob.glob(f'{image_dir}/fashionable/*')
    pop = []
    for x in tqdm(weeks, desc=f">>> Calcuate POP signal of {item_num}..."):
        image_list = glob.glob(f'{x}/*')
        if not image_list:
            print(f"\n{x}에 이미지 없으므로 skip")
            pop.append(0)
            continue
        output = model.encode_images(image_list, batch_size=batch_size)
        sim = cosine_similarity(target, output).flatten()
        pop.append(np.mean(sim))

    return item_num, np.array(pop)
