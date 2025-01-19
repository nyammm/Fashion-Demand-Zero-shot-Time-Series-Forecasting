import os
import pickle
import pandas as pd
import argparse
import shutil

from cleaning import cleaning_image
from calculate import calculate_POP
from collections import deque
import time

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pop_signal(args):
    ckpt_path = f"{args.save_dir}/{args.version}/pop_idx.pickle"

    if os.path.exists(ckpt_path):
        print(f"기존 체크포인트 무시: {ckpt_path}")
        with open(ckpt_path, 'rb') as f:
            result = pickle.load(f)
    else:
        os.makedirs(f"{args.save_dir}/{args.version}", exist_ok=True)
        result = {}
    
    fclip = pickle.load(open(args.fclip_path, "rb")) #AI19_04442 없음
    sales_df = pd.read_csv(args.sales_path, index_col="image_path")

    total_list = list(sales_df.index)
    n = len(sales_df)
    v = int(args.version[-1])
    total_list = total_list[(v-1)*n // args.n_split: v*n // args.n_split]
    
    # 크롤링 completed items
    # complete = pickle.load(open(f'completed_items_{args.version}.pkl', "rb"))
    # total_list = [c.split('.')[0].replace('/','_') for c in complete]

    # queue 초기화
    queue = deque([item for item in total_list if item not in result]) # 이미 완료된 아이템 제외
    print(f"처리되지 않은 아이템 {len(queue)}개 남음")

    i = 0
    while queue:
        item_num = queue.pop()
        print(f"--- Current Item : {item_num} ---")
        if item_num not in sales_df.index:
            continue

        image_set_dir = f"{args.data_dir}/{args.version}"
        image_dir = f"{image_set_dir}/{item_num}"
        if not os.path.exists(image_set_dir):
            os.makedirs(image_set_dir)
        
        # 노이즈 제거
        try:
            cleaning_image(image_dir)
        except ValueError as e:
            if str(e) == "Labels must contain at least 2 classes.":
                shutil.rmtree(image_dir)
                continue

        # signal 추출 
        key, value = calculate_POP(image_dir)
        result[key] = value
        print(f'>>> Pop index fin --- {item_num}: {value}')
        print(f'>>> ----------------- {len(queue)}개 남음')
        
        with open(ckpt_path, 'wb') as f:
            pickle.dump(result, f)
        
        
        # if os.path.exists(image_dir):
        #     shutil.rmtree(image_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pop Signal')
    parser.add_argument('--version', type=str, default='v6')
    parser.add_argument('--n_split', type=int, default=6)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default='download')
    parser.add_argument('--save_dir', type=str, default='result_20250114')
    parser.add_argument('--sales_path', type=str, default='data/sales_train.csv')
    parser.add_argument('--fclip_path', type=str, default='data/fclip_img.pkl')
    
    args = parser.parse_args()
    
    pop_signal(args)