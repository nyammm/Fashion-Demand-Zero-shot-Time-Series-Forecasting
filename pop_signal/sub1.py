import os
import pickle
import pandas as pd
import argparse
from datetime import datetime

from crawl import crawl_image_per_item_concurrently
from collections import deque
import wandb

wandb.login(key='?', relogin=True, force=True)

def crawling(args):
    stfore = pd.read_csv(args.sales_df_root)
    stfore['release_date'] = pd.to_datetime(stfore['release_date'])

    # Define checkpoint file
    ckpt_path = f"completed_items_{args.version}.pkl"

    # Load or initialize completed items list
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            completed_items = pickle.load(f)
        print(f"Loaded {len(completed_items)} completed items from checkpoint.")
    else:
        completed_items = []

    # Extract unique items from the CSV file
    unique_items = stfore[['image_path', 'category', 'color']].drop_duplicates()

    n = len(unique_items)
    v = int(args.version[-1])
    total_list_per_version = unique_items.iloc[(v-1)*n // args.n_split: v*n // args.n_split]

    # Filter out completed items
    remaining_items = total_list_per_version[~total_list_per_version['image_path'].isin(completed_items)]

    # Initialize queue
    queue = deque(remaining_items.itertuples(index=False))
    print(f"처리되지 않은 품목 {len(queue)}개 남음")

    dt_string = datetime.now().strftime("%Y%m%d-%H%M")[2:]
    model_savename = dt_string + '_' + args.train_test + '_' + args.version

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_proj,
        name=model_savename,
        dir=args.wandb_dir
    )

    while queue:
        item = queue.popleft()
        image_path, category, color = item.image_path, item.category, item.color
        print(">"*20, f'Current Item is {image_path}')

        # Generate query from category and color
        query = f"{color} {category}"

        # Create release_date based on the first occurrence in the CSV
        release_date = stfore[stfore['image_path'] == image_path]['release_date'].iloc[0]

        # Define image directory
        # image_name = '_'.join(image_path.split('.')[0].split('/'))
        image_dir = f'{args.version}/{image_path}'
        for label in ["fashionable", "unfashionable"]:
            os.makedirs(os.path.join(image_dir, label), exist_ok=True)

        # Perform image crawling
        try:
            crawl_image_per_item_concurrently(query, release_date, image_dir, keywords="fashionable", max_workers=args.max_workers)
            crawl_image_per_item_concurrently(query, release_date, image_dir, keywords="unfashionable", max_workers=args.max_workers)

            # Update completed items and save checkpoint
            completed_items.append(image_path)
            with open(ckpt_path, 'wb') as f:
                pickle.dump(completed_items, f)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawling')
    parser.add_argument('--version', type=str, default='v6')
    parser.add_argument('--n_split', type=int, default=6)
    parser.add_argument('--max_workers', type=int, default=8)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='ssl_project')
    parser.add_argument('--wandb_proj', type=str, default='pop_signal')
    parser.add_argument('--wandb_dir', type=str, default='../')

    args = parser.parse_args()

    args.train_test = 'train'
    args.sales_df_root = f'data/sales_{args.train_test}.csv'

    crawling(args)
