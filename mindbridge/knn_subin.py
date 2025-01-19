import os, pickle
import numpy as np
import pandas as pd
import argparse
import torch
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
from copy import deepcopy
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError, MeanAbsoluteError

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def calculate_metrics(predict, label):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ad_smape = SymmetricMeanAbsolutePercentageError().to(device)
    mae = MeanAbsoluteError().to(device)

    def wmape_func(label, predict):
        return torch.abs(label - predict).sum() / label.sum()

    mae = [mae(label[i,:], predict[i,:]).cpu().item() for i in range(label.size(0))]
    wmape = [wmape_func(label[i,:], predict[i,:]).cpu().item() for i in range(label.size(0))]
    adj_smape = [ad_smape(label[i,:], predict[i,:]).cpu().item() * 0.5 for i in range(label.size(0))]

    gt = torch.sum(label, dim=0)
    pred = torch.sum(predict, dim=0)
    accum_smape = ad_smape(gt, pred) * 0.5

    return np.mean(mae), np.mean(wmape), np.mean(adj_smape), accum_smape.cpu().item()


def load_csv():
    data = pd.read_csv(os.path.join(args.data_folder + 'total_data.csv'), parse_dates=['release_date'])
    with open('test_list.pkl', 'rb') as f:
            test_list = pickle.load(f)
    train_data = data[~data['item_number_color'].isin(test_list)].reset_index(drop=True)
    test_data = data[data['item_number_color'].isin(test_list)].reset_index(drop=True)

    #norm_scale = int(np.load(config.NORMALIZATION_VALUES_PATH))

    models_dict, colors_dict, fabric_dict = {}, {}, {}
    idx_model, idx_color, idx_fabric = 0, 0, 0
    tags = []
    img_paths = []
    series = []
    codes = []
    splits = [] 

    train_codes = train_data.index.values
    for code in tqdm(train_codes):
        codes.append(code)
        item = train_data.loc[code]
        series.append([item[str(i)] for i in range(12)])

        img_paths.append(item['item_number_color'])
        model = item['category']
        color = item['color']
        fabric = item['fabric']

        if model not in models_dict:
            models_dict[model] = idx_model
            idx_model += 1
        if color not in colors_dict:
            colors_dict[color] = idx_color
            idx_color += 1
        if fabric not in fabric_dict:
            fabric_dict[fabric] = idx_fabric
            idx_fabric +=1

        tags.append([models_dict[model], colors_dict[color], fabric_dict[fabric]])
        splits.append(0)

    test_codes = test_data.index.values
    for code in tqdm(test_codes):

        codes.append(code)
        item = test_data.loc[code]
        series.append([item[str(i)] for i in range(12)])

        # ['ORANGE', 'WOVEN', 'BL']
        img_paths.append(item['item_number_color'])
        model = item['category']
        color = item['color']
        fabric = item['fabric']

        if model not in models_dict:
            models_dict[model] = idx_model
            idx_model += 1
        if color not in colors_dict:
            colors_dict[color] = idx_color
            idx_color += 1
        if fabric not in fabric_dict:
            fabric_dict[fabric] = idx_fabric
            idx_fabric +=1

        tags.append([models_dict[model], colors_dict[color], fabric_dict[fabric]])
        splits.append(1)
    

    tags = np.stack(tags)
    splits = np.stack(splits)
    series = np.stack(series)

    # if config.NORM:
    #     series = series / norm_scale

    return tags, codes, series, splits, img_paths


# batch 단위로 거리 계산
def compute_batch_similarity(tags, num):
    batch_size = 2000
    tag = tags[:, num:num+1]
    dist = []
    process = 0
    for i in range(0, tag.size(0), batch_size):
        batch_tag = tag[i:i+batch_size]
        batch_dist = (batch_tag.unsqueeze(1) == tag).to(torch.bool)
        dist.append(batch_dist.cpu())
        process += batch_dist.shape[0]
        if process % 10000 == 0:
            print("[tag {}] Processing: {}/{}".format(num, process, tag.size(0)))

    return torch.cat(dist, dim=0)


def compute_similarity(tags, codes, img_emb, mode="tags"):
    """유사도 행렬을 GPU에서 계산."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == "tags":
        tags = torch.tensor(tags, device=device)
        similarity_matrix = compute_batch_similarity(tags, 0).to(torch.uint8)
        dist2 = compute_batch_similarity(tags, 1).to(torch.uint8)
        dist3 = compute_batch_similarity(tags, 2).to(torch.uint8)
        similarity_matrix.add_(dist2)
        del dist2
        similarity_matrix.add_(dist3)
        del dist3
        torch.sub(1, similarity_matrix, out=similarity_matrix)
        similarity_matrix = similarity_matrix.squeeze(-1).to(torch.float64)
        similarity_matrix /= 3

    elif mode == "img":
        stacked_array = torch.tensor([img_emb.get(code, np.zeros(512)) for code in codes], device=device)
        similarity_matrix = []
        process = 0
        for i in range(0, stacked_array.size(0), 2000):
            batch = stacked_array[i:i + 2000]         
            batch_dist = torch.cdist(batch, stacked_array)
            similarity_matrix.append(batch_dist.cpu())
            process += batch_dist.shape[0]
            if process % 10000 == 0:
                print("Processing: {}/{}".format(process, stacked_array.size(0)))

        similarity_matrix = torch.cat(similarity_matrix, dim=0)

    elif mode == "combined":
        # Tags + Image Embedding 결합
        tags = torch.tensor(tags, device=device)
        similarity_matrix = compute_batch_similarity(tags, 0).to(torch.uint8)
        dist2 = compute_batch_similarity(tags, 1).to(torch.uint8)
        dist3 = compute_batch_similarity(tags, 2).to(torch.uint8)
        similarity_matrix.add_(dist2)
        del dist2
        similarity_matrix.add_(dist3)
        del dist3
        torch.sub(1, similarity_matrix, out=similarity_matrix)
        similarity_matrix = similarity_matrix.squeeze(-1).to(torch.float64)
        similarity_matrix /= 3
        
        stacked_array = torch.tensor([img_emb.get(code, np.zeros(512)) for code in codes], device=device)
        imgs_similarity = []
        process = 0
        print("Calculating img similarity...")
        for i in range(0, stacked_array.size(0), 2000):
            batch = stacked_array[i:i + 2000]         
            batch_dist = torch.cdist(batch, stacked_array).to(torch.float16)
            imgs_similarity.append(batch_dist.cpu())
            process += batch_dist.shape[0]
            if process % 10000 == 0:
                print("Processing: {}/{}".format(process, stacked_array.size(0)))
        imgs_similarity = torch.cat(imgs_similarity, dim=0)

        similarity_matrix.mul_(0.5).add_(imgs_similarity.mul_(0.5))
        del imgs_similarity

    return similarity_matrix.cpu().numpy()


def estimate_trend_gpu(args, sim_mat, series, splits):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sim_mat_test = torch.tensor(sim_mat[splits == 1, :][:, splits == 0], device=device)  # numpy -> tensor 변환
    series = torch.tensor(series, device=device)
    splits = torch.tensor(splits, device=device)

    new_series = []
    for i in tqdm(range(sim_mat_test.size(0))):  # .size(0)로 수정
        best = torch.topk(sim_mat_test[i], k=args.k, largest=True).indices
        norm_coeff = sim_mat_test[i, best].sum()
        new_serie_tmp = torch.zeros((series.size(1),), device=device)

        for kk_n in best:
            new_serie_tmp += (sim_mat_test[i, kk_n] / norm_coeff) * series[splits == 0, :][kk_n]

        new_series.append(new_serie_tmp)

    return torch.stack(new_series).cpu().numpy()



def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args.fclip_image = "image_embedding.pkl"
    args.data_folder = "dataset/"

    with open(args.fclip_image, 'rb') as f:
        img_emb = pickle.load(f)
    # img_emb = {key: value[0] for key, value in img_emb.items()}

    print("Loading dataset...")
    tags, codes, series, splits, img_paths = load_csv()

    print("Computing similarity matrix...")
    if args.exp_num == 1:
        similarity_matrix = compute_similarity(tags, codes, img_emb, mode="tags")
    elif args.exp_num == 2:
        similarity_matrix = compute_similarity(tags, img_paths, img_emb, mode="img")
    elif args.exp_num == 3:
        similarity_matrix = compute_similarity(tags, img_paths, img_emb, mode="combined")

    # Normalizing
    min_val = similarity_matrix.min()
    max_val = similarity_matrix.max()
    similarity_matrix -= min_val
    similarity_matrix /= (max_val - min_val)
    print("Forecasting new series...")
    new_series = estimate_trend_gpu(args, similarity_matrix, series, splits)

    args.window_test_start = 0
    args.window_test_end = series.shape[1]

    pred = torch.tensor(new_series[:, args.window_test_start:args.window_test_end].T, device=device)
    gt = torch.tensor(series[splits == 1, args.window_test_start:args.window_test_end].T, device=device)
    
    with torch.no_grad():
        mae, wmape, adj_smape, accum_smape = calculate_metrics(pred, gt)

    model = [0,"tag","image","tag+image"]
    print("Resulting in {} KNN".format(model[args.exp_num]))
    print(f"MAE: {mae}\nWMAPE: {wmape}\nadj_SMAPE: {adj_smape}\naccum_SMAPE: {accum_smape}")
    print("-----------------------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KNN Baselines")

    parser.add_argument("--exp_num", type=int, help="1->KNN, 2->Embedded KNN with image, 3-> Embedded KNN with all", default=1)
    parser.add_argument('--k', type=int, default=11)
    parser.add_argument('--shuffle', type=int, default=50)
    parser.add_argument('--window_test_start', type=int, default=None)
    parser.add_argument('--window_test_end', type=int, default=12)
    parser.add_argument('--save_path', type=str, default="results")
    parser.add_argument('--save_tag', type=str, default="img_12")

    args = parser.parse_args()
    run(args)
