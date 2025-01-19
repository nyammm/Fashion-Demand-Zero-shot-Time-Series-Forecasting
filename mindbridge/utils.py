import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import config
import pickle
from sklearn.preprocessing import MinMaxScaler
import random


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, use_saved_feat=False):

        # Item id 
        with open(annotations_file, 'rb') as f:
            self.item_list = pickle.load(f)

        # Metadata + date + trend
        self.json_df = pd.read_csv(config.DATASET_PATH)
        self.json_df = self.json_df[self.json_df['item_number_color'].isin(self.item_list)].reset_index(drop=True)

        # Image embedding
        with open(img_dir, 'rb') as f:
            self.img_emb = pickle.load(f)

        # 실제로 존재하는 image embedding
        self.true_img_emb = {key: value for key, value in self.img_emb.items() if key in self.item_list}
        
        categ_dict = torch.load(config.CATEG_DICT)
        color_dict = torch.load(config.COLOR_DICT)
        fabric_dict = torch.load(config.FAB_DICT)

        # Category
        self.category = [categ_dict[x] for x in self.json_df['category'].tolist()]

        # Color
        self.color = [color_dict[x] for x in self.json_df['color'].tolist()]

        # Fabric
        self.fabric = [fabric_dict[x] for x in self.json_df['fabric'].tolist()]

        # Release date
        self.release_date = self.json_df['release_date'].tolist()

        # Labels(0-11)
        new_labels = self.json_df.iloc[:, -12:].values / 1820 # max값으로 나누기
        self.img_labels = pd.Series(new_labels.tolist())
        
        # Path
        self.path = self.json_df["item_number_color"]
        
        # Transform
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        # Image feature
        img_feature = self.img_emb.get(self.path[idx], find_other_img(self.path, self.true_img_emb, idx))
        # img_feature = self.img_emb.get(self.path[idx], np.zeros(512))
        # (512,) -> (1,512)
        img_feature = torch.as_tensor(img_feature, dtype = torch.float).unsqueeze(0)
        
        # Category
        category = self.category[idx]

        # Color
        color = self.color[idx]
        
        # Fabric
        fabric = self.fabric[idx]

        # Release date
        release_date = self.release_date[idx]

        # Temporal features
        temporal_features = [0,0,0,0] # 안쓰는 거라 0으로 채워줌
        temporal_features = torch.as_tensor(temporal_features, dtype=torch.float)

        # Label
        trend = self.img_labels.iloc[idx]
        trend = torch.FloatTensor(trend)

        # 필요 없는 거
        image_transformed = 0

        return (image_transformed, trend, category, color, fabric, release_date, temporal_features, img_feature, self.path.iloc[idx])

# image embedding이 없을 때 비슷한 item에서 뽑아서 noise 추가
def find_other_img(item_list, true_list, idx):

    it = item_list[idx]
    item_cat = it[2:4]
    item_col = it[-2:]
    close_idx = [id for id in true_list.keys() if (id[2:4] == item_cat)&(id[-2:] == item_col)]
    if close_idx == []:
        return np.zeros(512)
    random_idx = random.choice(close_idx)
    img_emb = true_list[random_idx] * np.random.normal(1, 0.001, 512)

    return img_emb


with open('/home/sflab/SFLAB/DATA/mind_br_data_240916/total_cat_trend_240916.pkl', 'rb') as f:
    categ_trend = pickle.load(f)
with open('/home/sflab/SFLAB/DATA/mind_br_data_240916/total_col_trend_240916.pkl', 'rb') as f:
    color_trend = pickle.load(f)
with open('/home/sflab/SFLAB/DATA/mind_br_data_240916/total_fab_trend_240916.pkl', 'rb') as f:
    fabric_trend = pickle.load(f)

def exog_extractor(date, item):

    item = np.asarray(item)

    out_gtrends = []
    weeks = config.EXOG_LEN
    for i in range(item.shape[0]):

        it = item[i]

        cat_gtrend = np.array(categ_trend.get(it, np.zeros(52))[:52])
        col_gtrend = np.array(color_trend.get(it, np.zeros(52))[:52])
        fab_gtrend = np.array(fabric_trend.get(it, np.zeros(52))[:52])

        cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1,1)).flatten()
        col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1,1)).flatten()
        fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1,1)).flatten()
        
        multitrends = np.hstack([cat_gtrend[:weeks], col_gtrend[:weeks], fab_gtrend[:weeks]]).astype(np.float32)

        out_gtrends.append(multitrends)
    out_gtrends = np.vstack(out_gtrends)
    return out_gtrends