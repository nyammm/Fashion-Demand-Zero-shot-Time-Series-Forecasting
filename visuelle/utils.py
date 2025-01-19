import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from image_lib import resize_to_square
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import config
from sklearn.preprocessing import MinMaxScaler
import random

normalization_values = np.load(config.NORMALIZATION_VALUES_PATH)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, use_saved_feat=False):
        self.json_df = pd.read_csv(annotations_file)
        self.json_df.loc[self.json_df["image_path"]=='AI19/04442.png', "image_path"] = "AI19/05385.png"
        self.img_dir = img_dir
        self.use_saved_feat = use_saved_feat


        categ_dict = torch.load(config.CATEG_DICT)
        color_dict = torch.load(config.COLOR_DICT)
        fabric_dict = torch.load(config.FAB_DICT)


        # Category
        self.category = [categ_dict[x] for x in self.json_df['category'].tolist()]
        # Color
        self.color = [color_dict[x] for x in self.json_df['color'].tolist()]
        # self.color = [color_dict[x] for x in self.json_df['exact_color'].tolist()]
        # Fabric
        self.fabric = [fabric_dict[x] for x in self.json_df['fabric'].tolist()]
        # self.fabric = [fabric_dict[x] for x in self.json_df['texture'].tolist()]


        # Release date
        self.release_date = self.json_df['release_date'].tolist()

        # time_feature_range = pd.date_range(self.release_date - relativedelta(weeks=52), 
        #                                   self.release_date + relativedelta(weeks=self.args.pred_len - 1), freq='7d')
        # temporal_features = [time_features(time_feature_range, freq='w')[0].tolist(),
        #                     time_features(time_feature_range, freq='m')[0].tolist(), 
        #                     time_features(time_feature_range, freq='y')[0].tolist()]

        # #Days
        # self.days = pd.to_datetime(self.release_date).day
        # #Weeks
        # self.weeks = pd.to_datetime(self.release_date).isocalendar().week
        # #Months
        # self.months = pd.to_datetime(self.release_date).month
        # #Years
        # self.years = pd.to_datetime(self.release_date).year

        # Labels
        new_labels =self.json_df.iloc[:, -12:].values.tolist()
        self.img_labels = pd.Series(new_labels)
        
        # Path
        self.path = self.json_df["image_path"]
        
        # Transform
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        # Image
        
        img_path = os.path.join(self.img_dir, self.path.iloc[idx])
        image = cv2.imread(img_path)
        image = resize_to_square(image)
        image_2 = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        orig_8x8 = cv2.resize(image_2, (8,8), interpolation = cv2.INTER_AREA)

        # Image feature
        img_feature = np.array([])
        if config.USE_SAVED_FEATURES:
            feat_path = os.path.join(config.SAVED_FEATURES_PATH, self.path.iloc[idx].replace(".png", ".pth"))
            img_feature=torch.load(feat_path).squeeze()
        

        # Category
        category = self.category[idx]
        # Color
        color = self.color[idx]
        
        # Fabric
        fabric = self.fabric[idx]

        # Release date
        release_date = self.release_date[idx]

        #Temporal features
        temporal_features = [0,0,0,0]
        # temporal_features.append(self.days[idx])
        # temporal_features.append(self.weeks.iloc[idx])
        # temporal_features.append(self.months[idx])
        # temporal_features.append(self.years[idx])
        temporal_features = torch.as_tensor(temporal_features, dtype=torch.float)


        # Label
        trend = np.array(self.img_labels.iloc[idx]) / 751 # max 값으로 scaling
        trend = torch.FloatTensor(trend)
        
        # Applying transform
        if self.transform:
            image_transformed = self.transform(image)
        if self.target_transform:
            trend = self.target_transform(trend)

        return (image_transformed, trend, category, color, fabric, orig_8x8, release_date, temporal_features, img_feature, self.path.iloc[idx])


def resize2d(img, size):
    from torch.autograd import Variable
    return (F.adaptive_avg_pool2d(Variable(img,volatile=True), size)).data

def exog_extractor(date, categ, color, fabric):

    categ = np.asarray(categ)
    color = np.asarray(color)
    fabric = np.asarray(fabric)

    gtrends = pd.read_csv(config.COMPOSED_GTREND, parse_dates=['date'], index_col=[0])

    out_gtrends = []
    weeks = config.EXOG_LEN
    for i in range(categ.shape[0]):
        categ_dict = torch.load(config.CATEG_DICT, weights_only=True)
        categ_dict = {v: k for k, v in categ_dict.items()}

        color_dict = torch.load(config.COLOR_DICT, weights_only=True)
        color_dict = {v: k for k, v in color_dict.items()}

        fabric_dict = torch.load(config.FAB_DICT, weights_only=True)
        fabric_dict = {v: k for k, v in fabric_dict.items()}

        cat = categ_dict[categ[i]]
        col = color_dict[color[i]]
        fab = fabric_dict[fabric[i]]

        start_date = pd.to_datetime(date[i])
        gtrend_start = start_date - pd.DateOffset(weeks=52)
        # 51주만 나올 때가 있어서 가장 가까운 날짜 찾아서 52주 나오게 함
        closest_date = gtrends.index[abs((gtrends.index - gtrend_start).to_numpy()).argmin()]
        cat_gtrend = gtrends.loc[closest_date:start_date][cat][-52:].values
        col_gtrend = gtrends.loc[closest_date:start_date][col][-52:].values
        fab_gtrend = gtrends.loc[closest_date:start_date][fab][-52:].values
        # fab_gtrend = gtrends.loc[gtrend_start:start_date][fab.replace(' ','')][-52:].values

        # 만약 51주 나오면 마지막 값 복사해서 사용
        if len(cat_gtrend) < 52:
            cat_gtrend = np.append(cat_gtrend, cat_gtrend[-1])
        if len(col_gtrend) < 52:
            col_gtrend = np.append(col_gtrend, col_gtrend[-1])
        if len(fab_gtrend) < 52:
            fab_gtrend = np.append(fab_gtrend, fab_gtrend[-1])

        cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1,1)).flatten()
        col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1,1)).flatten()
        fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1,1)).flatten()
        
        multitrends = np.hstack([cat_gtrend[:weeks], col_gtrend[:weeks], fab_gtrend[:weeks]]).astype(np.float32)

        out_gtrends.append(multitrends)
    out_gtrends = np.vstack(out_gtrends)
    return out_gtrends
