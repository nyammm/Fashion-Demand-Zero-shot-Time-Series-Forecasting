import torch

DEVICE = torch.device('cuda:2')
DATASET_PATH = "visuelle2_benchmark/images"
TRAIN_DATASET = "visuelle2_benchmark/stfore_train.csv"
TEST_DATASET = "visuelle2_benchmark/stfore_test.csv"
COMPOSED_GTREND = "visuelle2_benchmark/vis2_gtrends_data.csv"
CATEG_DICT = "visuelle2_benchmark/category_labels.pt"
COLOR_DICT = "visuelle2_benchmark/color_labels.pt"
FAB_DICT = "visuelle2_benchmark/fabric_labels.pt"
NUM_EPOCHS = 50
USE_TEACHERFORCING = True
TF_RATE = 0.5
LEARNING_RATE = 0.0001
NORMALIZATION_VALUES_PATH = "visuelle2_benchmark/stfore_sales_norm_scalar.npy"
BATCH_SIZE= 32
SHOW_PLOTS = False
NUM_WORKERS = 8
USE_EXOG = True
EXOG_NUM = 3
EXOG_LEN = 52
HIDDEN_SIZE = 300
SAVED_FEATURES_PATH = "incv3_features"
USE_SAVED_FEATURES = False # image embedding 있으면 true
NORM = True # normalization 된거면 true
model_types = ["image", "concat", "residual", "cross"]
MODEL = 0
