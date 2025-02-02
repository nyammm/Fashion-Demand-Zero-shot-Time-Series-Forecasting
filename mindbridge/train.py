import torch
import os
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
import model
import cross_att_model
import model_image
from utils import *
import matplotlib.pyplot as plt
import logging
import config
from skimage.transform import resize
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device(config.DEVICE)

dateTimeObj = datetime.now()
timestamp = dateTimeObj.strftime("%d-%m-%Y-%H-%M")

normalization_values = 1820

model_filename = "{}_lr{}_{}_{}_datas_10ago".format(config.model_types[config.MODEL], config.LEARNING_RATE, "tf_05" if config.USE_TEACHERFORCING else "", f"exog_{config.EXOG_LEN}" if config.USE_EXOG else "")

train_dir_name = model_filename + "_" + timestamp
print(train_dir_name)

attention_dim = config.HIDDEN_SIZE
image_feature_size = config.HIDDEN_SIZE
hidden_size = config.HIDDEN_SIZE * 2
gtrends_len = config.EXOG_LEN* config.EXOG_NUM

#Image or Residual
if config.model_types[config.MODEL] == "image": 
   hidden_size = config.HIDDEN_SIZE
   if config.USE_EXOG:
      input_size = config.HIDDEN_SIZE +gtrends_len
   else:
      input_size = config.HIDDEN_SIZE 

elif config.model_types[config.MODEL] == "residual":
   hidden_size = config.HIDDEN_SIZE * 2
   if config.USE_EXOG:
         input_size = config.HIDDEN_SIZE +gtrends_len
   else:
      input_size = config.HIDDEN_SIZE
#Concat or Cross (cross doesn't use these)
else:
   hidden_size = config.HIDDEN_SIZE*2
   if config.USE_EXOG:
      input_size = config.HIDDEN_SIZE*3 +gtrends_len
   else:
      input_size = config.HIDDEN_SIZE*3


# Logger
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
      logging.FileHandler(model_filename[:-4]+'.log'),
      logging.StreamHandler()
    ])


dataset_path = config.IMAGE_PATH


normalize = transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)

ds_trans = transforms.Compose([
                        transforms.ToTensor(),
                        normalize])

train_dataset = CustomDataset(config.TRAIN_DATASET, dataset_path, transform=ds_trans)
valid_dataset = CustomDataset(config.VALID_DATASET, dataset_path, transform=ds_trans)
test_dataset = CustomDataset(config.TEST_DATASET, dataset_path, transform=ds_trans)

train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
valid_dataloader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

encoder = model.Encoder()

#Image
if config.model_types[config.MODEL] == "image":
   decoder = model_image.DecoderRNN(input_size, hidden_size)
   model = model_image.EncoderDecoder(attention_dim, image_feature_size, hidden_size, encoder, decoder, use_teacher_forcing=config.USE_TEACHERFORCING).to(device)
#Cross
elif config.model_types[config.MODEL] == "cross":
   model = cross_att_model.EncoderDecoder(12, device, use_teacher_forcing=config.USE_TEACHERFORCING).to(device)
#Concat or Residual
else:
   decoder = model.DecoderRNN(input_size, hidden_size)
   model = model.EncoderDecoder(attention_dim, image_feature_size, hidden_size, encoder, decoder, use_teacher_forcing=config.USE_TEACHERFORCING).to(device)


def find_model_file(directory):
  best_model_file = ""
  for file in os.listdir(directory):
      if file.endswith(".bpt"):
          best_model_file = file
  return os.path.join(directory, best_model_file)


def trainModel(model, train_dataloader):
   # Loss and optimizer
   criterion = nn.MSELoss()
   criterionL1 = nn.L1Loss()
   model_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LEARNING_RATE * 1e-2)

   # Train the models
   total_step = len(train_dataloader)
   best_score = float('inf')

   # image embedding

   for epoch in range(config.NUM_EPOCHS):
      ep_loss = 0
      model.train()
      for i, (images, trend, categ, color, fabric, release_date, temporal_features, img_feature, item) in enumerate(train_dataloader):
         model_optimizer.zero_grad()

         images = images.to(device)
         trend = trend.squeeze().to(device)

         release_date = np.asarray(release_date)

         exogeneous_params = torch.from_numpy(exog_extractor(release_date, item)).to(device)
         categ = categ.to(device)
         color = color.to(device)
         fabric = fabric.to(device)

         temporal_features = temporal_features.to(device)

         #Image
         if config.model_types[config.MODEL] == "image":
            if config.USE_SAVED_FEATURES:
               outputs, _ = model(images, exogeneous_params, target=trend, img_feature=img_feature.to(device))
            else:
               outputs, _ = model(images, exogeneous_params, target=trend)
         #Cross
         elif config.model_types[config.MODEL] == "cross":
            if config.USE_SAVED_FEATURES:
               outputs = model(images, categ, color, fabric, temporal_features, exogeneous_params, target=trend, feats=img_feature.to(device))
            else:
               outputs = model(images, categ, color, fabric, temporal_features, exogeneous_params, target=trend)
         #Concat or Residual
         else:
            if config.USE_SAVED_FEATURES:
               outputs, _ = model(images, categ, color, fabric, temporal_features, exogeneous_params, target=trend, img_feature=img_feature.to(device))
            else:
               outputs, _ = model(images, categ, color, fabric, temporal_features, exogeneous_params, target=trend)

         # Forward, backward and optimize
         loss = criterion(outputs.to(device), trend)
         loss.backward()

         new_out = outputs * normalization_values
         new_out = new_out.to(device)
         new_trd = trend * normalization_values

         avg_batch_wmape = torch.sum(torch.sum(torch.abs(new_trd - new_out), dim=-1)) / torch.sum(torch.sum(new_trd, dim=-1))
         batch_adj_smape = torch.sum(torch.sum(torch.abs(new_trd - new_out), dim=-1) / torch.sum(torch.abs(new_trd) + torch.abs(new_out), dim=-1)) / config.BATCH_SIZE
         batch_accum_smape = torch.sum(torch.abs(torch.sum(new_trd, dim=-1) - torch.sum(new_out, dim=-1)) / torch.sum(torch.abs(new_trd) + torch.abs(new_out), dim=-1), dim=-1) / config.BATCH_SIZE
         model_optimizer.step()
         ep_loss += loss.item()

         # Print log info
         if i % 10 == 0: 
            logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, MAE: {:.4f}, adj-SMAPE: {:.4f}, accum-SMAPE: {:.4f}, wMAPE: {:.4f}'.format(epoch + 1, config.NUM_EPOCHS, i, total_step, criterion(new_out, new_trd).detach().cpu(), criterionL1(new_out, new_trd).detach().cpu(), batch_adj_smape.item(), batch_accum_smape.item(), avg_batch_wmape.item()))
      epoch_score = evaluate(model, valid_dataloader)

      if epoch_score < best_score:
         best_score = epoch_score
         if not os.path.exists(train_dir_name):
            os.mkdir(train_dir_name)

         for file in os.listdir(train_dir_name):
            if file.endswith('.bpt'):
               os.remove(os.path.join(train_dir_name, file)) 

         save_dir = os.path.join(train_dir_name, str(epoch+1) + ".bpt")
         torch.save(model.state_dict(), save_dir)

   return model


def evaluate(model, test_dataloader, show_plots=False):

   criterionL1 = nn.L1Loss()

   mae_mean = []
   model.eval()
   outs, gts, codes_list = [], [], []
   with torch.no_grad():
      for index, elem in enumerate(test_dataloader):
         # Unpacking
         images, trend, categ, color, fabric, release_date, temporal_features, img_feature, codes = elem

         images = images.to(device)
         release_date = np.asarray(release_date)
         exogeneous_params = torch.from_numpy(exog_extractor(release_date, codes)).to(device)
         categ = categ.to(device)
         color = color.to(device)
         fabric = fabric.to(device)

         temporal_features = temporal_features.to(device)

         #Image
         if config.model_types[config.MODEL] == "image":
            outputs, _ = model(images, exogeneous_params, img_feature=img_feature.to(device))
         #Cross
         elif config.model_types[config.MODEL] == "cross":
            outputs = model(images, categ, color, fabric, temporal_features, exogeneous_params, feats=img_feature.to(device))
         #Concat or Residual
         else:
            outputs, _ = model(images, categ, color, fabric, temporal_features, exogeneous_params, img_feature=img_feature.to(device))
         
         outputs = outputs.cpu()
         
         if config.NORM:
            trend_norm = trend.squeeze() 
            out_norm = outputs 
         else:
            trend_norm = trend.squeeze() * normalization_values
            out_norm = outputs * normalization_values

         outs = outs + [out_norm]
         gts = gts + [trend_norm]
         codes_list = codes_list + list(codes)


      outputs = torch.cat(outs, dim=0)
      trend = torch.cat(gts, dim=0)
      
      mae_mean = criterionL1(outputs, trend).detach().cpu()
      wMAPE = torch.sum(torch.sum(torch.abs(trend - outputs), dim=-1)) / torch.sum(torch.sum(trend, dim=-1))
      wape_mean = wMAPE.item()
      adj_SMAPE = torch.sum(torch.abs(trend - outputs), dim=-1) / torch.sum(torch.abs(trend) + torch.abs(outputs), dim=-1)
      accum_SMAPE = torch.abs(torch.sum(trend, dim=-1) - torch.sum(outputs, dim=-1)) / torch.sum(torch.abs(trend) + torch.abs(outputs), dim=-1)
      logging.info("mae_mean: {:.4f}".format(mae_mean))
      logging.info("adj_SMAPE: {:.4f}".format(torch.mean(adj_SMAPE).item()))
      logging.info("accum_SMAPE: {:.4f}".format(torch.mean(accum_SMAPE).item()))
      logging.info("wMAPE_mean: {:.4f}".format(wape_mean))

   
   weeks = [12,8,6,4]
   
   for w in weeks:
      outputs_np = np.array([out[0:w] for out in outputs.numpy()])
      trend_np = np.array([tr[0:w] for tr in trend.numpy()])
      codes = [sub[5:-4] for sub in codes_list]
      file = {"results": outputs_np,"gts": trend_np, "codes": codes}
      torch.save(file, model_filename+str(w)+"_dict.pth")
 
   return wape_mean


model = trainModel(model, train_dataloader)

 
logging.info("Saved log " + train_dir_name)
logging.info("Evaluating")
best_model_file = find_model_file(os.path.join(train_dir_name))
model.load_state_dict(torch.load(best_model_file, map_location=lambda storage, loc: storage.cuda(0)))
model.eval()
evaluate(model, test_dataloader, show_plots=False)
