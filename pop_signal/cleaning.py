import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import glob
import numpy as np
import cleanlab

from torchvision import models
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from sklearn.model_selection import KFold
from tqdm import tqdm

# None 값 필터링을 위해 함수 추가
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)

def cleaning_image(image_dir):
    """fashionable / unfashionable"""

    # 2가지 라벨의 이미지 파일을 하나의 데이터셋으로 묶어 불러오기
    fashion_images = np.array(glob.glob(f"{image_dir}/fashionable/*/*"))
    unfashion_images = np.array(glob.glob(f"{image_dir}/unfashionable/*/*"))

    # 라벨 부여 (fashionable : 1, unfashionable : 0)
    fashion_labels = np.ones(len(fashion_images), dtype=np.float32)
    unfashion_labels = np.zeros(len(unfashion_images), dtype=np.float32)

    # 이미지와 라벨을 합쳐서 하나의 데이터셋으로 통합
    total_images = np.concatenate([fashion_images, unfashion_images])
    total_labels = np.concatenate([fashion_labels, unfashion_labels])


    n_samples = len(total_images)  # 추가
    if n_samples < 2:
        print("KFold splitting을 위한 샘플 부족")
        return

    device = torch.device("cuda:0")
    n_splits = max(2, min(3, n_samples))
    kfold = KFold(n_splits=n_splits)

    total_prediction = []
    total_ground_truth = []
    total_img_path = []

    for train_idx, valid_idx in kfold.split(total_images):
        # 모델 초기화
        model = models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        
        input_dim = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        model = model.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)

        train_images, train_labels = total_images[train_idx], total_labels[train_idx]
        valid_images, valid_labels = total_images[valid_idx], total_labels[valid_idx]

        train_dataset = PopImageDataset(train_images, train_labels, transform=image_transform)
        valid_dataset = PopImageDataset(valid_images, valid_labels, transform=image_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn)

        model = train(model, train_dataloader, criterion, optimizer)

        valid_ground_truth, valid_prediction, img_path = inference(model, valid_dataloader)
        
        total_ground_truth.append(valid_ground_truth)
        total_prediction.append(valid_prediction)
        total_img_path.extend(img_path)

    total_ground_truth = torch.cat(total_ground_truth, dim=0)
    total_prediction = torch.cat(total_prediction, dim=0)

    # cleanlab을 사용해서 잘못된 이미지 찾기
    total_prediction = torch.cat([1 - total_prediction, total_prediction], dim=1)
    labels = np.array(total_ground_truth.detach().cpu().squeeze(), dtype=np.int64)
    pred_probs = np.array(total_prediction.detach().cpu())

    issue_idx = cleanlab.filter.find_label_issues(labels, pred_probs, return_indices_ranked_by='self_confidence')
    print(f'잘못된 이미지 수 : {len(issue_idx)}개')

    for x in tqdm(issue_idx, desc=">>> image deleting..."):
        del_img_path = total_img_path[x]
        os.remove(del_img_path)


class PopImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"이미지 로딩 에러 {img_path}: {e}")
            return None
        
        return image, torch.FloatTensor([label]), img_path


def train(model, dataloader, criterion, optimizer, num_epochs=3):
    device = torch.device("cuda:0")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for batch in tqdm(dataloader):
            if batch is None:
                continue
            inputs, labels, img_path = batch

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}')
    
    return model


def inference(model, dataloader):
    device = torch.device("cuda:0")

    prediction = []
    ground_truth = []
    total_img_path = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch == -1: continue
            inputs, labels, img_path = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)

            outputs = model(inputs)
            prediction.append(outputs)
            ground_truth.append(labels)
            total_img_path.extend(img_path)
    
    prediction = torch.cat(prediction)
    ground_truth = torch.cat(ground_truth)

    return ground_truth, prediction, total_img_path