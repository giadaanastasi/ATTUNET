from utils import *
from dataset import *
from model import AttentionUNet, ModelWrapper
from train import *
from loss import *
from stratification import *

from torchvision.utils import make_grid
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.model_selection import KFold


if torch.cuda.is_available():
  print("GPU Available:", torch.cuda.is_available())
  print("Numbers of GPUs:", torch.cuda.device_count())

  print("GPU Name:", torch.cuda.get_device_name(0))
  print("Total memory (MB):", torch.cuda.get_device_properties(0).total_memory / 1024 ** 2)
else:
  print("No available GPUs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_path = "path\to\your\images"
masks_path = "path\to\your\masks"
image_files = sorted(os.listdir(images_path))
mask_files = sorted(os.listdir(masks_path))

columns = ['earlyStop','Name' ,'Precision', 'Recall', 'F1Score', 'Q1_Precision', 'Q1_Recall', 'Q1_F1',
           'Q3_Precision', 'Q3_Recall', 'Q3_F1']
results = pd.DataFrame(columns=columns)

num_total = len(image_files)
assert len(image_files) == len(mask_files), "Image and mask counts do not match!"

model = AttentionUNet().to(device)
model.load_state_dict(torch.load(os.path.join(data_dir,'Unet1_round_'+str(round)+'.pth')))

transform = transforms.Compose([
    #transforms.Resize((128, 128)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

val_dataset = CustomDataset(valX, valY, images_path, masks_path,model.name, transform=transform)
test_dataset = CustomDataset(testX, testY, images_path, masks_path,model.name, transform=transform)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

precision, recall, f1 = evaluate_model(model, val_loader, device)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

test_list = test_dataset.get_list()

avg_metrics = evaluate_model_per_slice(model, test_loader, test_list, round, device)

for patient in test_list:
    if "4124" in patient or "26161" in patient or "31750" in patient or "20970" in patient:
        patient_id = str(patient.split('_')[0]) + "_" + str(patient.split('_')[1])
    else:
        patient_id = patient.split('_')[0]
    results.at[patient_id, 'earlyStop'] = epoch
    results.at[patient_id, 'Name'] = patient_id
    results.at[patient_id, 'Precision'] = avg_metrics[patient_id]['precision']
    results.at[patient_id, 'Recall'] = avg_metrics[patient_id]['recall']
    results.at[patient_id, 'F1Score'] = avg_metrics[patient_id]['f1']
    results.at[patient_id, 'Q1_Precision'] = avg_metrics[patient_id]['q1_precision']
    results.at[patient_id, 'Q1_Recall'] = avg_metrics[patient_id]['q1_recall']
    results.at[patient_id, 'Q1_F1'] = avg_metrics[patient_id]['q1_f1']
    results.at[patient_id, 'Q3_Precision'] = avg_metrics[patient_id]['q3_precision']
    results.at[patient_id, 'Q3_Recall'] = avg_metrics[patient_id]['q3_recall']
    results.at[patient_id, 'Q3_F1'] = avg_metrics[patient_id]['q3_f1']

results.to_csv(os.path.join(data_dir,'METRICS/metrics_global.csv'), index=False)
