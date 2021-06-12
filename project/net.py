import os
import json
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import datasets, transforms, models

class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img = Image.open(self.data_path[idx])
        img = img.convert("RGB")
        img = np.array(img.resize((256, 256)))
        if self.transform:
            img = self.transform(img)

        return img

class Net(nn.Module):
    def __init__(self, out_size):
        super(Net, self).__init__()
        self.out_size = out_size

        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32 * 126 * 126, out_size)

    def forward(self, x):
        # x -> [batch_size, channel, width, height]
        # x -> [128, 3, 256, 256]

        x = self.relu(self.conv1(x))  # x -> [128, 16, 254, 254]
        x = self.relu(self.conv2(x))  # x -> [128, 32, 252, 252]

        x = self.pool(x)             # x -> [128, 32, 126, 126]

        x = torch.flatten(x, 1)      # x -> [128, 32 * 126 * 126]

        x = self.fc1(x)

        return F.log_softmax(x, dim=1)

class NetType(nn.Module):
    def __init__(self, out_size):
        super(NetType, self).__init__()
        self.out_size = out_size

        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.conv3 = nn.Conv2d(32, 48, 5, 1)
        self.conv4 = nn.Conv2d(48, 64, 5, 1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # 50.176 x 128
        self.fc2 = nn.Linear(128, out_size)  # 128 x 14
        # self.fc3 = nn.Linear(1024, out_size)

    def forward(self, x):
        # x -> [batch_size, channel, width, height]
        # x -> [128, 3, 256, 256]

        x = self.relu(self.conv1(x))  # x -> [128, 16, 254, 254]
        x = self.relu(self.conv2(x))  # x -> [128, 32, 250, 250]
        x = self.pool(x)             # x -> [128, 32, 125, 125]
        x = self.relu(self.conv3(x))  # x -> [128, 48, 121, 121]
        x = self.pool(x)             # x -> [128, 48,  60,  60]
        x = self.relu(self.conv4(x))  # x -> [128, 64,  56,  56]
        x = self.pool(x)             # x -> [128, 64,  28,  28]

        x = torch.flatten(x, 1)      # x -> [128, 64 * 28 * 28]

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        # x = self.relu(self.fc2(x))
        x = self.fc2(x)

        return x

leaf_types_map = {
    'Apple': 0,
    'Blueberry': 1,
    'Cherry_(including_sour)': 2,
    'Corn_(maize)': 3,
    'Grape': 4,
    'Orange': 5,
    'Peach': 6,
    'Pepper,_bell': 7,
    'Potato': 8,
    'Raspberry': 9,
    'Soybean': 10,
    'Squash': 11,
    'Strawberry': 12,
    'Tomato': 13
}
leaf_types_map_inv = {v: k for k, v in leaf_types_map.items()}

device = torch.device("cpu")

def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device, dtype=torch.float)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            return output, pred

model = Net(2).to(device)
model.load_state_dict(torch.load("project/static/plant_model.pth", map_location=torch.device("cpu")))

modelType = NetType(len(leaf_types_map)).to(device)
modelType.load_state_dict(torch.load("project/static/plant_type_model.pth", map_location=torch.device("cpu")))
