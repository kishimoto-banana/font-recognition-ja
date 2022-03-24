import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from imutils import paths
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FontDataset
from model import DeepFont, ModelName, fetch_efficientnet_b3, fetch_vgg16
from pytorchtools import EarlyStopping
from transform import ImageTranform

load_dotenv()

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

font_labels = {
    "Stick-Regular": 0,
    "DelaGothicOne-Regular": 1,
    "ShipporiMincho-Regular": 2,
    "ReggaeOne-Regular": 3,
    "MPLUS1[wght]": 4,
    "TrainOne-Regular": 5,
    "KleeOne-Regular": 6,
    "OtomanopeeOne-Regular": 7,
    "Yomogi-Regular": 8,
    "MPLUS1Code[wght]": 9,
    "NewTegomin-Regular": 10,
    "SawarabiMincho-Regular": 11,
    "RocknRollOne-Regular": 12,
    "HinaMincho-Regular": 13,
    "YuseiMagic-Regular": 14,
    "KosugiMaru-Regular": 15,
    "SawarabiGothic-Regular": 16,
    "KaiseiTokumin-Regular": 17,
    "KaiseiDecol-Regular": 18,
    "HachiMaruPop-Regular": 19,
    "Kosugi-Regular": 20,
    "MPLUS2[wght]": 21,
    "KiwiMaru-Regular": 22,
    "DotGothic16-Regular": 23,
    "RampartOne-Regular": 24,
    "PottaOne-Regular": 25,
    "KaiseiOpti-Regular": 26,
    "KaiseiHarunoUmi-Regular": 27,
}

font_image_paths = list(paths.list_images(os.environ["IMAGES_PATH"]))
train_image_paths, test_image_paths = train_test_split(
    font_image_paths, test_size=0.25, random_state=5
)
batch_size = 256
num_epochs = 100
model_name = ModelName.efficientnetB3


def train_model(
    net: nn.Module,
    dataladers: Dict[str, DataLoader],
    criterion: nn.CrossEntropyLoss,
    optimizer: Optimizer,
    num_epochs: int,
    early_stopping: EarlyStopping,
) -> None:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    net.to(device)

    for epoch in range(num_epochs):
        print(f"{epoch+1} / {num_epochs}")
        print("----------")

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0
            epoch_corr = 0

            if epoch == 0 and phase == "train":
                continue

            for inputs, labels in tqdm(dataladers[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corr += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataladers[phase].dataset)
            epoch_acc = epoch_corr.double() / len(dataladers[phase].dataset)

            print(f"{[phase]} loss: {epoch_loss} acc: {epoch_acc}")

            if phase == "val":
                early_stopping(epoch_loss, net)
                if early_stopping.early_stop:
                    print("early_stop")
                    break


def main() -> None:
    transform = ImageTranform()

    print("make dataset")
    train_dataset = FontDataset(
        image_paths=train_image_paths,
        transform=transform,
        font_labels=font_labels,
        phase="train",
    )
    val_dataset = FontDataset(
        image_paths=test_image_paths,
        transform=transform,
        font_labels=font_labels,
        phase="val",
    )

    print("make dataloader")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    if model_name == ModelName.DeepFont:
        print("model: DeepFont")
        net = DeepFont()
        checkpoint_path = "deepfont_checkpoint.pt"
    elif model_name == ModelName.VGG16:
        print("model: VGG16")
        net = fetch_vgg16()
        checkpoint_path = "vgg16_checkpoint.pt"
    elif model_name == ModelName.efficientnetB3:
        print("model: efficientnetB3")
        net = fetch_efficientnet_b3()
        checkpoint_path = "efficientnetb3_checkpoint.pt"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005
    )

    print(f"training")

    early_stopping = EarlyStopping(patience=3, verbose=True, path=checkpoint_path)
    train_model(net, dataloaders, criterion, optimizer, num_epochs, early_stopping)


if __name__ == "__main__":
    main()
