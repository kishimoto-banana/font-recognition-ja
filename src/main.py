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
    "KaiseiOpti-Regular": 0,
    "KaiseiOpti-Bold": 1,
    "KaiseiOpti-Medium": 2,
    "KaiseiDecol-Regular": 3,
    "KaiseiDecol-Medium": 4,
    "KaiseiDecol-Bold": 5,
    "KaiseiHarunoUmi-Regular": 6,
    "KaiseiHarunoUmi-Medium": 7,
    "KaiseiHarunoUmi-Bold": 8,
    "KaiseiTokumin-ExtraBold": 9,
    "KaiseiTokumin-Bold": 10,
    "KaiseiTokumin-Medium": 11,
    "KaiseiTokumin-Regular": 12,
    "HinaMincho-Regular": 13,
    "OtomanopeeOne-Regular": 14,
    "KleeOne-Regular": 15,
    "KleeOne-SemiBold": 16,
    "RampartOne-Regular": 17,
    "ShipporiMincho-Bold": 18,
    "ShipporiMincho-SemiBold": 19,
    "ShipporiMincho-Regular": 20,
    "ShipporiMincho-Medium": 21,
    "ShipporiMincho-ExtraBold": 22,
    "SawarabiGothic-Regular": 23,
    "SawarabiMincho-Regular": 24,
    "NewTegomin-Regular": 25,
    "KiwiMaru-Medium": 26,
    "KiwiMaru-Light": 27,
    "KiwiMaru-Regular": 28,
    "DelaGothicOne-Regular": 29,
    "Yomogi-Regular": 30,
    "HachiMaruPop-Regular": 31,
    "PottaOne-Regular": 32,
    "Stick-Regular": 33,
    "RocknRollOne-Regular": 34,
    "ReggaeOne-Regular": 35,
    "TrainOne-Regular": 36,
    "DotGothic16-Regular": 37,
    "YuseiMagic-Regular": 38,
    "MPLUS1[wght]": 39,
    "MPLUS2[wght]": 40,
    "MPLUS1Code[wght]": 41,
    "ZenAntiqueSoft-Regular": 42,
    "Murecho[wght]": 43,
    "MochiyPopOne-Regular": 44,
    "YujiSyuku-Regular": 45,
    "YujiBoku-Regular": 46,
    "YujiMai-Regular": 47,
    "ZenKakuGothicNew-Bold": 48,
    "ZenKakuGothicNew-Light": 49,
    "ZenKakuGothicNew-Medium": 50,
    "ZenKakuGothicNew-Black": 51,
    "ZenKakuGothicNew-Regular": 52,
    "ZenMaruGothic-Bold": 53,
    "ZenMaruGothic-Light": 54,
    "ZenMaruGothic-Medium": 55,
    "ZenMaruGothic-Black": 56,
    "ZenMaruGothic-Regular": 57,
    "ZenKakuGothicAntique-Light": 58,
    "ZenKakuGothicAntique-Bold": 59,
    "ZenKakuGothicAntique-Medium": 60,
    "ZenKakuGothicAntique-Regular": 61,
    "ZenKakuGothicAntique-Black": 62,
    "ZenOldMincho-SemiBold": 63,
    "ZenOldMincho-Black": 64,
    "ZenOldMincho-Bold": 65,
    "ZenOldMincho-Regular": 66,
    "ZenOldMincho-Medium": 67,
    "ZenAntique-Regular": 68,
    "ZenKurenaido-Regular": 69,
    "ShipporiAntique-Regular": 70,
    "BIZUDGothic-Bold": 71,
    "BIZUDGothic-Regular": 72,
    "BIZUDMincho-Regular": 73,
    "BIZUDPMincho-Regular": 74,
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
