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
    "Kosugi-Regular": 39,
    "KosugiMaru-Regular": 40,
    "MPLUS1[wght]": 41,
    "MPLUS2[wght]": 42,
    "MPLUS1Code[wght]": 43,
    "ZenAntiqueSoft-Regular": 44,
    "Murecho[wght]": 45,
    "MochiyPopOne-Regular": 46,
    "YujiSyuku-Regular": 47,
    "YujiBoku-Regular": 48,
    "YujiMai-Regular": 49,
    "ZenKakuGothicNew-Bold": 50,
    "ZenKakuGothicNew-Light": 51,
    "ZenKakuGothicNew-Medium": 52,
    "ZenKakuGothicNew-Black": 53,
    "ZenKakuGothicNew-Regular": 54,
    "ZenMaruGothic-Bold": 55,
    "ZenMaruGothic-Light": 56,
    "ZenMaruGothic-Medium": 57,
    "ZenMaruGothic-Black": 58,
    "ZenMaruGothic-Regular": 59,
    "ZenKakuGothicAntique-Light": 60,
    "ZenKakuGothicAntique-Bold": 61,
    "ZenKakuGothicAntique-Medium": 62,
    "ZenKakuGothicAntique-Regular": 63,
    "ZenKakuGothicAntique-Black": 64,
    "ZenOldMincho-SemiBold": 65,
    "ZenOldMincho-Black": 66,
    "ZenOldMincho-Bold": 67,
    "ZenOldMincho-Regular": 68,
    "ZenOldMincho-Medium": 69,
    "ZenAntique-Regular": 70,
    "ZenKurenaido-Regular": 71,
    "ShipporiAntique-Regular": 72,
    "BIZUDGothic-Bold": 73,
    "BIZUDGothic-Regular": 74,
    "BIZUDMincho-Regular": 75,
    "BIZUDPMincho-Regular": 76,
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
