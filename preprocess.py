import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from mridataset import MRIDataset
from torch.utils.data import DataLoader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def image_augmentation():
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    return train_transforms, val_transforms, test_transforms


def preprocessing(seed, train_list, train_labels_list, test_list, batch_size):
    print("Preprocessing:")
    seed_everything(seed)
    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=train_labels_list, random_state=seed)

    print("Splitting dataset...")
    print("Summary of dataset:")
    print(f"Train Data: {len(train_list)}" + " images")
    print(f"Validation Data: {len(valid_list)}" + " images")
    print(f"Test Data: {len(test_list)}" + " images" + '\n')

    print("Image Augmentation...")
    train_transforms, val_transforms, test_transforms = image_augmentation()

    train_data = MRIDataset(train_list, transform=train_transforms)
    valid_data = MRIDataset(valid_list, transform=val_transforms)
    test_data = MRIDataset(test_list, transform=test_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    print("Final Summary of Data:")
    print(f"Size of Train Data: {len(train_data)}")
    print(f"Size of Train Data Loader: {len(train_loader)}")
    print(f"Size of Validation Data: {len(valid_data)}")
    print(f"Size of Validation Data Loader: {len(valid_loader)}")
    print(f"Size of Test Data: {len(test_data)}")
    print(f"Size of Test Data Loader: {len(test_loader)}" + '\n')

    return train_loader, valid_loader, test_loader


# def preprocessing(train_list, batch_size):
#
#
#     print("Image Augmentation...")
#     train_transforms, val_transforms, test_transforms = image_augmentation()
#
#     train_data = MRIDataset(train_list, transform=train_transforms)
#
#     train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
#
#
#     print("Final Summary of Data:")
#     print(f"Size of Train Data: {len(train_data)}")
#     print(f"Size of Train Data Loader: {len(train_loader)}")
#
#
#     return train_loader
