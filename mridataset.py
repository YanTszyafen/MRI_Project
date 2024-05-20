from torch.utils.data import Dataset
from PIL import Image

class MRIDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)

        label = img_path.split("\\")[1]
        if label == "glioma":
            label = 0
        elif label =="meningioma":
            label = 1
        elif label == "notumor":
            label = 2
        elif label == "pituitary":
            label = 3

        return img_transformed, label

