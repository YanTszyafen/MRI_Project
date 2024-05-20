import os
import glob
from PIL import Image
import matplotlib.pyplot as plt


class MRIDatalist:
    def __init__(self, train_dir:str, test_dir:str):
        self.train_dir = train_dir
        self.test_dir = test_dir

    def get_data_list(self):

        labels = os.listdir(self.train_dir)
        print("labels: " + str(labels) + '\n')

        print("Getting data list...")
        train_list = []
        train_labels_list = []
        test_list = []
        test_labels_list = []
        for i in labels:
            train_path = os.path.join(self.train_dir, i)
            temp1 = glob.glob(os.path.join(train_path, '*.jpg'))
            train_list += temp1
            temp2 = [i for j in range(len(temp1))]
            train_labels_list += temp2
            test_path = os.path.join(self.test_dir, i)
            temp3 = glob.glob(os.path.join(test_path, '*.jpg'))
            test_list += temp3
            temp4 = [i for j in range(len(temp3))]
            test_labels_list += temp4

        print("Summary of data list:")
        print(f"Train Data: {len(train_list)}" + " images")
        print(f"Train Labels: {len(train_labels_list)}" + " images")
        print(f"Test Data: {len(test_list)}" + " images")
        print(f"Test Labels: {len(test_labels_list)}" + " images" + '\n')

        return train_list, train_labels_list, test_list, test_labels_list

    def plot_image(self, datalist, labellist,idx):
        img = Image.open(datalist[idx])
        plt.plot(img)
        plt.title(labellist[idx])
