import matplotlib.pyplot as plt
from mridatalist import MRIDatalist
from preprocess import preprocessing,seed_everything
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np


# 设置训练和测试数据集路径
train_dir = "./Training"
test_dir = "./Testing"

# 创建MRIDatalist实例
data = MRIDatalist(train_dir, test_dir)
# 预处理数据
batch_size = 32
seed = 326
# 获取数据列表
train_list, train_labels_list, test_list, test_labels_list = data.get_data_list()
seed_everything(seed)
train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=train_labels_list, random_state=seed)

train_loader= preprocessing(train_list, batch_size)



# # 可视化预处理后的图像
# def visualize_images(data_loader):
#     images, labels = next(iter(data_loader))
#     num_images = len(images)
#
#     fig, axs = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
#
#     for idx, ax in enumerate(axs.flat):
#         ax.imshow(images[idx].permute(1, 2, 0))
#         ax.set_title(f"Label: {labels[idx].item()}")
#         ax.axis('off')
#
#     plt.tight_layout()
#     plt.show()

# def visualize_images(data_loader, num_images=6):
#     images, labels = next(iter(data_loader))
#     images = images[:num_images]
#     labels = labels[:num_images]
#
#     fig, axs = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
#
#     for idx, ax in enumerate(axs.flat):
#         ax.imshow(images[idx].permute(1, 2, 0))
#         ax.set_title(f"Label: {labels[idx].item()}")
#         ax.axis('off')
#
#         # Add image size text
#         width, height = images[idx].size()[1], images[idx].size()[2]
#         ax.text(0, -20, f"Size: {width}x{height}", color='black', fontsize=10, ha='left',
#                 bbox=dict(facecolor='white', alpha=0.5))
#
#
#     plt.tight_layout()
#     plt.show()

def visualize_images(data_loader, num_images=6):
    images, labels = next(iter(data_loader))
    images = images[:num_images]
    labels = labels[:num_images]

    fig, axs = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))

    for idx, ax in enumerate(axs.flat):
        img = images[idx].permute(1, 2, 0)
        ax.imshow(img)

        # Add coordinate axes
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)

        # Show pixel values on axes
        ax.set_xticks(np.arange(0, img.shape[1], 50))
        ax.set_xticklabels(np.arange(0, img.shape[1], 50))
        ax.set_yticks(np.arange(0, img.shape[0], 50))
        ax.set_yticklabels(np.arange(0, img.shape[0], 50))

    plt.tight_layout()
    plt.show()

# def visualize_images2(image_paths):
#     fig, axs = plt.subplots(1, 6, figsize=(18, 4))
#
#     for i, image_path in enumerate(image_paths[:6]):
#         row = i // 6
#         col = i % 6
#         img = Image.open(image_path)
#         axs[row, col].imshow(img)
#         axs[row, col].axis('off')
#
#         width, height = img.size
#         axs[row, col].text(0, -20, f"Size: {width}x{height}", color='white', fontsize=10,
#                            bbox=dict(facecolor='black', alpha=0.5))
#
#
#     plt.show()


from PIL import Image

def visualize_images2(image_paths):
    fig, axs = plt.subplots(2, 6, figsize=(18, 4))

    for i, image_path in enumerate(image_paths[:12]):
        row = i // 6
        col = i % 6
        img = Image.open(image_path)
        axs[row, col].imshow(img, cmap='gray')
        axs[row, col].axis('on')  # 显示坐标轴

        width, height = img.size
        axs[row, col].set_xlabel('Width')
        axs[row, col].set_ylabel('Height')
        axs[row, col].set_xticks(np.arange(0, width, 100))
        axs[row, col].set_yticks(np.arange(0, height, 100))
        axs[row, col].set_xticklabels(np.arange(0, width, 100))
        axs[row, col].set_yticklabels(np.arange(0, height, 100))
        axs[row, col].tick_params(axis='both', which='both', length=0)  # 隐藏刻度线

        for spine in axs[row, col].spines.values():  # 隐藏边框
            spine.set_visible(False)

    plt.tight_layout()
    plt.show()


# 可视化训练数据
# print("Visualizing training data:")
# visualize_images(train_loader)

print("Visualizing training data:")
visualize_images2(train_list)

# # 可视化验证数据
# print("Visualizing validation data:")
# visualize_images(valid_loader)
#
# # 可视化测试数据
# print("Visualizing test data:")
# visualize_images(test_loader)