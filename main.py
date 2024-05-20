from mridatalist import MRIDatalist
from preprocess import preprocessing
import torch
import torch.nn as nn
from training import train
from swin_transformer import SwinTransformer, BasicLayer
from Simple_ViT import SimpleViT
from ViT import ViT
from vit_pytorch import ViT, efficient
from vit_pytorch.cait import CaiT
from linformer import Linformer
from nystrom_attention import NystromAttention,Nystromformer
from sinkhorn_transformer import SinkhornTransformer, SinkhornTransformerLM, Autopadder
from vit_pytorch.deepvit import DeepViT
from testing import testing
from VGG import Vgg16_net
from timm.models.deit import deit_small_distilled_patch16_224

print(f"Torch: {torch.__version__}")
print(f"Cuda is available: {torch.cuda.is_available()}")
device = 'cuda'

seed = 326
batch_size = 32
lr = 7e-5
gamma = 0.7
epochs = 100

train_dir = './Training'
test_dir = './Testing'

mri_datalist_ = MRIDatalist(train_dir, test_dir)
train_list, train_labels_list, test_list, test_labels_list = mri_datalist_.get_data_list()
train_loader, valid_loader, test_loader = preprocessing(seed, train_list, train_labels_list, test_list, batch_size)

##############################################################################################################################
# SwinTransformer
# model = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=4,
#              embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
#              window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#              drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#              norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#              use_checkpoint=False, fused_window_process=False).to(device)
#
# pretrained_dict = torch.load('Swin-Transformer/swin_large_patch4_window7_224_22k.pth')
# model_dict = model.state_dict()
#
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
################################################################################################################################

# model = SimpleViT(image_size=224, num_classes=4, patch_size=32, dim=128, channels=3, depth=12, heads=8, mlp_dim=256).to(device)

# model = ViT(image_size=224, patch_size=32, num_classes=4,
#             dim=128, depth=12, heads=8, mlp_dim=256, pool = 'cls', channels = 3,
#             dim_head = 64, dropout = 0., emb_dropout = 0.).to(device)

# Linformer
# efficient_transformer = Linformer(dim=128, seq_len=49+1, depth=12, heads=8, k=64).to(device)
# model = efficient.ViT(dim=128, image_size=224, num_classes=4, patch_size=32, transformer=efficient_transformer, channels=3).to(device)
#
# Nystromformer !pip install nystrom-attention
# efficient_transformer = Nystromformer(dim=128, depth=12, heads=8, num_landmarks=32).to(device)
# model = efficient.ViT(dim=128, image_size=224, num_classes=4, patch_size=32, transformer=efficient_transformer, channels=3).to(device)

# Sinkhorn Transformer !pip install sinkhorn_transformer
# efficient_transformer = SinkhornTransformer(dim=128, heads=8, depth=12, bucket_size=256).to(device)
# model = efficient.ViT(dim = 128, image_size = 224, patch_size = 32, num_classes = 4, transformer = Autopadder(efficient_transformer, pad_left=True)).to(device)

#CaiT
# model = CaiT(image_size=224, patch_size=32, num_classes=4, dim=128, depth=12, heads=16, mlp_dim=256, cls_depth=2, dropout=0.1, emb_dropout=0.1, layer_dropout=0.05).to(device)

#DeepViT
# model = DeepViT(image_size=224, patch_size=32, num_classes=4, dim=128, depth=12, heads=8, mlp_dim=256, dropout=0.1, emb_dropout=0.1).to(device)

# model = Vgg16_net().to(device)

model_ft = deit_small_distilled_patch16_224(pretrained=True)
# model_ft.reset_classifier(classes)
model = torch.load('./DeiT/model_100_99.39.pth')
model_ft.load_state_dict(model['state_dict'])
Best_ACC = model['Best_ACC']
model_ft.to(device)

# train(model, device, lr, gamma, epochs, train_loader, valid_loader)
# torch.save(model.state_dict(), './VGG/VGG2.pth')


testing(model, device, test_loader)
# v, distiller = distill_model(device=device)
# train_vd(v, distiller, device, lr, epochs, train_loader, valid_loader)
