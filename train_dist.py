import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torchvision import datasets
from timm.models.deit import deit_small_distilled_patch16_224
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
from ema import EMA

torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
from ema import EMA

def seed_everything(seed=42):
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        samples, targets = mixup_fn(data, target)
        output = model(samples)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = criterion_train(output, targets)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
        else:
            loss = criterion_train(output, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
        if use_ema:
            ema.update()
        torch.cuda.synchronize()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    ave_loss = loss_meter.avg
    acc = acc1_meter.avg
    print('epoch:{}\tloss:{:.2f}\tacc:{:.2f}'.format(epoch, ave_loss, acc))
    return ave_loss, acc


# 验证过程
@torch.no_grad()
def val(model, device, test_loader):
    global Best_ACC
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    val_list = []
    pred_list = []
    if use_ema:
        ema.apply_shadow()
    for data, target in test_loader:
        for t in target:
            val_list.append(t.data.item())
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg, acc, acc5_meter.avg))
    if acc > Best_ACC:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module, file_dir + '/' + 'best.pth')
        else:
            torch.save(model, file_dir + '/' + 'best.pth')
        Best_ACC = acc
    if isinstance(model, torch.nn.DataParallel):
        state = {

            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'Best_ACC': Best_ACC
        }
        torch.save(state, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
    else:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'Best_ACC': Best_ACC
        }
        torch.save(state, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
    if use_ema:
        ema.restore()
    return val_list, pred_list, loss_meter.avg, acc


if __name__ == '__main__':
    #创建保存模型的文件夹
    file_dir = 'checkpoints/DEiT_dist'
    if os.path.exists(file_dir):
        print('true')

        os.makedirs(file_dir,exist_ok=True)
    else:
        os.makedirs(file_dir)

    # 设置全局参数
    model_lr = 7e-5
    BATCH_SIZE = 32
    EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = True  # 是否使用混合精度
    use_dp = False #是否开启dp方式的多卡训练
    classes = 4
    resume = False #是否继续训练
    CLIP_GRAD = 5.0 #梯度下降的一个参数
    model_path = 'best.pth' #当resume为True，指向预训练模型路径
    Best_ACC = 0 #记录最高得分
    use_ema = True #一般是有预训练模型时使用ema，没有预训练模型使用ema可能不会上分，这时可以将ema设置为False训练一段时间，接着设置resume为True继续训练
    SEED=326
    seed_everything(SEED)

    # 数据预处理7
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), #明暗亮度饱和度
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18541224, 0.18541516, 0.18545938], std=[0.18545432, 0.18545458, 0.18547246])

    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18541224, 0.18541516, 0.18545938], std=[0.18545432, 0.18545458, 0.18547246])
    ])
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.1, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=classes)


    # 读取数据
    dataset_train = datasets.ImageFolder('./Training', transform=transform)
    dataset_test = datasets.ImageFolder("./Testing", transform=transform_test)
    with open('class.txt', 'w') as file:
        file.write(str(dataset_train.class_to_idx))
    with open('class.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(dataset_train.class_to_idx))
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True) #视频中加了drop_last=True
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)


    # 实例化模型并且移动到GPU
    criterion_train = SoftTargetCrossEntropy()
    criterion_val = torch.nn.CrossEntropyLoss()
    # 设置模型
    model_ft = deit_small_distilled_patch16_224(pretrained=True)
    model_ft.reset_classifier(classes)
    # num_ftrs = model_ft.head.in_features
    # model_ft.head = nn.Linear(num_ftrs, classes)
    if resume:
        model = torch.load(model_path)
        model_ft.load_state_dict(model['state_dict'])
        Best_ACC = model['Best_ACC']
        start_epoch = model['epoch'] + 1
    model_ft.to(DEVICE)
    print(model_ft)
    # 选择简单暴力的Adam优化器，学习率调低
    optimizer = optim.AdamW(model_ft.parameters(), lr=model_lr)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6) #学习率调整策略选择为余弦退火
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    if torch.cuda.device_count() > 1 and use_dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = torch.nn.DataParallel(model_ft)
    if use_ema:
        ema = EMA(model_ft, 0.9998)
        ema.register()


     # 训练与验证
    is_set_lr = False
    log_dir = {}
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    for epoch in range(1, EPOCHS + 1):
        epoch_list.append(epoch)
        train_loss, train_acc = train(model_ft, DEVICE, train_loader, optimizer, epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list
        val_list, pred_list, val_loss, val_acc = val(model_ft, DEVICE, test_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = Best_ACC
        with open(file_dir + '/result.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(log_dir))
        print(classification_report(val_list, pred_list, target_names=dataset_train.class_to_idx, digits=4))
        if epoch < 600:
            cosine_schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-6
                    is_set_lr = True
        fig1 = plt.figure()
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # 显示图例
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(file_dir + "/loss.png")
        plt.close(fig1)
        fig2 = plt.figure()
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(file_dir + "/acc.png")
        plt.close(fig2)









