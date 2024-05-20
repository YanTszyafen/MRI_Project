import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from plotting import get_confusion_matrix, plot_accuracies, plot_losses,plot_confusion_matrix


def train(model_, device, lr, gamma, epochs, train_loader, valid_loader):
    train_loss = []
    train_acc = []
    val_loss_list = []
    val_acc_list = []
    outputs_train = []
    outputs_val = []
    targets_train = []
    targets_val = []
    # loss function
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer
    optimizer = optim.Adam(model_.parameters(), lr=lr)
    # for param in model_.parameters():
    #     param.requires_grad = False  # 先将所有参数设为不可训练
    #
    #     # 最后几层需要训练
    # for param in model_.layers[-3:].parameters():
    #     param.requires_grad = True
    #
    #     # 设置优化器和学习率调度器
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_.parameters()), lr=lr)
    # scheduler
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model_(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

            outputs_train.extend(item.cpu().detach().numpy() for item in output.argmax(1))
            targets_train.extend(item.cpu().detach().numpy() for item in label)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_accuracy)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model_(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

                outputs_val.extend(item.cpu().detach().numpy() for item in val_output.argmax(1))
                targets_val.extend(item.cpu().detach().numpy() for item in label)

            val_loss_list.append(epoch_val_loss)
            val_acc_list.append(epoch_val_accuracy)
        print(
            f"Epoch : {epoch +1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        torch.cuda.empty_cache()

    ######################################################################################################################################
    print("Epoch:{}".format(epoch+1))
    print("Loss value in the last epoch of training is:{}".format(epoch_loss))
    print("Accuracy in the last epoch of training is:{}".format(epoch_accuracy))

    sklearn_accuracy1 = accuracy_score(targets_train, outputs_train)
    sklearn_precision1 = precision_score(targets_train, outputs_train, average='macro')
    sklearn_recall1 = recall_score(targets_train, outputs_train, average='macro')
    sklearn_f11 = f1_score(targets_train, outputs_train, average='macro')
    print("Accuracy of training:{}".format(sklearn_accuracy1))
    print("Precision of training:{}".format(sklearn_precision1))
    print("Recall of training:{}".format(sklearn_recall1))
    print("Macro f1 score of training:{}".format(sklearn_f11))

    sklearn_report1 = classification_report(targets_train, outputs_train, digits=5)
    print("Classification report of training:")
    print(sklearn_report1)

    conf_matrix1 = get_confusion_matrix(targets_train, outputs_train)
    print("Confusion matrix of training:")
    print(conf_matrix1)
    plot_confusion_matrix(conf_matrix1 ,epoch)

    #####################################################################################################################################
    print("Loss value in the last epoch of validation is:{}".format(epoch_val_loss))
    print("Accuracy in the last epoch of validation is:{}".format(epoch_val_accuracy))

    sklearn_accuracy2 = accuracy_score(targets_val, outputs_val)
    sklearn_precision2 = precision_score(targets_val, outputs_val, average='macro')
    sklearn_recall2 = recall_score(targets_val, outputs_val, average='macro')
    sklearn_f12 = f1_score(targets_val, outputs_val, average='macro')
    print("Accuracy of validation:{}".format(sklearn_accuracy2))
    print("Precision of validation:{}".format(sklearn_precision2))
    print("Recall of validation:{}".format(sklearn_recall2))
    print("Macro f1 score of validation:{}".format(sklearn_f12))

    sklearn_report2 = classification_report(targets_val, outputs_val, digits=5)
    print("Classification report of validation:")
    print(sklearn_report2)

    conf_matrix2 = get_confusion_matrix(targets_val, outputs_val)
    print("Confusion matrix of validation:")
    print(conf_matrix2)
    plot_confusion_matrix(conf_matrix2, epoch)

    plot_accuracies(train_acc, val_acc_list)
    plot_losses(train_loss, val_loss_list)


def train2(model_, device, criterion, optimizer, scheduler, epochs, train_loader, valid_loader):
    train_loss = []
    train_acc = []
    val_loss_list = []
    val_acc_list = []
    outputs_train = []
    outputs_val = []
    targets_train = []
    targets_val = []
    # loss function
    criterion = criterion
    # optimizer
    optimizer = optimizer
    # scheduler
    scheduler = scheduler
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model_(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

            outputs_train.extend(item.cpu().detach().numpy() for item in output.argmax(1))
            targets_train.extend(item.cpu().detach().numpy() for item in label)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_accuracy)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model_(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

                outputs_val.extend(item.cpu().detach().numpy() for item in val_output.argmax(1))
                targets_val.extend(item.cpu().detach().numpy() for item in label)

            val_loss_list.append(epoch_val_loss)
            val_acc_list.append(epoch_val_accuracy)
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

    ######################################################################################################################################
    print("Epoch:{}".format(epoch+1))
    print("Loss value in the last epoch of training is:{}".format(epoch_loss))
    print("Accuracy in the last epoch of training is:{}".format(epoch_accuracy))

    sklearn_accuracy1 = accuracy_score(targets_train, outputs_train)
    sklearn_precision1 = precision_score(targets_train, outputs_train, average='macro')
    sklearn_recall1 = recall_score(targets_train, outputs_train, average='macro')
    sklearn_f11 = f1_score(targets_train, outputs_train, average='macro')
    print("Accuracy of training:{}".format(sklearn_accuracy1))
    print("Precision of training:{}".format(sklearn_precision1))
    print("Recall of training:{}".format(sklearn_recall1))
    print("Micro f1 score of training:{}".format(sklearn_f11))

    sklearn_report1 = classification_report(targets_train, outputs_train, digits=5)
    print("Classification report of training:")
    print(sklearn_report1)

    conf_matrix1 = get_confusion_matrix(targets_train, outputs_train)
    print("Confusion matrix of training:")
    print(conf_matrix1)
    plot_confusion_matrix(conf_matrix1,epoch)

    #####################################################################################################################################
    print("Loss value in the last epoch of validation is:{}".format(epoch_val_loss))
    print("Accuracy in the last epoch of validation is:{}".format(epoch_val_accuracy))

    sklearn_accuracy2 = accuracy_score(targets_val, outputs_val)
    sklearn_precision2 = precision_score(targets_val, outputs_val, average='macro')
    sklearn_recall2 = recall_score(targets_val, outputs_val, average='macro')
    sklearn_f12 = f1_score(targets_val, outputs_val, average='macro')
    print("Accuracy of validation:{}".format(sklearn_accuracy2))
    print("Precision of validation:{}".format(sklearn_precision2))
    print("Recall of validation:{}".format(sklearn_recall2))
    print("Micro f1 score of validation:{}".format(sklearn_f12))

    sklearn_report2 = classification_report(targets_val, outputs_val, digits=5)
    print("Classification report of validation:")
    print(sklearn_report2)

    conf_matrix2 = get_confusion_matrix(targets_val, outputs_val)
    print("Confusion matrix of validation:")
    print(conf_matrix2)
    plot_confusion_matrix(conf_matrix2, epoch)


def train_vd(v, distiller, device, lr, epochs, train_loader, valid_loader):
    train_loss = []
    train_acc = []
    val_loss_list = []
    val_acc_list = []
    outputs_train = []
    outputs_val = []
    targets_train = []
    targets_val = []
    # loss function
    # criterion = nn.CrossEntropyLoss().to(device)
    # optimizer
    optimizer = optim.Adam(v.parameters(), lr=lr)
    # scheduler
    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = v(data)
            loss = distiller(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

            outputs_train.extend(item.cpu().detach().numpy() for item in output.argmax(1))
            targets_train.extend(item.cpu().detach().numpy() for item in label)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_accuracy)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = v(data)
                val_loss = distiller(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

                outputs_val.extend(item.cpu().detach().numpy() for item in val_output.argmax(1))
                targets_val.extend(item.cpu().detach().numpy() for item in label)

            val_loss_list.append(epoch_val_loss)
            val_acc_list.append(epoch_val_accuracy)
        print(
            f"Epoch : {epoch +1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

    ######################################################################################################################################
    print("Epoch:{}".format(epoch+1))
    print("Loss value in the last epoch of training is:{}".format(epoch_loss))
    print("Accuracy in the last epoch of training is:{}".format(epoch_accuracy))

    sklearn_accuracy1 = accuracy_score(targets_train, outputs_train)
    sklearn_precision1 = precision_score(targets_train, outputs_train, average='macro')
    sklearn_recall1 = recall_score(targets_train, outputs_train, average='macro')
    sklearn_f11 = f1_score(targets_train, outputs_train, average='macro')
    print("Accuracy of training:{}".format(sklearn_accuracy1))
    print("Precision of training:{}".format(sklearn_precision1))
    print("Recall of training:{}".format(sklearn_recall1))
    print("Micro f1 score of training:{}".format(sklearn_f11))

    sklearn_report1 = classification_report(targets_train, outputs_train, digits=4)
    print("Classification report of training:")
    print(sklearn_report1)

    conf_matrix1 = get_confusion_matrix(targets_train, outputs_train)
    print("Confusion matrix of training:")
    print(conf_matrix1)
    plot_confusion_matrix(conf_matrix1 ,epoch)

    #####################################################################################################################################
    print("Loss value in the last epoch of validation is:{}".format(epoch_val_loss))
    print("Accuracy in the last epoch of validation is:{}".format(epoch_val_accuracy))

    sklearn_accuracy2 = accuracy_score(targets_val, outputs_val)
    sklearn_precision2 = precision_score(targets_val, outputs_val, average='macro')
    sklearn_recall2 = recall_score(targets_val, outputs_val, average='macro')
    sklearn_f12 = f1_score(targets_val, outputs_val, average='macro')
    print("Accuracy of validation:{}".format(sklearn_accuracy2))
    print("Precision of validation:{}".format(sklearn_precision2))
    print("Recall of validation:{}".format(sklearn_recall2))
    print("Micro f1 score of validation:{}".format(sklearn_f12))

    sklearn_report2 = classification_report(targets_val, outputs_val, digits=4)
    print("Classification report of validation:")
    print(sklearn_report2)

    conf_matrix2 = get_confusion_matrix(targets_val, outputs_val)
    print("Confusion matrix of validation:")
    print(conf_matrix2)
    plot_confusion_matrix(conf_matrix2, epoch)

    plot_accuracies(train_acc, val_acc_list)
    plot_losses(train_loss, val_loss_list)