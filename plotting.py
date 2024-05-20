from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def get_confusion_matrix(targets, outputs):
    labels = [0, 1, 2, 3]
    conf_matrix = confusion_matrix(y_true=targets, y_pred=outputs, labels=labels)
    return conf_matrix


def plot_confusion_matrix(conf_matrix, i):
    plt.imshow(X=conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = [0, 1, 2, 3]
    plt.xticks(ticks=indices, labels=labels)
    plt.yticks(ticks=indices, labels=labels)
    plt.colorbar()
    plt.xlabel('outputs')
    plt.ylabel('targets')
    plt.title('Confusion matrix'.format(i))
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(x=first_index, y=second_index, s=conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()


def plot_accuracies(train_acc, val_acc):
    accuracies = [acc.item() for acc in train_acc]
    val_accuracies = [acc.item() for acc in val_acc]
    plt.plot(accuracies, '-bx')
    plt.plot(val_accuracies, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig('accuracies.jpg')
    plt.show()


def plot_losses(train_loss, val_loss):
    train_losses = [loss.item() for loss in train_loss]
    val_losses = [loss.item() for loss in val_loss]
    plt.plot(train_losses, '-x')
    plt.plot(val_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig('losses.jpg')
    plt.show()

