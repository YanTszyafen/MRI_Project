import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from plotting import get_confusion_matrix, plot_accuracies, plot_losses,plot_confusion_matrix
import time

def testing(model2, device, test_loader):
    with torch.no_grad():
        model2.eval()

        y_true_test = []
        y_pred_test = []

        inference_time = 0

        for batch_idx, (img, labels) in enumerate(test_loader):
            img = img.to(device)
            # print('img dimensions:', img.shape[0])

            start_time = time.time()
            preds = model2(img)
            end_time = time.time()
            total_inference_time = end_time - start_time
            # print('Total Inference Time:', total_inference_time)
            inference_time += total_inference_time

            y_pred_test.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_test.extend(labels.detach().tolist())
        average_inference_time = inference_time / len(test_loader.dataset)
        print('Average Inference Time:', average_inference_time)
        total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x == y])
        total = len(y_pred_test)
        accuracy = total_correct * 100 / total
        print("Test Accuracy%: ", accuracy, "==", total_correct, "/", total)

        sklearn_accuracy2 = accuracy_score(y_true_test, y_pred_test)
        sklearn_precision2 = precision_score(y_true_test, y_pred_test, average='macro')
        sklearn_recall2 = recall_score(y_true_test, y_pred_test, average='macro')
        sklearn_f12 = f1_score(y_true_test, y_pred_test, average='macro')
        print("Accuracy of test:{}".format(sklearn_accuracy2))
        print("Precision of test:{}".format(sklearn_precision2))
        print("Recall of test:{}".format(sklearn_recall2))
        print("Macro f1 score of test:{}".format(sklearn_f12))

        sklearn_report2 = classification_report(y_true_test, y_pred_test, digits=5)
        print("Classification report of test:")
        print(sklearn_report2)

        conf_matrix2 = get_confusion_matrix(y_true_test, y_pred_test)
        print("Confusion matrix of test:")
        print(conf_matrix2)

        plot_confusion_matrix(conf_matrix2, 1)





