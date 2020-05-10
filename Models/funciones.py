import numpy as np
def sumar():
    return 2+2


def classification_error(y_est, y_real):
    err = 0
    for y_e, y_r in zip(y_est, y_real):

        if y_e != y_r:
            err += 1

    return err/np.size(y_est)

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def grafica_roc(model,Xtest,Ytest):
    y_predict_proba = model.predict_proba(Xtest)
    # Compute ROC curve and ROC AUC for each class
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
        y_test_i = list(map(lambda x: 1 if x == i else 0, Ytest))
        all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
        all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
        fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = auc(fpr["average"], tpr["average"])

#     # Plot average ROC Curve
    plt.figure()
#     plt.plot(fpr["average"], tpr["average"],
#              label='Average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["average"]),
#              color='deeppink', linestyle=':', linewidth=4)

    # Plot each individual ROC curve
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio de falsos positivos')
    plt.ylabel('Ratio de verdaderos positivos')
    plt.title('Gráfica roc multiclase')
    plt.legend(loc="lower right")
    plt.show()

def grafica_roc1(model,Xtest,Ytest):
    y_predict_proba = model.predict_proba(Xtest)
    # Compute ROC curve and ROC AUC for each class
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
        y_test_i = list(map(lambda x: 1 if x == i else 0, Ytest))
        all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
        all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
        fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = auc(fpr["average"], tpr["average"])

#     # Plot average ROC Curve
    plt.figure()
#     plt.plot(fpr["average"], tpr["average"],
#              label='Average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["average"]),
#              color='deeppink', linestyle=':', linewidth=4)

    # Plot each individual ROC curve
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()