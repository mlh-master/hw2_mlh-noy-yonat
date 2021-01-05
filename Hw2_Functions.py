# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def test_train_table(X_train, x_test):
    data_dic = {}
    for col in X_train:
        if col != 'Age':
            temp_dict = {'train %': round(100 * X_train.loc[:, col].sum() / len(X_train.loc[:, col])),
                         'test %': round(100 * x_test.loc[:, col].sum() / len(x_test.loc[:, col]))}
            data_dic[col] = temp_dict
    return pd.DataFrame.from_dict(data_dic, orient='index')


def data_hist(df):
    Diagnosis = df.loc[:, 'Diagnosis']
    for col in df:
        if col != 'Diagnosis':
            plt.title(col)
            x1 = df.loc[Diagnosis == 'Positive', col]
            x2 = df.loc[Diagnosis == 'Negative', col]
            plt.hist([x1, x2], color=['blue', 'orange'], bins=4)

            plt.xlabel(col)
            plt.ylabel('count')
            plt.legend(['Positive', 'Negative'])
            plt.show()

    return None


# statistics_calculate
# Input- y and y_pred
# Output-  list of [Acc, F1, Auc]
def statistics_calculate(y, ypred, ypred_proba):
    from sklearn.metrics import confusion_matrix
    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    from sklearn.metrics import plot_confusion_matrix, roc_auc_score
    TN = calc_TN(y, ypred)
    TP = calc_TP(y, ypred)
    FN = calc_FN(y, ypred)
    FP = calc_FP(y, ypred)
    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)

    Acc = (TP + TN) / (TP + TN + FP + FN)
    F1 = (2 * PPV * Se) / (PPV + Se)
    Auc = (roc_auc_score(y, ypred_proba[:, 1]))

    return [Acc, F1, Auc]




def plt_2d_pca(X_pca,y):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='b')
    ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='r')
    ax.legend(('B','M'))
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    ax.set_title('2D PCA')