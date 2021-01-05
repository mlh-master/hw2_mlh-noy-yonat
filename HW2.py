import numpy as np
import pickle
import sys
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
mpl.style.use(['ggplot'])
# %matplotlib inline
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from turtledemo import forest
from sklearn.metrics import plot_confusion_matrix

from Hw2_Functions import test_train_table
from Hw2_Functions import data_hist
from Hw2_Functions import plt_2d_pca
from Hw2_Functions import statistics_calculate


# ----------------------------------------------------------------------------------------------------------------------
# MAIN CODE:
# Load the Data and Drop NULL:
df_org = pd.read_csv('HW2_data.csv')
# preprocessing
tmpData = df_org
tmpData = tmpData.dropna()  # removing nan rows
T1D_dataset_clean = tmpData

# Change VAlues to Binaries Values:
T1D_dataset_clean_bin = pd.DataFrame()
T1D_dataset_clean_bin = pd.get_dummies(data=T1D_dataset_clean, drop_first=True)


# Create Y and Drop Diagnosis from X:
X = T1D_dataset_clean_bin.drop('Diagnosis_Positive', axis=1)
# X = X.drop('Age', axis=1)
y = T1D_dataset_clean_bin['Diagnosis_Positive']
# Create Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=10, stratify=y)

# Comparisment between X_test and X_train- Creating Table 1
x_table_compare = test_train_table(X_train, X_test)
print(x_table_compare)


# Plots that describe the data:
# Preparation: change 'Family History' to string:
tmpData_str = tmpData.replace({'Family History': 0}, 'No')
tmpData_str = tmpData_str.replace({'Family History': 1}, 'Yes')
# Creating Figure 1- showing the relationship between features and labels
Hist = data_hist(tmpData_str)

diagnosicData = T1D_dataset_clean.iloc[:, 16]
diagnosicData.value_counts().plot(kind="pie", colors=['steelblue', 'salmon'], autopct='%1.1f%%')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Q4:
# encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
# hot_vector = encoder.fit_transform(X)
# print(hot_vector)


# ----------------------------------------------------------------------------------------------------------------------
# Q5.a:
# Split the training set into Train and Validation:
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
C = np.array([0.001, 0.01, 1, 10, 100, 1000])
max_iter = 2000

# Linear model:
# Logistic Regression:
pen = ['l1', 'l2']
solver = 'saga'

from sklearn.model_selection import GridSearchCV
log_reg = LogisticRegression(random_state=5, max_iter=max_iter, solver=solver)
C = np.array([0.01, 0.01, 1, 10, 100, 1000])
pipe = Pipeline(steps=[('scale', StandardScaler()), ('logistic', log_reg)])
lin_log_reg = GridSearchCV(estimator=pipe, param_grid={'logistic__C': 1/C, 'logistic__penalty': pen},
                           scoring=['roc_auc'], cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
lin_log_reg.fit(X_train, Y_train)
best_lin_log_reg = lin_log_reg.best_estimator_

#Y:
best_lin_log_reg.fit(X_train, Y_train)

y_pred_proba_train_lin=best_lin_log_reg.predict_proba(X_train)
y_pred_train_lin=best_lin_log_reg.predict(X_train)
y_pred_proba_test_lin = best_lin_log_reg.predict_proba(X_test)
y_pred_test_lin = best_lin_log_reg.predict(X_test)


# Non-Linear model:
# SVM:
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
svc = SVC(probability=True)
C2 = np.array([1, 100, 1000])#, 10, 100, 1000])

pipe = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
svm_nonlin = GridSearchCV(estimator=pipe, param_grid={'svm__C': C2, 'svm__kernel': ['rbf', 'poly'], 'svm__gamma': ['auto', 'scale']},
                          scoring=['roc_auc'], cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_nonlin.fit(X_train, Y_train)
best_svm_nonlin = svm_nonlin.best_estimator_

#Y
best_svm_nonlin.fit(X_train, Y_train)

y_pred_proba_train_nonlin = best_svm_nonlin.predict_proba(X_train)
y_pred_train_nonlin = best_svm_nonlin.predict(X_train)
y_pred_proba_test_nonlin = best_svm_nonlin.predict_proba(X_test)
y_pred_test_nonlin = best_svm_nonlin.predict(X_test)

#Print Best Parameters
print('Best parameters for linear model logistic regression:')
print(lin_log_reg.best_params_)
print('Best parameters for non-linear model SVM:')
print(svm_nonlin.best_params_)


# ----------------------------------------------------------------------------------------------------------------------
# Q5.b:

# Linear Classifier Parameters:
# loss:
loss_lin_test = log_loss(Y_test, y_pred_proba_test_lin)
loss_lin_train = log_loss(Y_train, y_pred_proba_train_lin)


# Train
[Acc, F1, Auc] = statistics_calculate(Y_train, y_pred_train_lin, y_pred_proba_train_lin)
print("For linear model the evaluation metrics are:")
print(f'train AUROC is {Auc:.3f}')
print(f'train Accuracy is {Acc:.3f}')
print(f'train F1 is {F1:.3f}')
print(f'train Loss is {loss_lin_train:.3f}')


# Test
[Acc, F1, Auc] = statistics_calculate(Y_test, y_pred_test_lin, y_pred_proba_test_lin)
print(f'test AUROC is {Auc:.3f}')
print(f'test Accuracy is {Acc:.3f}')
print(f'test F1 is {F1:.3f}')
print(f'test Loss is {loss_lin_test:.3f}')

print("confusion matrix on linear test set:")
plot_confusion_matrix(best_lin_log_reg, X_test, Y_test, cmap=plt.cm.Blues)
plt.show()

# NonLinear Classifier Parameters:
# loss:
from sklearn.metrics import hinge_loss
#best_svm_nonlin.fit(X_train,Y_train)

#Y for loss calculate:
y_pred_train_nonlin_loss = best_svm_nonlin.decision_function(X_train)
y_pred_test_nonlin_loss = best_svm_nonlin.decision_function(X_test)

# loss:
loss_nonlin_train = hinge_loss(Y_train, y_pred_train_nonlin_loss)
loss_nonlin_test = hinge_loss(Y_test, y_pred_test_nonlin_loss)

# Train
[Acc, F1, Auc] = statistics_calculate(Y_train, y_pred_train_nonlin, y_pred_proba_train_nonlin)
print("For non-linear model the evaluation metrics are:")
print(f'train AUROC is {Auc:.3f}')
print(f'train Accuracy is {Acc:.3f}')
print(f'train F1 is {F1:.3f}')
print(f'train Loss is {loss_nonlin_train:.3f}')


# Test
[Acc, F1, Auc] = statistics_calculate(Y_test, y_pred_test_nonlin, y_pred_proba_test_nonlin)
print(f'test AUROC is {Auc:.3f}')
print(f'test Accuracy is {Acc:.3f}')
print(f'test F1 is {F1:.3f}')
print(f'test Loss is {loss_nonlin_test:.3f}')

print("confusion matrix on non-linear test set:")
plot_confusion_matrix(best_svm_nonlin, X_test, Y_test, cmap=plt.cm.Blues)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Q6: Random Forest

indicesNames = []

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib

# make a list of features names
names = list(X_train.columns)
# Random Forest
rd_clf = rfc(n_estimators=10)
rd_clf.fit(X_train, Y_train)

# arrange the features
importances_tmp = rd_clf.feature_importances_
indices = np.argsort(importances_tmp)[::-1]  # arrange the features

# arrange thr features names
for index in indices:
    indicesNames.append(names[index])

# figure:
plt.figure(figsize=(14, 10))
matplotlib.rcParams.update({'font.size': 8})

importances = pd.DataFrame({'importances_grade': rd_clf.feature_importances_})
ax =importances.plot(kind='bar')
ax.set_title('feature importances')
ax.set_xlabel('feature')
ax.set_ylabel('importances')
ax.set_xticklabels(names)
plt.show()

print(f'2 most important features according to random forest are: {names[indices[0]]} and {names[indices[1]]}')
# ----------------------------------------------------------------------------------------------------------------------


#------------------------- C7 ------------------------------#
#C7.a
# scaling the data and fitting 
from sklearn.decomposition import PCA
n_components = 2 
pca=PCA(n_components=n_components, whiten = True)
scaler= StandardScaler()
X_train_org= X_train 
X_test_org= X_test 
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# apply PCA transformation
X_train_pca = pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

#plotting 2D
plt_2d_pca(X_test_pca,Y_test)


#C7.c
#Training the models
#------- Linear-------#
pipe_pca = Pipeline(steps=[('scale', StandardScaler()),('pca', pca), ('logistic', best_lin_log_reg)])
pipe_pca.fit(X_train, Y_train)
print('The score on the test set with PCA preprocessing is {:.2f}'.format(pipe_pca.score(X_test,Y_test)))

y_pred_lin_pca=pipe_pca.predict(X_test)
y_pred_proba_pca=pipe_pca.predict_proba(X_test)

[Acc, F1, Auc] = statistics_calculate(Y_test ,y_pred_lin_pca,y_pred_proba_pca)
print(f'test AUROC for LR is {Auc:.3f}')
print(f'test Accuracy for LR is {Acc:.3f}')
print(f'test F1 for LR is {F1:.3f}')

print("confusion matrix on linear PCA test set:")
plot_confusion_matrix(pipe_pca, X_test, Y_test, cmap=plt.cm.Blues)
plt.show()
#------- SVM-------#

pipe_pca_svm = Pipeline(steps=[('scale', StandardScaler()),('pca', pca), ('svm', best_svm_nonlin)])
pipe_pca_svm.fit(X_train, Y_train)
print('The score on the test set with PCA preprocessing is {:.2f}'.format(pipe_pca_svm.score(X_test,Y_test)))

y_pred_svm_pca=pipe_pca_svm.predict(X_test)
y_pred_svm_proba_pca= pipe_pca_svm.predict_proba(X_test)
[Acc, F1, Auc] = statistics_calculate(Y_test ,y_pred_svm_pca,y_pred_svm_proba_pca)
print(f'test AUROC is {Auc:.3f}')
print(f'test Accuracy is {Acc:.3f}')
print(f'test F1 is {F1:.3f}')

print("confusion matrix on non-linear PCA SVM test set:")
plot_confusion_matrix(pipe_pca_svm, X_test, Y_test, cmap=plt.cm.Blues)
plt.show()

#C7.d
# Now we train the models on the best two features from section 6
#------- Linear-------#

best_cols= ['Increased Thirst_Yes', 'Increased Urination_Yes']
X_train_two_feat = X_train_org[best_cols]
X_test_two_feat = X_test_org[best_cols]

X_train_two_feat=scaler.fit_transform(X_train_two_feat)
X_test_two_feat=scaler.transform(X_test_two_feat)

best_lin_log_reg.fit(X_train_two_feat, Y_train)
y_pred_best_feat_lin=best_lin_log_reg.predict(X_test_two_feat)
y_pred_best_feat_lin_proba= best_lin_log_reg.predict_proba(X_test_two_feat)
[Acc, F1, Auc] = statistics_calculate(Y_test ,y_pred_best_feat_lin,y_pred_best_feat_lin_proba)
print(f'test AUROC for LR- 2 features is {Auc:.3f}')
print(f'test Accuracy for LR- 2 features is {Acc:.3f}')
print(f'test F1 for LR- 2 features is {F1:.3f}')

print("confusion matrix on linear 2 features test set:")
plot_confusion_matrix(best_lin_log_reg, X_test_two_feat, Y_test, cmap=plt.cm.Blues)
plt.show()

#------- SVM-------#
best_svm_nonlin.fit(X_train_two_feat,Y_train)
y_pred_best_feat_svm=best_svm_nonlin.predict(X_test_two_feat)
y_pred_best_feat_svm_proba= best_svm_nonlin.predict_proba(X_test_two_feat)
[Acc, F1, Auc] = statistics_calculate(Y_test ,y_pred_best_feat_svm,y_pred_best_feat_svm_proba)
print(f'test AUROC for SVM- 2 features is: {Auc:.3f}')
print(f'test Accuracy for SVM- 2 features is {Acc:.3f}')
print(f'test F1 for SVM- 2 features is {F1:.3f}')

print("confusion matrix on non-linear SVM 2 features test set:")
plot_confusion_matrix(best_svm_nonlin, X_test_two_feat, Y_test, cmap=plt.cm.Blues)
plt.show()

