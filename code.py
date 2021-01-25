import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

df_x = pd.read_csv('x_train.csv')
x_temp = df_x.values
df_y = pd.read_csv('y_train.csv')
y = df_y.values[:, 1].T
y_temp = df_y.values[:, 1].T

cnt = 0
x = []
for i in range(len(y)):
    compute_mean = x_temp[i * 100:(i + 1) * 100, 1:7]
    x.append(np.mean(compute_mean, axis=0))


arr = np.array(x[0:50])
df1 = pd.DataFrame(data=arr, columns=['H3K4me3', 'H3K4me1', 'H3K36me3', 'H3K9me3', 'H3K27me3'])

# Clustered heatmap representation for gene expression data
sns.clustermap(df1, cmap='viridis')
plt.title("Heatmap for gene expression")
plt.savefig("Heatmap for gene expression")
plt.show()

y = np.zeros((10485, 1))

for i in range(len(y_temp)):
    y[i] = y_temp[i]

x_t = np.hstack((x, y))

train_t = x_t[0:10000, :]
test_t = x_t[10000:, :]

#sorting train and test data
train = train_t[train_t[:, -1].argsort(kind='mergesort')]
test = test_t[test_t[:, -1].argsort(kind='mergesort')]

for i in range(len(train)):
    if train[i, -1] == 0:
        train[i, -1] += -1

low_expr = []
high_e = []
for i in range(len(train)):
    if train[i, -1] == -1:
        low_expr.append(train[i, :])
    else:
        high_e.append(train[i, :])

high_expr = []
for i in range(4353):
    high_expr.append(high_e[i])

train = np.vstack((low_expr, high_expr))

X = train[:, 0:-1]
y = train[:, -1]

# Implementation of AdaBoost algorithm

# Initializing uniform weights
w = (1 / len(train)) * np.ones(len(train))

# threshold for weak classifier
calc_thresh = np.mean(X, axis=0)

thresh = calc_thresh


T = 4  # number of iterations
ind = []
err_check = []
thr = []
p = []
alpha = []

# compare every value with threshold and predict low or high gene expression level
pred = np.zeros((len(train), len(thresh)))
for i in range(len(train)):
    for j in range(len(thresh)):
        if X[i][j] <= thresh[j]:
            pred[i][j] = -1
        else:
            pred[i][j] = 1

h_x = np.zeros((len(train), len(thresh)))
# compare predictions with actual output and create a matrix with 1 if misclassified
for i in range(len(train)):
    for j in range(len(thresh)):
        if pred[i][j] == y[i]:
            h_x[i][j] = 0
        else:
            h_x[i][j] = 1

# run adaboost to create strong classifier
for t in range(T):

    for i in range(len(train)):
        pred[i, :] = w[i] * h_x[i, :]

    err = np.sum(pred, axis=0)  # compute error for all the features

    # obtain the value of feature index that  gives least error
    index = np.argmin(err)
    ind.append(index)

    # obtain corresponding column of h which contains 1 for misclassified samples else 0
    h = h_x[:, index]
    # store threshold of corresponding index
    thr.append(thresh[index])

    X_f = [row[index] for row in X]  # corresponding feature values that gave minimum error

    # compute mean for high gene expression level to compare with threshold and update polarity
    X_h = X_f[0:len(high_expr)]
    mean_high = np.mean(X_h)

    # store polarity values. Update polarity as -1 if threshold is greater than mean_high
    if mean_high < thresh[index]:
        p.append(1)
    else:
        p.append(-1)

    # compute epsilon and alpha values
    e = (1 / len(train)) * np.min(err)
    a = 0.5 * np.log((1 - e) / e)
    alpha.append(a)

    # updating weights and normalization
    prod1 = []
    for i in range(len(y)):
        prod1.append(np.multiply(y[i], np.multiply(a, h[i])))

    prod1 = np.asarray(prod1)
    e_term = np.exp(-prod1)

    prod = []
    for i in range(len(y)):
        prod.append(w[i] * e_term[i])

    prod = np.asarray(prod)
    w = prod / np.sum(prod)


# Predict the response for test dataset
for i in range(len(test)):
    if test[i, -1] == 0:
        test[i, -1] += -1

low_e_t = []
high_expr_t = []
for i in range(len(test)):
    if test[i, -1] == -1:
        low_e_t.append(test[i, :])
    else:
        high_expr_t.append(test[i, :])

low_expr_t = []
for i in range(214):
    low_expr_t.append(low_e_t[i])

test = np.vstack((low_expr_t, high_expr_t))

X_t = test[:, 0:-1]
y_t = test[:, -1]

y_final = []
score = []
total = np.sum(alpha)
for i in range(len(test)):
    pre = []
    for j in range(len(alpha)):
        n = X_t[i][ind[j]]
        if p[j] == 1:  # if polarity is 1, corresponding feature value > threshold, classify as high else as low
            if n > thr[j]:
                pre.append(1 * alpha[j])
            else:
                pre.append(-1 * alpha[j])
        else:
            if n > thr[j]:
                pre.append(-1 * alpha[j])
            else:
                pre.append(1 * alpha[j])

    # store the corresponding scores for ROC
    score.append(np.sum(pre))
    # obtaining the final prediction
    yf = np.sign(np.sum(pre))

    y_final.append(yf)

# compute accuracy by comparing predictions with actual class labels
acc = np.zeros(len(y_final))
for i in range(len(y_final)):
    if y_final[i] == y_t[i]:
        acc[i] = 1

# final value of accuracy
accuracy = np.sum(acc) / len(y_t)

# plotting ROC curve
y_actual = np.append(np.zeros(len(low_expr_t)), np.ones(len(high_expr_t)))
fpr, tpr, th = roc_curve(y_actual, score, pos_label=1)
plt.plot(fpr, tpr)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.title("ROC Curve for AdaBoost")
plt.ylabel("True Positives")
plt.xlabel("False Positives")
plt.savefig("ROC_curve_adaboost")
plt.show()
cv2.waitKey(0)
roc_auc_ab = auc(fpr, tpr)

print("Accuracy for AdaBoost:", accuracy * 100, "%")
print("Area under ROC curve for AdaBoost = {:0.4f}".format(roc_auc_ab))

# Implementing Gradient Boosting using sklearn
gbc = GradientBoostingClassifier(n_estimators=50)
model_g = gbc.fit(X, y)
y_pred_g = model_g.predict(X_t)
print("Accuracy of Gradient Boosting:", metrics.accuracy_score(y_t, y_pred_g)*100, "%")

y_scores_gb = gbc.decision_function(X_t)
fpr_gb, tpr_gb, _ = roc_curve(y_t, y_scores_gb)
plt.plot(fpr_gb, tpr_gb)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.title("ROC Curve for Gradient Boosting")
plt.ylabel("True Positives")
plt.xlabel("False Positives")
plt.savefig("ROC_curve_GradientBoost")
plt.show()
cv2.waitKey(0)
roc_auc_gb = auc(fpr_gb, tpr_gb)
print("Area under ROC curve for Gradient Boosting = {:0.4f}".format(roc_auc_gb))