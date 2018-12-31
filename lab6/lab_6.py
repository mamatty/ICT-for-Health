import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def optimazed_matrix_PCA(U, Lambda, percentage):

    tot = 0
    pcr = []

    for l in Lambda.argsort()[::-1]:
        tot += Lambda[l]
        pcr.append(l)
        if tot >= Lambda.sum() * percentage:
            break

    UF = U[:, pcr]
    return UF

def diag_covariance_matrix(y):

    cov = np.cov(y.T)
    Lambda, U = np.linalg.eig(cov)
    diag_lambda = np.diag(Lambda)
    R = np.dot(np.dot(U, diag_lambda), U.T)
    return R, Lambda, U

def PDF(s, R, w, F):

    # Define the PDF how is reported into the slides
    a = 1 / (np.sqrt(np.dot(np.power(2 * np.pi, F), np.linalg.det(R))))
    exponential = -1/2 * np.dot(np.dot((s - w).T, np.linalg.inv(R)), (s - w))
    pdf = a * np.exp(exponential)
    return pdf

# This method has been copied from the dedicated web page scikit-learn for confusion matrix plotting.
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == "__main__":
    # Read the data and set NaN where there are ?
    data = pd.read_csv("arrhythmia.data", sep=',', header=None, na_values=['?', '\t?'])

    # Remove the columns with these NaN values
    data_clear = data.dropna(axis=1)

    # Override data_clear removing the columns whose values are all equal to zero
    data_clear = data_clear.loc[:, (data_clear != 0).any(axis=0)]

    # Create a copy of the original dataframe
    df1 = data_clear.copy()

    # %% Binary
    # Set elements of last column equal to 2 if their values are higher than 1
    last_column = df1.iloc[:, -1]
    last_column_correct = last_column.map(lambda x: 2 if (x >1) else 1)
    df1.iloc[:, -1] = last_column_correct

    # Define y and class_id
    y = df1.iloc[:, :-1]
    class_id = df1.iloc[:, -1]

    class_id_counts = class_id.value_counts().tolist()
    n_healthy = class_id_counts[0]
    n_ill = class_id_counts[1]

    # Number of patients and features
    N = y.shape[0]
    F = y.shape[1]

    # Define y1,y2,x1 and x2 as described in the slides
    y1 = y.loc[(class_id == 1),:]
    y2 = y.loc[(class_id == 2),:]
    x1 = y1.mean(axis=0)
    x2 = y2.mean(axis=0)

    # Minimum distance criterion as described in the slides
    # Define a vector of zeros equal to the number of patients that I have (rows of the matrix) -> I will fill it with in the iteration
    est_class_id = np.zeros(N)

    for i in range(N):
        if np.linalg.norm(y.loc[i, :] - x1) ** 2 < np.linalg.norm(y.loc[i, :] - x2) ** 2:
            est_class_id[i] = 1
        else:
            est_class_id[i] = 2

    # Evaluate probabilities of true/false positives & negatives
    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0

    for i in range(N):
        if (est_class_id[i] == class_id[i]):
            if (est_class_id[i] == 1):
                true_negative += 1
            else:
                true_positive += 1
        else:
            if (est_class_id[i] == 1):
                false_negative += 1
            else:
                false_positive += 1

    tn_normalized = 100 * true_negative / n_healthy
    tp_normalized = 100 * true_positive / n_ill
    fn_normalized = 100 * false_negative / n_ill
    fp_normalized = 100 * false_positive / n_healthy

    plt.figure(figsize=(12, 8))
    plt.figure(1)
    plt.title('Classification Results: Minimum Distance criterion', fontsize=15)
    plt.bar(1, tp_normalized, align='center', label="True Positive", color='g')
    plt.bar(2, tn_normalized, align='center', label="True Negative", color='r')
    plt.bar(3, fp_normalized, align='center', label="False Positive", color='m')
    plt.bar(4, fn_normalized, align='center', label="False Negative", color='y')
    plt.legend()
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.grid(axis='y')

    plt.show()

    print('Specificity: ', float(true_negative) / (true_negative + false_positive))
    print('Sensibility: ', float(true_positive) / (true_positive + false_negative))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(class_id, est_class_id)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(1, 3), normalize=True,
                          title='Normalized confusion matrix - Binary Class')

    plt.show()

    # Plot not normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(1, 3), normalize=False,
                          title='Not normalized confusion matrix - Binary Class')

    plt.show()

    # %%Multiple Classes (16-classes)

    means = []
    for i in range(1, 17):
        y_i = data_clear.loc[data_clear[279] == i]
        y_i = y_i.drop(columns=279)
        if i in [11, 12, 13]:
            means.append(np.zeros(257))
        else:
            means.append(y_i.mean(axis=0))

    # Create a copy of the original dataframe
    df2 = data_clear.copy()
    target = df2.iloc[:, -1].values
    datas = df2.iloc[:, :-1]

    est_multiple_class_id = []

    for i in datas.iterrows():
        i = i[1]
        distances = []
        for j in range(len(means)):
            distances.append(np.power(np.linalg.norm(i - means[j]), 2))
        est_multiple_class_id.append(np.argmin(distances) + 1)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(df2.iloc[:, -1].values, est_multiple_class_id)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(9, 9))
    plot_confusion_matrix(cnf_matrix, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16], normalize=True,
                              title='Normalized confusion matrix - Multiple Classes')

    plt.show()

    # %% Bayes Criterion
    index_perc = 0.999
    pi1 = n_healthy / N
    pi2 = n_ill / N

    R1, lambda1, U1 = diag_covariance_matrix(y1)
    R2, lambda2, U2 = diag_covariance_matrix(y2)

    UF1 = optimazed_matrix_PCA(U1, lambda1, index_perc)
    UF2 = optimazed_matrix_PCA(U2, lambda2, index_perc)

    z1 = y1.dot(UF1)
    z2 = y2.dot(UF2)

    w1 = z1.mean(axis=0)
    w2 = z2.mean(axis=0)

    s1 = y.dot(UF1)
    s2 = y.dot(UF2)

    R1_red = np.cov(z1.T)
    R2_red = np.cov(z2.T)

    # Same procedures of minimum distance criterion applied to the Bayes Theorem
    est_class_id_bayes = np.zeros(N)

    for i in range(N):
        if pi1 * PDF(s1.loc[i, :], R1_red, w1, s1.shape[1]) > pi2 * PDF(s2.loc[i, :], R2_red, w2, s2.shape[1]):
            est_class_id_bayes[i] = 1
        else:
            est_class_id_bayes[i] = 2

    # evaluate probabilities of true/false positives & negatives
    true_negative_bayes = 0
    true_positive_bayes = 0
    false_negative_bayes = 0
    false_positive_bayes = 0

    for i in range(N):
        if (est_class_id_bayes[i] == class_id[i]):
            if (est_class_id_bayes[i] == 1):
                true_negative_bayes += 1
            else:
                true_positive_bayes += 1
        else:
            if (est_class_id_bayes[i] == 1):
                false_negative_bayes += 1
            else:
                false_positive_bayes += 1

    tn_normalized_bayes = 100 * true_negative_bayes / n_healthy
    tp_normalized_bayes = 100 * true_positive_bayes / n_ill
    fn_normalized_bayes = 100 * false_negative_bayes / n_ill
    fp_normalized_bayes = 100 * false_positive_bayes / n_healthy

    plt.figure(figsize=(12, 8))
    plt.title('Classification Results: Bayes criterion', fontsize=15)
    plt.bar(1, tp_normalized_bayes, align='center', label="True Positive", color='g')
    plt.bar(2, tn_normalized_bayes, align='center', label="True Negative", color='r')
    plt.bar(3, fp_normalized_bayes, align='center', label="False Positive", color='m')
    plt.bar(4, fn_normalized_bayes, align='center', label="False Negative", color='y')
    plt.legend()
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.grid(axis='y')

    plt.show()

    print('Specificity: ', float(true_negative) / (true_negative + false_positive))
    print('Sensibility: ', float(true_positive) / (true_positive + false_negative))

    # Compute confusion matrix
    cnf_matrix_bayes = confusion_matrix(class_id, est_class_id_bayes)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_bayes, classes=range(1, 3), normalize=True,
                          title='Normalized confusion matrix - Bayes approach')

    plt.show()

    # Plot not normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_bayes, classes=range(1, 3), normalize=False,
                          title='Not normalized confusion matrix - Bayes approach')

    plt.show()