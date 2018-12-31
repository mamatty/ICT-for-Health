import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from matplotlib import pyplot as plt

if __name__ == "__main__":

    # Read the data and set NaN where there are ?
    data = pd.read_csv("arrhythmia.data", sep=',', header=None, na_values=['?', '\t?'])

    # Remove the columns with these NaN values
    data_clear = data.dropna(axis=1)

    # Override data_clear removing the columns whose values are all equal to zero
    data_clear = data_clear.loc[:, (data_clear != 0).any(axis=0)]

    # Set elements of last column equal to 0 if their values are higher than 1
    last_column = data_clear.iloc[:, -1]
    last_column_correct = last_column.map(lambda x: 1 if (x > 1) else 0)
    data_clear.iloc[:, -1] = last_column_correct

    # Define y and class_id
    x = data_clear.iloc[:, :-1]
    class_id = data_clear.iloc[:, -1]

    Npatients, Nfeatures = data.shape
    Ntrain = Npatients / 2

    # Define the train and test subsets
    x_train = x.loc[:Ntrain - 1, :]
    x_test = x.loc[Ntrain:, :]
    y_train = class_id.loc[:Ntrain - 1]
    y_test = class_id.loc[Ntrain:]

    # Create empty arrays that will be filled with the value of Specificity and Sensibility for the train and test phase
    spec_train = []
    sens_train = []
    spec_test = []
    sens_test = []

    for c in [0.1, 0.01, 0.001, 0.0001]:

        print(c)
        # PCA
        pca = PCA(n_components=88)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        # Defining svm
        clf = SVC(C=c, kernel="linear")

        # Train phase
        clf.fit(x_train, y_train)

        # Test phase
        class_id_train_hat = clf.predict(x_train)
        class_id_test_hat = clf.predict(x_test)

        # %% Statistics
        true_positive_train = 0
        false_positive_train = 0
        true_negative_train = 0
        false_negative_train = 0

        for i in range(int(Ntrain)):
            if y_train[i] == 1:
                if class_id_train_hat[i] == 1:
                    true_negative_train += 1
                else:
                    false_negative_train += 1
            else:
                if class_id_train_hat[i] == 0:
                    true_positive_train += 1
                else:
                    false_positive_train += 1

        sensitivity_train = float(true_positive_train) / (true_positive_train + false_negative_train)
        specificity_train = float(true_negative_train) / (true_negative_train + false_positive_train)
        sens_train.append(sensitivity_train)
        spec_train.append(specificity_train)

        print("Train")
        print("Sensitivity:", sensitivity_train)
        print("Specificity:", specificity_train)

        true_positive_test = 0
        false_positive_test = 0
        true_negative_test = 0
        false_negative_test = 0

        for i in range(int(Npatients - Ntrain)):
            if y_test.iloc[i] == 1:
                if class_id_test_hat[i] == 1:
                    true_negative_test += 1
                else:
                    false_negative_test += 1
            else:
                if class_id_test_hat[i] == 0:
                    true_positive_test += 1
                else:
                    false_positive_test += 1

        sensitivity_test = float(true_positive_test) / (true_positive_test + false_negative_test)
        specificity_test = float(true_negative_test) / (true_negative_test + false_positive_test)

        sens_test.append(sensitivity_test)
        spec_test.append(specificity_test)

        print("Test")
        print("Sensitivity:", sensitivity_test)
        print("Specificity:", specificity_test)
        print("-------------------------------------")

    # %%plotting
    plt.figure(figsize=(12, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.stem([0.1, 0.01, 0.001, 0.0001], spec_test)
    ax2.stem([0.1, 0.01, 0.001, 0.0001], sens_test)
    ax1.set_title('Sensitivity', fontsize=12)
    ax2.set_title('Sensitivity', fontsize=12)
    ax1.set_xlabel('Relaxation Parameter', fontsize=12)
    ax2.set_xlabel('Relaxation Parameter', fontsize=12)
    ax1.set_ylabel('Sensitivity', fontsize=12)
    ax2.set_ylabel('Specificity', fontsize=12)

    plt.show()