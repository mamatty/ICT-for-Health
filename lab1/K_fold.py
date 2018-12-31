import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold
from lab1.Lab1_health import Regression


class K_fold(Regression):

    def __init__(self, path, column, graph, splitting):
        Regression.__init__(self, path, column, graph, splitting)

    def K_fold(self):

        # reading the csv file
        data = Regression.Create_Dataset(self)
        num_patients = data["subject#"].max()

        # kf = KFold(n_splits=5, shuffle=True)
        # test_sets = []
        # for train, test in kf.split(range(num_patients)):
        #     test_sets.append(test)

        test_sets = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24, 25, 26]
                     , [27, 28, 29, 30, 31, 32, 33, 34], [35, 36, 37, 38, 39, 40, 41, 42]]
        results_tr = []
        results_te = []

        for i in range(len(test_sets)):

            data_train_norm, data_test_norm = Regression.split(self, data, pref_subsets_test=test_sets[i])

            # column vectors normalized; firstly I start regressing jitter%, column 7 (6 in Python)
            y_train_norm = data_train_norm[FO].values
            y_test_norm = data_test_norm[FO].values

            # matrices normalized obtained removing the column above
            X_train_norm = data_train_norm.drop(FO, axis=1).values
            X_test_norm = data_test_norm.drop(FO, axis=1).values

            # applying the different algorithm
            np.random.seed(1000)
            weights_sd = np.random.rand(X_train_norm.shape[1])
            weights_gr = np.random.rand(X_train_norm.shape[1])

            print('Iteration {}:'.format(int(i)+1))

            se_MSE_tr, se_MSE_te = Regression.MSE(self, y_train_norm, y_test_norm, X_train_norm, X_test_norm)
            se_gr_tr,  se_gr_te = Regression.Gradient(self, weights_gr, 1e-4, 1.0e-4, y_train_norm, y_test_norm, X_train_norm, X_test_norm)
            se_sd_tr, se_sd_te = Regression.Steepest_Descent(self, weights_sd, 1.0e-5, y_train_norm, y_test_norm, X_train_norm, X_test_norm)
            se_ridge_tr, se_ridge_te = Regression.Ridge_regression(self, 800, y_train_norm, y_test_norm, X_train_norm, X_test_norm)
            se_PCR_tr, se_PCR_te = Regression.PCR(self, X_train_norm, X_test_norm, y_train_norm, y_test_norm)
            se_PCR_red_tr, se_PCR_red_te = Regression.PCR_reduced(self, X_train_norm, X_test_norm, y_train_norm, y_test_norm)

            #store all the results in a list
            results_tr.append([se_MSE_tr, se_gr_tr, se_sd_tr, se_ridge_tr, se_PCR_tr, se_PCR_red_tr])
            results_te.append([se_MSE_te, se_gr_te, se_sd_te, se_ridge_te, se_PCR_te, se_PCR_red_te])

        return results_tr, results_te

if __name__ == "__main__":
    PATH = 'C:\\Users\\matte\\Documents\\ICT for health\\Labs\\lab1\\data.csv'
    FO = 'total_UPDRS'
    #it's better to leave graph parameter to False;
    #if you want to plot also the total_UPDRS, make sure to have enough memory RAM, at least 2000 MB
    k_fold = K_fold(PATH, column=FO, graph=False, splitting=False)

    np.random.seed(1000)
    results_tr, results_te = k_fold.K_fold()

    se_MSE_tr, se_gr_tr, se_sd_tr, se_ridge_tr, se_PCR_tr, se_PCR_red_tr = [*zip(*results_tr)]
    se_MSE_te, se_gr_te, se_sd_te, se_ridge_te, se_PCR_te, se_PCR_red_te = [*zip(*results_te)]

    x_pos = np.arange(1,6)

    se_PCR_red_tr_90 = []
    se_PCR_red_tr_99 = []

    se_PCR_red_te_90 = []
    se_PCR_red_te_99 = []

    for x in se_PCR_red_tr:
        se_PCR_red_tr_90.append(x[0])
        se_PCR_red_tr_99.append(x[1])

    for y in se_PCR_red_te:
        se_PCR_red_te_90.append(y[0])
        se_PCR_red_te_99.append(y[1])

    #train
    plt.figure(1)
    plt.title('MSE train', fontsize=15)
    plt.bar(x_pos, se_MSE_tr, align='center')
    plt.grid(axis='y')
    plt.figure(2)
    plt.title('GRADIENT train', fontsize=15)
    plt.bar(x_pos, se_gr_tr, align='center')
    plt.grid(axis='y')
    plt.figure(3)
    plt.title('STEEPEST DESCENT train', fontsize=15)
    plt.bar(x_pos, se_sd_tr, align='center')
    plt.grid(axis='y')
    plt.figure(4)
    plt.title('RIDGE train', fontsize=15)
    plt.bar(x_pos, se_ridge_tr, align='center')
    plt.grid(axis='y')
    plt.figure(5)
    plt.title('PCR train', fontsize=15)
    plt.bar(x_pos, se_PCR_tr, align='center')
    plt.grid(axis='y')
    plt.figure(6)
    plt.title('PCR train - 90%', fontsize=15)
    plt.bar(x_pos, se_PCR_red_tr_90, align='center')
    plt.grid(axis='y')
    plt.figure(7)
    plt.title('PCR train - 99%', fontsize=15)
    plt.bar(x_pos, se_PCR_red_tr_99, align='center')
    plt.grid(axis='y')

    #test
    plt.figure(8)
    plt.title('MSE test', fontsize=15)
    plt.bar(x_pos, se_MSE_te, align='center')
    plt.grid(axis='y')
    plt.figure(9)
    plt.title('GRADIENT test', fontsize=15)
    plt.bar(x_pos, se_gr_te, align='center')
    plt.grid(axis='y')
    plt.figure(10)
    plt.title('STEEPEST DESCENT test', fontsize=15)
    plt.bar(x_pos, se_sd_te, align='center')
    plt.grid(axis='y')
    plt.figure(11)
    plt.title('RIDGE test', fontsize=15)
    plt.bar(x_pos, se_ridge_te, align='center')
    plt.grid(axis='y')
    plt.figure(12)
    plt.title('PCR test', fontsize=15)
    plt.bar(x_pos, se_PCR_te, align='center')
    plt.grid(axis='y')
    plt.figure(13)
    plt.title('PCR test - 90%', fontsize=15)
    plt.bar(x_pos, se_PCR_red_te_90, align='center')
    plt.grid(axis='y')
    plt.figure(14)
    plt.title('PCR test - 99%', fontsize=15)
    plt.bar(x_pos, se_PCR_red_te_99, align='center')
    plt.grid(axis='y')

    #prova multiple charts

    plt.figure(15)
    plt.figure(figsize=(12, 8))
    bar_width = 0.10
    plt.bar(x_pos, se_MSE_tr, width=bar_width, color='green', zorder=2)
    plt.bar(x_pos+bar_width, se_gr_tr, width=bar_width, color='red', zorder=2)
    plt.bar(x_pos+bar_width*2, se_sd_tr, width=bar_width, color='orange', zorder=2)
    plt.bar(x_pos+bar_width*3, se_ridge_tr, width=bar_width, color='purple', zorder=2)
    plt.bar(x_pos+bar_width*4, se_PCR_tr, width=bar_width, color='blue', zorder=2)
    plt.bar(x_pos+bar_width*5, se_PCR_red_tr_90, width=bar_width, color='black', zorder=2)
    plt.bar(x_pos + bar_width*6, se_PCR_red_tr_99, width=bar_width, color='yellow', zorder=2)

    plt.xticks(x_pos+bar_width*2, ['1', '2', '3', '4', '5'])
    plt.title('Train error comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Squared Errors')

    green_patch = mpatches.Patch(color='green', label='MSE')
    red_patch = mpatches.Patch(color='red', label='GRADIENT')
    orange_patch = mpatches.Patch(color='orange', label='STEEPEST DESCENT')
    purple_patch = mpatches.Patch(color='purple', label='RIDGE')
    blue_patch = mpatches.Patch(color='blue', label='PCR')
    black_patch = mpatches.Patch(color='black', label='PCR - 90%')
    yellow_patch = mpatches.Patch(color='yellow', label='PCR - 99%')
    plt.legend(handles=[green_patch, red_patch, orange_patch, purple_patch, blue_patch, black_patch, yellow_patch])

    plt.grid(axis='y')
    plt.savefig('total_UPDRS/weight_comparison_train.png')

    plt.figure(16)
    plt.figure(figsize=(12, 8))
    bar_width = 0.10
    plt.bar(x_pos, se_MSE_te, width=bar_width, color='green', zorder=2)
    plt.bar(x_pos + bar_width, se_gr_te, width=bar_width, color='red', zorder=2)
    plt.bar(x_pos + bar_width * 2, se_sd_te, width=bar_width, color='orange', zorder=2)
    plt.bar(x_pos + bar_width * 3, se_ridge_te, width=bar_width, color='purple', zorder=2)
    plt.bar(x_pos + bar_width * 4, se_PCR_te, width=bar_width, color='blue', zorder=2)
    plt.bar(x_pos + bar_width * 5, se_PCR_red_te_90, width=bar_width, color='black', zorder=2)
    plt.bar(x_pos + bar_width * 6, se_PCR_red_te_99, width=bar_width, color='yellow', zorder=2)

    plt.xticks(x_pos + bar_width * 2, ['1', '2', '3', '4', '5'])
    plt.title('Test error comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Squared Errors')

    green_patch = mpatches.Patch(color='green', label='MSE')
    red_patch = mpatches.Patch(color='red', label='GRADIENT')
    orange_patch = mpatches.Patch(color='orange', label='STEEPEST DESCENT')
    purple_patch = mpatches.Patch(color='purple', label='RIDGE')
    blue_patch = mpatches.Patch(color='blue', label='PCR')
    black_patch = mpatches.Patch(color='black', label='PCR - 90%')
    yellow_patch = mpatches.Patch(color='yellow', label='PCR - 99%')
    plt.legend(handles=[green_patch, red_patch, orange_patch, purple_patch, blue_patch, black_patch, yellow_patch])

    plt.grid(axis='y')
    plt.savefig('total_UPDRS/weight_comparison_test.png')

    plt.show()