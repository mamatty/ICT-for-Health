import pandas as pd
import numpy as np
from lab1.Graphication import Graphication


class Regression(object):

    def __init__(self, path=None, column=None, graphs=False, splitting=False):

        self.path = path
        if path is not None and column is not None:
            df = pd.read_csv(path)
            self.indices = df.columns.values[4:]
            index = np.argwhere(self.indices == column)
            self.indices = np.delete(self.indices, index)
        self.graphs = graphs
        self.splitting = splitting

        self.error_history_gradient = []
        self.error_history_SD = []
        self.weights = []

    def Create_Dataset(self):

        # reading the csv file
        if self.path is not None:
            x = pd.read_csv(self.path)
        else:
            raise IOError('No CSV path passed!')

        x['test_time'] = x['test_time'].astype(int)

        return x if not self.splitting else self.split(x)

    def split(self, x, pref_subsets_test=None):

        if pref_subsets_test is None:
            # creating the two data set
            limit = 37
            data_tr = x[x['subject#'] < limit]
            data_te = x[x['subject#'] >= limit]

            # removing duplicate in column 4 (test_time) and computing the mean of values from column 7 to 22
            data_train = data_tr.groupby(['subject#', 'test_time'], sort=False, as_index=False).mean()
            data_test = data_te.groupby(['subject#', 'test_time'], sort=False, as_index=False).mean()

            data_train = data_train.drop(['subject#', 'age', 'sex', 'test_time'], axis=1)
            data_test = data_test.drop(['subject#', 'age', 'sex', 'test_time'], axis=1)
        else:
            data_train = x.loc[x['subject#'].isin(pref_subsets_test) == False]
            data_test = x.loc[x['subject#'].isin(pref_subsets_test)]

            data_train = data_train.groupby(['subject#', 'test_time'], sort=False, as_index=False).mean()
            data_test = data_test.groupby(['subject#', 'test_time'], sort=False, as_index=False).mean()

            data_train = data_train.drop(['subject#', 'age', 'sex', 'test_time'], axis=1)
            data_test = data_test.drop(['subject#', 'age', 'sex', 'test_time'], axis=1)

        # creating an output of the obtained file
        data_train.to_csv('data_train.csv', sep=',', encoding='utf-8')
        data_test.to_csv('data_test.csv', sep=',', encoding='utf-8')

        # now let's normalize the two data sets
        df_copy = data_train.copy()
        dt_copy = data_test.copy()
        for feature in data_train.columns:
            df_copy[feature] = (data_train[feature] - data_train[feature].mean()) / np.sqrt(data_train[feature].var())
            dt_copy[feature] = (data_test[feature] - data_test[feature].mean()) / np.sqrt(data_test[feature].var())

        return df_copy, dt_copy

    def MSE(self, y_train_norm, y_test_norm, X_train_norm, X_test_norm):

        # obtaining the pseudo_inverse of the two rectangular matrices
        X_train_plus = np.linalg.pinv(X_train_norm)

        # computing the weight
        weights = np.dot(X_train_plus, y_train_norm)

        # computing the prediction
        y_train_pred = np.dot(X_train_norm, weights)
        y_test_pred = np.dot(X_test_norm, weights)

        # computing mse
        mse_train_norm = y_train_norm - y_train_pred
        mse_test_norm = y_test_norm - y_test_pred

        # squared error
        squared_error_tr = np.linalg.norm(mse_train_norm, ord=2)
        squared_error_te = np.linalg.norm(mse_test_norm, ord=2)

        print('Error Train for MSE: {0:.8f}'.format(squared_error_tr))
        print('Error Test for MSE: {0:.8f}'.format(squared_error_te))
        self.weights.append(weights)

        # print
        if self.graphs:
            Graphication(y_train_norm, y_train_pred, y_test_norm, y_test_pred, mse_train_norm, mse_test_norm,
                         'MSE', self.indices)


        return squared_error_tr, squared_error_te


    def Gradient(self, weights, tollerance, learning_coefficient, y_train_norm, y_test_norm, X_train_norm, X_test_norm):

        weights_t = np.zeros(X_train_norm.shape[1])

        X_train_trans = np.transpose(X_train_norm)

        num_iterations = 0
        max_iterations = 10000

        distance_w_tr = []

        while np.linalg.norm(weights - weights_t) >= tollerance and num_iterations < max_iterations:
            error_train = np.linalg.norm(X_train_norm.dot(weights) - y_train_norm, ord=2)
            difference_tr = np.linalg.norm(weights_t - weights)

            gradient_weights = - 2 * np.dot(X_train_trans, y_train_norm) + 2 * np.dot(X_train_trans, X_train_norm).dot(weights)
            weights_t = weights.copy()
            weights = weights - learning_coefficient * gradient_weights

            self.error_history_gradient.append(error_train)
            distance_w_tr.append(difference_tr)

            num_iterations += 1

        print('GRADIENT: Number of iterations {0} stopped with a weight difference equal to {1}'
              .format(num_iterations, difference_tr))

        # computing the prediction
        y_train_pred = np.dot(X_train_norm, weights)
        y_test_pred = np.dot(X_test_norm, weights)

        # computing difference
        err_train_norm = y_train_norm - y_train_pred
        err_test_norm = y_test_norm - y_test_pred

        # computing error
        squared_error_tr = np.linalg.norm(err_train_norm, ord=2)
        squared_error_te = np.linalg.norm(err_test_norm, ord=2)

        self.weights.append(weights)

        if self.graphs:
           graph = Graphication(y_train_norm, y_train_pred, y_test_norm, y_test_pred, err_train_norm, err_test_norm,
                                 'GRADIENT', self.indices)

           if self.error_history_SD:
               Graphication.grafication(graph, self.error_history_gradient, self.error_history_SD)

        return squared_error_tr, squared_error_te


    def Steepest_Descent(self, weights_sd, tollerance, y_train_norm, y_test_norm, X_train_norm, X_test_norm):

        weights_t = np.zeros((1, X_train_norm.shape[1]))

        X_train_trans = np.transpose(X_train_norm)

        num_iterations = 0
        max_iterations = 10000

        distance_w = []

        while np.linalg.norm(weights_sd - weights_t) >= tollerance and num_iterations < max_iterations:
            error_train = np.linalg.norm(X_train_norm.dot(weights_sd) - y_train_norm, ord=2)
            difference_tr = np.linalg.norm(weights_t - weights_sd)

            gradient_weights = - 2 * np.dot(X_train_trans, y_train_norm) + 2 * np.dot(X_train_trans, X_train_norm).dot(
                weights_sd)
            Hessian = 4 * np.dot(X_train_trans, X_train_norm)
            weights_t = weights_sd.copy()
            gradient_weights_trans = np.transpose(gradient_weights)
            second = np.dot(np.linalg.norm(gradient_weights) ** 2, gradient_weights) / np.dot(gradient_weights_trans,
                                                                                              Hessian).dot(gradient_weights)
            weights_sd = weights_sd - second
            num_iterations += 1

            distance_w.append(difference_tr)
            self.error_history_SD.append(error_train)

        print('STEEPEST DESCENT: Number of iterations {0} stopped with a weight difference equal to {1}'
              .format(num_iterations, difference_tr))

        # computing the prediction
        y_train_pred = np.dot(X_train_norm, weights_sd)
        y_test_pred = np.dot(X_test_norm, weights_sd)

        # computing difference
        err_train_norm = y_train_norm - y_train_pred
        err_test_norm = y_test_norm - y_test_pred

        # computing error
        squared_error_tr = np.linalg.norm(err_train_norm, ord=2)
        squared_error_te = np.linalg.norm(err_test_norm, ord=2)

        self.weights.append(weights_sd)

        if self.graphs:
            graph = Graphication(y_train_norm, y_train_pred, y_test_norm, y_test_pred, err_train_norm, err_test_norm,
                                  'STEEPEST DESCENT', self.indices)

            if self.error_history_gradient:
                Graphication.grafication(graph,self.error_history_gradient, self.error_history_SD)

        return squared_error_tr, squared_error_te


    def Ridge_regression(self, lamb, y_train_norm, y_test_norm, X_train_norm, X_test_norm):

        error_test = []
        error_train = []

        for l in range(lamb):

            # obtaining the pseudo_inverse of the two rectangular matrices
            X_tr = np.dot(X_train_norm.T, X_train_norm)

            first_tr = np.linalg.inv(X_tr + l * np.identity(X_train_norm.shape[1]))
            X_train_ps = np.dot(first_tr, X_train_norm.T)

            # computing the weight
            weights = np.dot(X_train_ps, y_train_norm)

            # computing the prediction
            y_train_pred = np.dot(X_train_norm, weights)

            # computing the error
            err_train_norm_ridge = y_train_norm - y_train_pred

            error_train.append(np.linalg.norm(abs(err_train_norm_ridge)))

        lagrangian = np.amin(error_train)

        print('Ridge Lagrangian: {}'.format(lagrangian))

        w_ridge = (np.linalg.inv(X_train_norm.T.dot(X_train_norm) + lagrangian *
                                     np.identity(X_train_norm.shape[1])).dot(X_train_norm.T)).dot(y_train_norm)

        y_train_ridge_pred = X_train_norm.dot(w_ridge)
        y_test_ridge_pred = X_test_norm.dot(w_ridge)

        #computing the error
        err_train_norm_ridge = y_train_norm - y_train_ridge_pred
        err_test_norm_ridge = y_test_norm - y_test_ridge_pred

        squared_error_tr = np.linalg.norm(err_train_norm_ridge, ord=2)
        squared_error_te = np.linalg.norm(err_test_norm_ridge, ord=2)

        self.weights.append(w_ridge)

        if self.graphs:
            Graphication(y_train_norm, y_train_ridge_pred, y_test_norm, y_test_ridge_pred, err_train_norm_ridge,
                         err_test_norm_ridge, 'RIDGE', self.indices)

        return squared_error_tr, squared_error_te

    def PCR(self, X_train_norm, X_test_norm, y_train_norm, y_test_norm):

        # extimation of the covariance matrix of the row random vector (FxF)
        R_x_train = (1 / X_train_norm.shape[0]) * np.dot(X_train_norm.T, X_train_norm)

        # e-values and e-vectors
        A_tr, U_tr = np.linalg.eig(R_x_train)

        # linear transformation of X (NxF)
        first_tr = np.dot(X_train_norm.T, y_train_norm)
        second_tr = np.dot(np.linalg.inv(np.diag(A_tr)), U_tr.T)
        weights_PCR = (1 / X_train_norm.shape[0]) * (np.dot(U_tr, second_tr)).dot(first_tr)

        # performing the prediction
        y_train_pred = np.dot(X_train_norm, weights_PCR)
        y_test_pred = np.dot(X_test_norm, weights_PCR)

        # computing the error
        pcr_err_tr = y_train_norm - y_train_pred
        pcr_err_te = y_test_norm - y_test_pred

        # squared error
        squared_error_tr = np.linalg.norm(pcr_err_tr, ord=2)
        squared_error_te = np.linalg.norm(pcr_err_te, ord=2)

        self.weights.append(weights_PCR)

        # print
        if self.graphs:
            Graphication(y_train_norm, y_train_pred, y_test_norm, y_test_pred, pcr_err_tr,
                         pcr_err_te, 'PCR', self.indices)

        return squared_error_tr, squared_error_te

    def PCR_reduced(self, X_train_norm, X_test_norm, y_train_norm, y_test_norm):

        percentage = [0.9, 0.99]

        #train sets
        Rx = X_train_norm.T.dot(X_train_norm) / X_train_norm.shape[0]
        eig_values, eig_vectors = np.linalg.eig(Rx)
        P = eig_values.sum()

        order_tr = len(eig_values) - 1
        errors_tr = []
        errors_te = []

        squared_errors_tr = []
        squared_errors_te = []

        y_train_preds = []
        y_test_preds = []

        weights = []

        L = 0
        limit = 0

        # iterate different values of percentage
        for x in percentage:

            for i in range(order_tr):
                if not limit > x * P:
                    limit += eig_values[i]
                    L += 1
                else:
                    break

            eig_values_L = eig_values[:L]
            lambda_L = np.eye(L) * eig_values_L

            eig_vectors_L = np.zeros((len(eig_values), L))
            for n in range(len(eig_vectors)):
                tmp = eig_vectors[n]
                eig_vectors_L[n] = tmp[:L]

            temp_L = eig_vectors_L.dot(np.linalg.inv(lambda_L)) / X_train_norm.shape[0]
            weights_PCR_red = temp_L.dot(eig_vectors_L.T).dot(X_train_norm.T).dot(y_train_norm)

            json_w = {
                'percentage': x,
                'weights': weights_PCR_red
            }
            weights.append(json_w)

            self.weights.append(weights_PCR_red)

            L = 0
            limit = 0

            # performing the prediction
            y_train_pred = np.dot(X_train_norm, weights_PCR_red)
            y_test_pred = np.dot(X_test_norm, weights_PCR_red)

            y_train_preds.append(y_train_pred)
            y_test_preds.append(y_test_pred)

            # computing the error
            pcr_err_tr = y_train_norm - y_train_pred
            pcr_err_te = y_test_norm - y_test_pred

            errors_tr.append(pcr_err_tr)
            errors_te.append(pcr_err_tr)

            squared_error_tr = np.linalg.norm(pcr_err_tr, ord=2)
            squared_error_te = np.linalg.norm(pcr_err_te, ord=2)

            squared_errors_tr.append(squared_error_tr)
            squared_errors_te.append(squared_error_te)

        # print
        for l in range(len(percentage)):
            if self.graphs:
                Graphication(y_train_norm, y_train_preds[l], y_test_norm, y_test_preds[l], errors_tr[l], errors_te[l],
                                    'PCR REDUCED {}%'.format(weights[l]['percentage'] * 100), self.indices)

        return squared_errors_tr, squared_errors_te


if __name__ == "__main__":

    PATH = 'C:\\Users\\matte\\Documents\\ICT for health\\Labs\\lab1\\data.csv'
    FO = 'total_UPDRS'

    regression = Regression(PATH, column=FO, graphs=False, splitting=True)
    data_train_norm, data_test_norm = regression.Create_Dataset()

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
    se_MSE_tr, se_MSE_te = regression.MSE(y_train_norm, y_test_norm, X_train_norm, X_test_norm)
    se_gr_tr, se_gr_te = regression.Gradient(weights_gr, 1e-4, 1.0e-4, y_train_norm, y_test_norm, X_train_norm, X_test_norm)
    se_sd_tr, se_sd_te = regression.Steepest_Descent(weights_sd, 1.0e-5, y_train_norm, y_test_norm, X_train_norm, X_test_norm)
    se_ridge_tr, se_ridge_te = regression.Ridge_regression(800, y_train_norm, y_test_norm, X_train_norm, X_test_norm)
    se_PCR_tr, se_PCR_te = regression.PCR(X_train_norm, X_test_norm, y_train_norm, y_test_norm)
    se_PCR_red_tr, se_PCR_red_te = regression.PCR_reduced(X_train_norm, X_test_norm, y_train_norm, y_test_norm)

    if regression.weights:

        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.figure(figsize=(12, 6))
        plt.plot(regression.weights[0], marker=".", label="minimum squares")
        plt.plot(regression.weights[1], marker="o", label="gradient descent")
        plt.plot(regression.weights[2], marker="x", label="steepest descent")
        plt.plot(regression.weights[3], marker="+", label="ridge")
        plt.plot(regression.weights[4], marker="*", label="PCR")
        plt.xticks(range(len(regression.weights[0])), regression.indices, rotation="vertical")
        plt.title("Regression coefficients")
        plt.xlabel("Feature")
        plt.ylabel("Weight")
        plt.legend(loc=0)
        plt.grid(True)
        plt.savefig('Jitter(%)/weight_comparison.png')

        plt.figure(2)
        plt.figure(figsize=(12, 6))
        plt.plot(regression.weights[0], marker=".", label="minimum squares")
        plt.plot(regression.weights[4], marker="o", label="PCR")
        plt.plot(regression.weights[5], marker="+", label="PCR - 90%")
        plt.plot(regression.weights[6], marker="*", label="PCR - 99%")
        plt.xticks(range(len(regression.weights[0])), regression.indices, rotation="vertical")
        plt.title("Regression coefficients")
        plt.xlabel("Feature")
        plt.ylabel("Weight")
        plt.legend(loc=0)
        plt.grid(True)
        plt.savefig('Jitter(%)/weight_comparison_PCR.png')

        plt._show()


