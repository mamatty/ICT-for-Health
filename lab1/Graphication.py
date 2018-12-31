import matplotlib.pyplot as plt


class Graphication:

    def __init__(self, y_train_norm, y_train_pred, y_test_norm, y_test_pred, error_train, error_test, alg, indices):

        self.y_train_norm = y_train_norm
        self.y_train_pred = y_train_pred
        self.y_test_norm = y_test_norm
        self.y_test_pred = y_test_pred
        self.error_train = error_train
        self.error_test = error_test
        self.alg = alg
        self.indices = indices

        self.comparison()
        self.difference()
        self.error_graph()

    def grafication(self, error_history_gradient, error_history_sd):

        plt.figure(figsize=(12, 6))
        plt.plot(error_history_gradient[:100], marker="o", label="gradient descent")
        plt.plot(error_history_sd[:100], marker="o", label="steepest descent")
        plt.title("Error history - first 50 iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.legend(loc=0)
        plt.grid(True)
        plt.savefig('Jitter(%)/error_histrory.png')
        plt.show()


    def comparison(self):

        plt.figure(figsize=(12, 6))
        plt.plot(range(-2, 18), range(-2, 18), 'r', label='Ideal Regression Line')
        plt.scatter(self.y_train_pred, self.y_train_norm)
        plt.xlabel('y_hat_train', fontsize=12)
        plt.ylabel('y_train', fontsize=12)
        plt.title('Comparison y_hat_train and y_train - {}'.format(self.alg), fontsize=15)
        plt.grid()
        plt.legend()
        plt.savefig('Jitter(%)/comparison_y_train_{}.png'.format(self.alg))

        plt.figure(figsize=(12,6))
        plt.plot(range(-2, 4), range(-2, 4), 'r', label='Ideal Regression Line')
        plt.scatter(self.y_test_pred, self.y_test_norm)
        plt.xlabel('y_hat_test', fontsize=12)
        plt.ylabel('y_test', fontsize=12)
        plt.title('Comparison y_hat_test and y_test - {}'.format(self.alg), fontsize=15)
        plt.grid()
        plt.legend()
        plt.savefig('Jitter(%)/comparison_y_test_{}.png'.format(self.alg))

        plt.show()

    def difference(self):
        plt.figure(1)
        plt.figure(figsize=(12, 10))
        plt.subplot(211)
        plt.plot(self.y_train_pred, 'r', label="yhat_train", linewidth=2.0)
        plt.title('yhat_train')
        plt.xlabel("Sample index")
        plt.ylabel("Normalized values")
        plt.grid(True)
        plt.subplot(212)
        plt.plot(self.y_train_norm, 'b', label="y_train", linewidth=2.0)
        plt.title('y_train')
        plt.xlabel("Sample index")
        plt.ylabel("Normalized values")
        plt.grid(True)
        plt.savefig('Jitter(%)/difference_1_y_train_{}.png'.format(self.alg))

        plt.figure(2)
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_train_pred, 'r', self.y_train_norm, 'b')
        plt.title('Comparison y_train')
        plt.xlabel("Sample index")
        plt.ylabel("Normalized values")
        plt.grid(True)
        plt.savefig('Jitter(%)/difference_2_y_train_{}.png'.format(self.alg))

        plt.figure(3)
        plt.figure(figsize=(12, 10))
        plt.subplot(211)
        plt.plot(self.y_test_pred, 'r', label="yhat_train", linewidth=2.0)
        plt.title('yhat_test')
        plt.grid(True)
        plt.subplot(212)
        plt.plot(self.y_test_norm, 'b', label="y_train", linewidth=2.0)
        plt.title('y_test')
        plt.xlabel("Sample index")
        plt.ylabel("Normalized values")
        plt.grid(True)
        plt.savefig('Jitter(%)/difference_1_y_test_{}.png'.format(self.alg))

        plt.figure(4)
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_test_pred, 'r', self.y_test_norm, 'b')
        plt.title('Comparison y_test')
        plt.xlabel("Sample index")
        plt.ylabel("Normalized values")
        plt.grid(True)
        plt.savefig('Jitter(%)/difference_2_y_test_{}.png'.format(self.alg))

        plt.show()

    def error_graph(self):

        plt.figure(5)
        plt.figure(figsize=(12, 10))
        plt.subplot(211)
        plt.hist(self.error_train, 50, normed=1, facecolor='green', alpha=0.75)
        plt.title('{}: histogram of the error distribution (Train)'.format(self.alg))
        plt.xlabel("Error")
        plt.ylabel("Occurrencies")
        plt.grid(True)
        plt.subplot(212)
        plt.hist(self.error_test, 50, normed=1, facecolor='green', alpha=0.75)
        plt.title('{}: histogram of the error distribution (Test)'.format(self.alg))
        plt.xlabel("Error")
        plt.ylabel("Occurrencies")
        plt.grid(True)
        plt.savefig('Jitter(%)/histogram_error_{}.png'.format(self.alg))

        plt.show()
