import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
    cm = pd.DataFrame(data=cm)
    cm = cm.fillna(0)
    cm = cm.values

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

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":

    # Read the data and set NaN where there are ?
    data = pd.read_csv("arrhythmia.data", sep=',', header=None, na_values=['?', '\t?'])

    # Remove the columns with these NaN values
    data_clear = data.dropna(axis=1)

    # Override data_clear removing the columns whose values are all equal to zero
    data_clear = data_clear.loc[:, (data_clear != 0).any(axis=0)]

    # Create a first copy of the original dataframe for binary classification
    df1 = data_clear.copy()

    # Set elements of last column equal to 0 if their values are higher than 1
    last_column = df1.iloc[:, -1]
    last_column_correct = last_column.map(lambda x: 1 if (x > 1) else 0)
    df1.iloc[:, -1] = last_column_correct

    # Define y and class_id
    y = df1.iloc[:, :-1]
    class_id = df1.iloc[:, -1]

    class_id_counts = class_id.value_counts().tolist()
    n_healthy = class_id_counts[0]
    n_ill = class_id_counts[1]

    N, F = y.shape

    nodes_hl = round(F / 2)  # use F/2 hidden nodes for the hideen layers
    num_train = round(1 * N / 3)  # use N/2 patients for training and N/2 for testing

    y_train = y.loc[:num_train - 1, :]
    y_test = y.loc[num_train:, :]
    class_id_train = class_id.loc[:num_train - 1]
    class_id_test = class_id.loc[num_train:]

    # %% Binary Classification

    tf.set_random_seed(1000)
    np.random.seed(1234)

    x = tf.placeholder(tf.float32, [None, F])  # F input nodes

    t = tf.placeholder(tf.float32, [None, 1])  # 1 output node. With None we can give as much data as we want
    w1 = tf.Variable(tf.truncated_normal(shape=[F, nodes_hl], mean=0.0,
                                         stddev=1.0,
                                         dtype=tf.float32))  # it generates a random tensor with the specified size and according to
    # a Gaussian distribution with specified mean and variance
    b1 = tf.Variable(tf.truncated_normal(shape=[nodes_hl, ], mean=0.0,
                                         stddev=1.0, dtype=tf.float32))
    a1 = tf.matmul(x, w1) + b1

    z1 = tf.nn.sigmoid(a1)  # Computes sigmoid of x element-wise.

    w2 = tf.Variable(tf.truncated_normal(shape=[nodes_hl, 1], mean=0.0,
                                         stddev=1.0, dtype=tf.float32))
    b2 = tf.Variable(tf.truncated_normal(shape=[1, ], mean=0.0,
                                         stddev=1.0, dtype=tf.float32))
    y = tf.nn.sigmoid(tf.matmul(z1, w2) + b2)  # activation functions of the hidden layer nodes

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=t, logits=y))
    optim = tf.train.GradientDescentOptimizer(0.4)
    optim_op = optim.minimize(cost, var_list=[w1, b1, w2, b2])

    # definition of true and false positives and negatives
    true_neg, true_neg_op = tf.metrics.true_negatives(
        labels=t, predictions=tf.round(y))
    true_pos, true_pos_op = tf.metrics.true_positives(
        labels=t, predictions=tf.round(y))
    false_neg, false_neg_op = tf.metrics.false_negatives(
        labels=t, predictions=tf.round(y))
    false_pos, false_pos_op = tf.metrics.false_positives(
        labels=t, predictions=tf.round(y))

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    tf.local_variables_initializer().run(session=sess)
    train_data = {}

    for i in range(80000):
        train_data = {x: y_train, t: class_id_train.values.reshape(num_train, 1)}
        sess.run([optim_op], feed_dict=train_data)
        if i % 1000 == 0:
            print(i, cost.eval(feed_dict=train_data, session=sess))

    # calculating for training data
    tn_train = sess.run(true_neg_op,
                        feed_dict=train_data)  # in the feed dict we declare which data we are giving to the var
    tp_train = sess.run(true_pos_op, feed_dict=train_data)
    fn_train = sess.run(false_neg_op, feed_dict=train_data)
    fp_train = sess.run(false_pos_op, feed_dict=train_data)

    true_neg_train_nn = tn_train / (class_id_train == 0).sum()
    true_pos_train_nn = tp_train / (class_id_train == 1).sum()
    false_neg_train_nn = fn_train / (class_id_train == 1).sum()
    false_pos_train_nn = fp_train / (class_id_train == 0).sum()

    print('\nNeural Networks:\n')
    print('Train data:')
    print('True positive probability: ', true_pos_train_nn)
    print('True negative probability: ', true_neg_train_nn)
    print('False positive probability: ', false_pos_train_nn)
    print('False negative probability: ', false_neg_train_nn)

    plt.figure(figsize=(12, 8))
    plt.figure(1)
    plt.title('Classification Train Results: Neural Network - Binary', fontsize=15)
    plt.bar(1, true_pos_train_nn, align='center', label="True Positive", color='g')
    plt.bar(2, true_neg_train_nn, align='center', label="True Negative", color='r')
    plt.bar(3, false_pos_train_nn, align='center', label="False Positive", color='m')
    plt.bar(4, false_neg_train_nn, align='center', label="False Negative", color='y')
    plt.legend()
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.grid(axis='y')

    plt.show()

    # calculating for testing data
    test_data = {x: y_test, t: class_id_test.values.reshape(301, 1)}

    tf.local_variables_initializer().run(session=sess)
    tn_test = sess.run(true_neg_op, feed_dict=test_data)
    tp_test = sess.run(true_pos_op, feed_dict=test_data)
    fn_test = sess.run(false_neg_op, feed_dict=test_data)
    fp_test = sess.run(false_pos_op, feed_dict=test_data)

    true_neg_test_nn = tn_test / (class_id_test == 0).sum()
    true_pos_test_nn = tp_test / (class_id_test == 1).sum()
    false_neg_test_nn = fn_test / (class_id_test == 1).sum()
    false_pos_test_nn = fp_test / (class_id_test == 0).sum()

    print('Test data:')
    print('True positive probability: ', true_pos_test_nn)
    print('True negative probability: ', true_neg_test_nn)
    print('False positive probability: ', false_pos_test_nn)
    print('False negative probability: ', false_neg_test_nn)

    plt.figure(figsize=(12, 8))
    plt.figure(1)
    plt.title('Classification Test Results: Neural Network - Binary', fontsize=15)
    plt.bar(1, true_pos_test_nn, align='center', label="True Positive", color='g')
    plt.bar(2, true_neg_test_nn, align='center', label="True Negative", color='r')
    plt.bar(3, false_pos_test_nn, align='center', label="False Positive", color='m')
    plt.bar(4, false_neg_test_nn, align='center', label="False Negative", color='y')
    plt.legend()
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.grid(axis='y')

    plt.show()