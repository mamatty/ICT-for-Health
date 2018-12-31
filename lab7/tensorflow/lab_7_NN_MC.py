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

    class_id = data_clear.iloc[:, -1]

    classN16 = np.zeros((len(class_id), 16))
    for i in range(len(class_id)):
        classN16[i][class_id[i] - 1] = 1

    y = data_clear.iloc[:, :-1]
    N, F = y.shape

    nodes_hl = round(F / 2)  # use F/2 hidden nodes for the hideen layers
    num_train = N / 2  # use N/2 patients for training and N/2 for testing

    y_train = y.loc[:num_train - 1, :]
    y_test = y.loc[num_train:, :]
    class_id_train = classN16[:int(num_train)]
    class_id_test = classN16[int(num_train):]

    x = tf.placeholder(tf.float32, [y_train.shape[0], F])
    t = tf.placeholder(tf.float32, [y_train.shape[0], 16])

    w1 = tf.Variable(tf.random_normal(shape=[F, nodes_hl], mean=0.0, stddev=1), dtype=tf.float32, name='w1')
    b1 = tf.Variable(tf.random_normal(shape=[1, nodes_hl], mean=0.0, stddev=1), dtype=tf.float32, name='b1')
    a1 = tf.matmul(x, w1) + b1
    z1 = tf.nn.sigmoid(a1)

    w2 = tf.Variable(tf.random_normal(shape=[nodes_hl, 16], mean=0.0, stddev=1), dtype=tf.float32, name='w2')
    b2 = tf.Variable(tf.random_normal(shape=[1, 16], mean=0.0, stddev=1), dtype=tf.float32, name='b2')
    y = tf.nn.softmax(tf.matmul(z1, w2) + b2)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=t, logits=y))
    optim = tf.train.GradientDescentOptimizer(0.4)
    optim_op = optim.minimize(cost, var_list=[w1, b1, w2, b2])

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    tf.local_variables_initializer().run(session=sess)

    iterations = []

    for i in range(1000):
        train_data = {x: y_train, t: class_id_train}
        sess.run(optim_op, feed_dict=train_data)
        iterations.append(cost.eval(feed_dict=train_data, session=sess))
        if i % 1000 == 0:
            print(i, cost.eval(feed_dict=train_data, session=sess))

    # --- print	the	final	results
    print(sess.run(w1), sess.run(b1))
    print(sess.run(w2), sess.run(b2))

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, marker="o", label="gradient descent")
    plt.title("Cross Entropy")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid(True)
    #plt.savefig('Jitter(%)/error_histrory.png')
    plt.show()

    conf_matrix = tf.confusion_matrix(labels=tf.argmax(t, axis=1), predictions=tf.argmax(tf.round(y), axis=1),
                                      num_classes=16)

    cm_train = conf_matrix.eval(feed_dict=train_data, session=sess)

    plt.figure(figsize=(9, 9))
    plot_confusion_matrix(cm_train, range(1, 17), normalize=True, title='Confusion Matrix - Multi Class -  Training')
    plt.show()

    test_data = {x: y_test, t: class_id_test}
    cm_test = conf_matrix.eval(feed_dict=test_data, session=sess)

    plt.figure(figsize=(9, 9))
    plot_confusion_matrix(cm_test, range(1, 17), normalize=True, title='Confusion Matrix - Multi Class -  Test')
    plt.show()