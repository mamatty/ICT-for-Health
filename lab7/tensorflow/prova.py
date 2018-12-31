#	we	import	the	.mat	file	of	arrhythmia
import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import itertools

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

# Create a second copy of the original dataframe for multiple classes classification
data_original = pd.read_csv("arrhythmia.data", header=None, na_values="?")
data = data_original.dropna(axis=1)  # drop nan values
data = data.loc[:, (data != 0).any(axis=0)]  # drop columns having all zeroes
class_id = data[279]
n_classes = int(max(class_id))
(N, F) = np.shape(data)
mx_classes = np.zeros((N, n_classes))
for i in range(0, N - 1):
    mx_classes[i][int(class_id[i]) - 1] = 1
data = data.iloc[:, :-1]
(N, F) = np.shape(data)
mean = np.mean(data)
std = np.std(data)
x_norm = (data - mean) / std
mean = np.mean(x_norm, 0)
var = np.var(x_norm, 0)
n_healthy = sum(class_id == 0)
n_ill = sum(class_id == 1)
#	initializing	the	neural	network	graph
tf.set_random_seed(1234)
learning_rate = 1e-2
n_hidden_nodes_1 = 64
n_hidden_nodes_2 = 32

x = tf.placeholder(tf.float64, [N, F])
t = tf.placeholder(tf.float64, [N, n_classes])
#	first	layer
w1 = tf.Variable(tf.random_normal(shape=[F, n_hidden_nodes_1], mean=0.0, stddev=1.0,
                                  dtype=tf.float64, name="weights"))
b1 = tf.Variable(tf.random_normal(shape=[1, n_hidden_nodes_1], mean=0.0, stddev=1.0,
                                  dtype=tf.float64, name="biases"))
a1 = tf.matmul(x, w1) + b1
z1 = tf.nn.sigmoid(a1)
#	second	layer
w2 = tf.Variable(tf.random_normal(shape=[n_hidden_nodes_1, n_hidden_nodes_2], mean=0.0,
                                  stddev=1.0, dtype=tf.float64, name="weights2"))
b2 = tf.Variable(tf.random_normal(shape=[1, n_hidden_nodes_2], mean=0.0, stddev=1.0,
                                  dtype=tf.float64, name="biases2"))
a2 = tf.matmul(z1, w2) + b2
z2 = tf.nn.sigmoid(a2)
#	second	layer
w3 = tf.Variable(tf.random_normal(shape=[n_hidden_nodes_2, n_classes], mean=0.0, stddev=1.0,
                                  dtype=tf.float64, name="weights3"))
b3 = tf.Variable(tf.random_normal(shape=[1, 1], mean=0.0, stddev=1.0, dtype=tf.float64,
                                  name="biases3"))
y = tf.nn.softmax(tf.matmul(z2, w3) + b3)

# implementation	of	gradient	algorithm
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=t, logits=y))
optim = tf.train.GradientDescentOptimizer(learning_rate, name="GradientDescent")
optim_op = optim.minimize(cost, var_list=[w1, b1, w2, b2, w2, b3])
init = tf.global_variables_initializer()

# --- run	the	learning	machine
sess = tf.Session()
sess.run(init)
xval = x_norm.values.reshape(N, F)
tval = mx_classes.reshape(N, n_classes)
for i in range(10000):
    #	generate	the	data
    #	train
    input_data = {x: xval, t: tval}
    sess.run(optim_op, feed_dict=input_data)
    if i % 1000 == 0:  # print	the	intermediate	result
        print(i, cost.eval(feed_dict=input_data, session=sess))
# --- print	the	final	results
print(sess.run(w1), sess.run(b1))
print(sess.run(w2), sess.run(b2))
print(sess.run(w3), sess.run(b3))
decisions = np.zeros(N)
yval = y.eval(feed_dict=input_data, session=sess)
for i in range(0, N):
    decisions[i] = np.argmax(yval[i])

hist, bins = np.histogram((class_id - decisions), bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.xlabel('Squared	error')
plt.title('Error	distribution	with	learning	rate	' + str(learning_rate))
plt.savefig('squared_error_16classes' + str(learning_rate) + '.png', format='png')
plt.show()

conf_matrix = tf.confusion_matrix(labels=tf.argmax(t, axis=1), predictions=tf.argmax(tf.round(y), axis=1),
                                  num_classes=16)

cm_train = conf_matrix.eval(feed_dict=input_data, session=sess)

plt.figure(figsize=(9, 9))
plot_confusion_matrix(cm_train, range(1, 17), normalize=True, title='Confusion Matrix - Multi Class -  Training')
plt.show()