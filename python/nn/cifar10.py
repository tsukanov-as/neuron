import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
clf = MultinomialNB()

import numpy as np

# x_train = tf.reshape(tf.image.rgb_to_grayscale(x_train), (50000, 32*32)).numpy()
# x_test = tf.reshape(tf.image.rgb_to_grayscale(x_test), (10000, 32*32)).numpy()

rgb_to_grayscale = [0.299, 0.587, 0.114]
x_train = np.dot(x_train[...,:3], rgb_to_grayscale).reshape(50000, 32*32)
x_test = np.dot(x_test[...,:3], rgb_to_grayscale).reshape(10000, 32*32)

y_train = y_train.ravel()
y_test = y_test.ravel()

clf.fit(x_train, y_train)
y_predicted = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print("MultinomialNB = {:.2f}%".format(accuracy_score(y_test, y_predicted)*100))

# from sklearn.metrics import classification_report
# print("Classification Report \n {}".format(classification_report(y_test, y_predicted, labels=range(0,10))))

from nn import LayerOr, LayerAnd

x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255

lo = LayerOr(10, 32*32)
for i, v in enumerate(x_train):
    lo.feed(y_train[i], v)

total = 0
for i, v in enumerate(x_test):
    p = lo.calc(v)
    if p.argmax() == y_test[i]:
        total += 1

print("LayerOr = {:.2f}%".format(total / len(x_test) * 100))

x_train = np.append(x_train, 1 - x_train, axis=1)
x_test = np.append(x_test, 1 - x_test, axis=1)

la = LayerAnd(10, 32*32*2)
for i, v in enumerate(x_train):
    la.feed(y_train[i], v)

total = 0
for i, v in enumerate(x_test):
    p = la.calc(v)
    if p.argmax() == y_test[i]:
        total += 1

print("LayerAnd = {:.2f}%".format(total / len(x_test) * 100))