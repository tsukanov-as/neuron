from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', parser='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42, train_size=60000)

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
clf = MultinomialNB()

clf.fit(x_train,y_train)
y_predicted = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print("MultinomialNB = {:.2f}%".format(accuracy_score(y_test, y_predicted)*100))

# from sklearn.metrics import classification_report
# print("Classification Report \n {}".format(classification_report(y_test, y_predicted, labels=range(0,10))))

import numpy as np
from nn import LayerOr, LayerAnd

x_train = x_train.values.astype(float) / 255
y_train = y_train.array.codes

x_test = x_test.values.astype(float) / 255
y_test = y_test.array.codes

lo = LayerOr(10, 28*28)
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

la = LayerAnd(10, 28*28*2)
for i, v in enumerate(x_train):
    la.feed(y_train[i], v)

total = 0
for i, v in enumerate(x_test):
    p = la.calc(v)
    if p.argmax() == y_test[i]:
        total += 1

print("LayerAnd = {:.2f}%".format(total / len(x_test) * 100))