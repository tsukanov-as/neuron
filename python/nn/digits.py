from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.05, random_state=42)

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

x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255

lo = LayerOr(10, 8*8)
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

la = LayerAnd(10, 8*8*2)
for i, v in enumerate(x_train):
    la.feed(y_train[i], v)

total = 0
for i, v in enumerate(x_test):
    p = la.calc(v)
    if p.argmax() == y_test[i]:
        total += 1

print("LayerAnd = {:.2f}%".format(total / len(x_test) * 100))