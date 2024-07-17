#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load digits dataset
digits = load_digits()

print("Image Data Shape:", digits.data.shape)
print("Label Data Shape:", digits.target.shape)

plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title("Training: %i\n" % label, fontsize=20)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

logisticRegr = LogisticRegression(max_iter=10000)
logisticRegr.fit(x_train, y_train)

print(logisticRegr.predict(x_test[0].reshape(1, -1)))

logisticRegr.predict(x_test[0:10])

predictions = logisticRegr.predict(x_test)

score = logisticRegr.score(x_test, y_test)
print(score)

index = 0
misclassifiedIndex = []
for predict, actual in zip(predictions, y_test):
    if predict != actual:
        misclassifiedIndex.append(index)
    index += 1

plt.figure(figsize=(20, 3))
for plotIndex, wrong in enumerate(misclassifiedIndex[0:4]):
    plt.subplot(1, 4, plotIndex + 1)
    plt.imshow(np.reshape(x_test[wrong], (8, 8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual: {}".format(predictions[wrong], y_test[wrong]), fontsize=20)

plt.show()
