import csv
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_precision_recall_curve

# Logistic regression code for real-world credit card data
# iterations    - controls the number of models fit
# APs           - stores the AP scores
#
# Additionally, generates historgrams of AP scores

iterations = 30
APs = np.zeros((iterations, 2))

result = pd.read_csv("creditcard.csv")
Y = result.values[:, -1].astype(int)
Time = result.values[:, 0].astype(int)
X = result.values[:, 1:-1]

for i in range(iterations):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    lr = LogisticRegression(max_iter = 2000)
    lr.fit(X_train, y_train)
    # Uncomment to plot PR curve
#    disp = plot_precision_recall_curve(clf, X_test, y_test)
#    plt.show()
    APs[i, 0] = average_precision_score(y_train, lr.predict(X_train))
    APs[i, 1] = average_precision_score(y_test, lr.predict(X_test))

print(APs)
print(np.mean(APs, axis=0))
print(np.std(APs, axis=0))

fig, ((ax0), (ax1)) = plt.subplots(nrows=1, ncols=2)
ax0.title.set_text('AP of Logistic Regression on training data')
ax1.title.set_text('AP of Logistic Regression on testing data')
ax0.hist(APs[:,0], alpha=.8, edgecolor='blue')
ax1.hist(APs[:,1], alpha=.8, edgecolor='red')
plt.show()
