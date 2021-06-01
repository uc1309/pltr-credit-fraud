import csv
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

# Random forest code for real-world credit card data
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
    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    clf.fit(X_train, y_train)
    # Uncomment to plot PR curve
#    disp = plot_precision_recall_curve(clf, X_test, y_test)
#    plt.show()
    APs[i, 0] = average_precision_score(y_train, clf.predict(X_train))
    APs[i, 1] = average_precision_score(y_test, clf.predict(X_test))

print(APs)
print(np.mean(APs, axis=0))
print(np.std(APs, axis=0))

fig, ((ax0), (ax1)) = plt.subplots(nrows=1, ncols=2)
ax0.title.set_text('AP of Random Forest on training data')
ax1.title.set_text('AP of Random Forest on testing data')
ax0.hist(APs[:,0], alpha=.8, edgecolor='blue')
ax1.hist(APs[:,1], alpha=.8, edgecolor='red')
plt.show()
