import csv
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
import os
import matplotlib.pyplot as plt

def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f>=BEGIN_DATE+'.pkl' and f<=END_DATE+'.pkl']
    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)
    df_final=df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True,inplace=True)
    df_final=df_final.replace([-1],0)
    return df_final

input_features=['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
       'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
       'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
       'TERMINAL_ID_RISK_30DAY_WINDOW']

iterations = 100

APs = np.zeros((iterations, 4))
# RF train, RF test, Logist train, logist test

for i in range(1, iterations+1):
    print(i)
    data = read_from_files(r"C:\Users\mason\Desktop\simulations_new\simul_" + str(i), '2018-05-14', '2018-05-15')
    X = data[input_features]
#    print(np.mean(X, axis=0))
    Y = data['TX_FRAUD']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    #np.count_nonzero(X_train['TX_AMOUNT'] > 220)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    lr = LogisticRegression(max_iter = 2000)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    APs[i-1, 0] = average_precision_score(y_train, rf.predict(X_train))
    APs[i-1, 1] = average_precision_score(y_test, rf.predict(X_test))
    APs[i-1, 2] = average_precision_score(y_train, lr.predict(X_train))
    APs[i-1, 3] = average_precision_score(y_test, lr.predict(X_test))
#    disp1 = plot_precision_recall_curve(rf, X_test, y_test)
#    disp2 = plot_precision_recall_curve(lr, X_test, y_test)
#    plt.show()
print(APs)
print(np.mean(APs, axis=0))
print(np.std(APs, axis=0))


fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

ax0.hist(APs[:,0], label='RF train', alpha=.8, edgecolor='blue')
ax1.hist(APs[:,1], label='RF test', alpha=.8, edgecolor='red')
ax2.hist(APs[:,2], label='LR train', alpha=.8, edgecolor='green')
ax3.hist(APs[:,3], label='LR test', alpha=.8, edgecolor='orange')
ax0.title.set_text('AP of Random Forest on training Monte Carlo data')
ax1.title.set_text('AP of Random Forest on testing Monte Carlo data')
ax2.title.set_text('AP of Logistic Regression on training Monte Carlo data')
ax3.title.set_text('AP of Logistic Regression on testing Monte Carlo data')
plt.show()
