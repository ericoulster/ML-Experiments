from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

df = pd.read_csv('creditcardcsvpresent.csv')


### Exploratory Data Analysis ###


# How many Fraud Samples?
df.isFradulent.value_counts()

### Feature Engineering ###

# turn Y/N into booleans
df['Is declined'] = df['Is declined'].map(dict(Y=1, N=0))
df['isFradulent'] = df['isFradulent'].map(dict(Y=1, N=0))
df['isForeignTransaction'] = df['isForeignTransaction'].map(dict(Y=1, N=0))
df['isHighRiskCountry'] = df['isHighRiskCountry'].map(dict(Y=1, N=0))

# Dropping isFradulent because it's our target. Dropping Transaction date because it's empty.
X = df.drop(columns=['isFradulent', 'Transaction date'])

y = df[['isFradulent']]

### Analysis ###

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


model = LogisticRegression(penalty='l1')

model.fit(X_train, y_train)

predicted = model.predict(X_test)

print(classification_report(y_test, predicted))

# According to the above results, this model classifies 99% of results correctly

print(confusion_matrix(y_true=y_test, y_pred=predicted))


# Based on the above results, you can see:
# 
# -769 test samples true positive (normal transactions correctly identified as normal transactions)
# 
# -7 test samples false positive (fraud incorrectly identified as normal transactions)
# 
# -143 samples true negative (fraud correctly identified as fraud)
# 
# -4 samples false negative (normal transactions incorrectly identified as fraud)
# 
