import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('features.csv')
X = df.drop(['filename','genre'],axis=1).values
y = df['genre'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train,y_train)

print(classification_report(y_test, clf.predict(X_test)))
joblib.dump(clf, 'genre_clf.joblib')
