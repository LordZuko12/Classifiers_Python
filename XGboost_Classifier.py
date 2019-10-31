from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.metrics import confusion_matrix

wine = datasets.load_wine()
data = wine.data
target = wine.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=8)

xgb = XGBClassifier()
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)

#predictions = [round(value) for value in y_pred]
#print(predictions)

print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)