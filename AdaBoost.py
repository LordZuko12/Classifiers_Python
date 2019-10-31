from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

wine = datasets.load_wine()
data = wine.data
target = wine.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

#base learner DecisionTreeClassifier
abc1 = AdaBoostClassifier(n_estimators=50,learning_rate=1)

#base learner SupportVectorClassifier
svc = SVC(probability=True, kernel='linear')
abc2 = AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

model1 = abc1.fit(x_train, y_train)
model2 = abc2.fit(x_train, y_train)

y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)

print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred1))
print("Accuracy using Decision Tree:",metrics.accuracy_score(y_test, y_pred1))
print("-----------")
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred2))
print("Accuracy using Support Vector:",metrics.accuracy_score(y_test, y_pred2))
