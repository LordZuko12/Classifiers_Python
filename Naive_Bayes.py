from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

"""weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

le = preprocessing.LabelEncoder()
weather_encoded=le.fit_transform(weather)
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

features=zip(weather_encoded,temp_encoded)

model = GaussianNB()
model.fit(tuple(features),label)
predicted = model.predict([[2,1]])

print(predicted)"""


wine = datasets.load_wine()
data = wine.data
target = wine.target
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3,random_state=109)
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

