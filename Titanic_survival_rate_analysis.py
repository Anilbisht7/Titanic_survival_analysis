import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("train.csv")

print(data.head())

data.info()

print(data.isnull().sum())

data["Age"] = data["Age"].fillna(data["Age"].median())

data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

data = data.drop("Cabin", axis=1)

print(data["Survived"].value_counts())
print(data["Pclass"].value_counts())
print(data["Sex"].value_counts())

data["Survived"].value_counts().plot(kind="bar")

plt.title("Survivle count")
plt.xlabel("Survived (0 = No and 1 = Yes)")
plt.ylabel("Passengers")

plt.show()


pd.crosstab(data["Sex"], data["Embarked"]).plot(kind = "bar")

plt.title("Survival by gender")
plt.xlabel("Gender")
plt.ylabel("count")

plt.show()


print(pd.crosstab(data["Sex"],data["Survived"]))

print(pd.crosstab(data["Pclass"],data["Survived"]))

print(data.head())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["Sex"] = le.fit_transform(data["Age"])
data["Embarked"] = le.fit_transform(data["Embarked"])

print(data.head(10))

X = data[[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]]
y = data["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(predictions[:10])


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,predictions)
print("Accuracy:", accuracy)



from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
