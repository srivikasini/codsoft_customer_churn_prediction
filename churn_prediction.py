import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("Churn_Modelling.csv")

data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

label_encoder = LabelEncoder()
data["Gender"] = label_encoder.fit_transform(data["Gender"])
data["Geography"] = label_encoder.fit_transform(data["Geography"])

X = data.drop("Exited", axis=1)
y = data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sample_customer = pd.DataFrame(
    [[600, 1, 0, 40, 3, 60000, 2, 1, 1, 50000]],
    columns=X.columns
)

prediction = model.predict(sample_customer)


if prediction[0] == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")
