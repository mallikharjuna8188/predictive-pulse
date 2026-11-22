"""Run once:  python train.py
Needs hypertension_dataset.csv in the same folder.
Creates scaler.pkl and logreg_model.pkl (used by app.py)."""
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("hypertension_dataset.csv")

# text -> number encoding (same mapping is used in index.html)
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df["Smoking_Status"] = df["Smoking_Status"].map({"Never": 0, "Former": 1, "Current": 2})
df["Physical_Activity_Level"] = df["Physical_Activity_Level"].map({"Low": 0, "Moderate": 1, "High": 2})
df["Family_History"] = df["Family_History"].map({"No": 0, "Yes": 1})
df["Diabetes"] = df["Diabetes"].map({"No": 0, "Yes": 1})
y = df["Hypertension"].map({"Low": 0, "High": 1})

FEATURES = [
    "Age", "BMI", "Systolic_BP", "Diastolic_BP", "Stress_Level", "Salt_Intake",
    "Sleep_Duration", "Gender", "Smoking_Status", "Physical_Activity_Level",
    "Family_History", "Diabetes",
]
X = df[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# class_weight="balanced": otherwise the model answers "High" for everyone
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_s, y_train)

pred = model.predict(X_test_s)
proba = model.predict_proba(X_test_s)[:, 1]
print("Accuracy :", round(accuracy_score(y_test, pred), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, pred, target_names=["Low", "High"]))

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved scaler.pkl and logreg_model.pkl")
