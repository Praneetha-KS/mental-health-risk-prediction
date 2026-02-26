import pandas as pd
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# Load dataset

df = pd.read_excel("MentalHealthSurvey.xlsx")

# Preprocessing

from preprocessing import (
    convert_cgpa,
    convert_sleep,
    convert_sports,
    stress_activity_binary,
    convert_campus_discrimination
)

df["cgpa"] = df["cgpa"].apply(convert_cgpa)
df["average_sleep"] = df["average_sleep"].apply(convert_sleep)
df["sports_engagement"] = df["sports_engagement"].apply(convert_sports)
df["stress_relief_activities"] = df["stress_relief_activities"].apply(stress_activity_binary)
df["campus_discrimination"] = df["campus_discrimination"].apply(convert_campus_discrimination)

# Target creation

RISK_FEATURES = ["depression", "anxiety", "isolation", "future_insecurity"]
df["risk_score"] = df[RISK_FEATURES].mean(axis=1)
df["at_risk"] = (df["risk_score"] >= 3.5).astype(int)

print("Class Distribution:")
print(df["at_risk"].value_counts())
print("\nPercentage Distribution:")
print(df["at_risk"].value_counts(normalize=True))

FEATURES = [
    "age",
    "cgpa",
    "average_sleep",
    "academic_workload",
    "academic_pressure",
    "financial_concerns",
    "social_relationships",
    "campus_discrimination",
    "sports_engagement",
    "study_satisfaction",
    "stress_relief_activities"
]

df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

X = df[FEATURES]
y = df["at_risk"]

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
}

results = {}

best_model = None
best_f1 = 0
best_model_name = ""

# Training & Evaluation

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results[name] = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }

    print(f"\n {name} ")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# Save best model

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save metrics comparison
output = {
    "best_model": best_model_name,
    "model_comparison": results
}

with open("metrics.json", "w") as f:
    json.dump(output, f, indent=4)

print("\nBest Model Selected:", best_model_name)
print("Model saved as model.pkl")
print("Metrics saved as metrics.json")

