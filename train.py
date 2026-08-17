import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# Load cleaned data
df = pd.read_csv("data/cleaned/rainfall_dataset_cleaned.csv")

FEATURES = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "cloud",
    "uv_index",
    "visibility_km",
    "gust_kph"
]

TARGET = "rain"

X = df[FEATURES]
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "RandomForest": RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf"),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

best_model = None
best_accuracy = 0

print("\nTraining models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {acc * 100:.2f}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save best model
pickle.dump(best_model, open("model.pkl", "wb"))

print(f"\nBest Model Saved (Accuracy: {best_accuracy * 100:.2f}%)")
