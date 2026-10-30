
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load cleaned dataset
data = pd.read_csv("Cleaned_College_Student_Performance.csv")

# 2. Select input features
features = [
    "Gender",
    "StudyHours",
    "Attendance",
    "PreviousScore",
    "AssignmentScore",
    "SleepHrs",
    "SocialMedia",
    "PhysicalActivity",
    "StressLevel",
    "CGPA"
]


# 3. Select X and y
X = data[features].copy()
y = data["FinalMarks"]

# 4. Identify categorical columns automatically
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Store available categories for Flask
category_values = {}

for col in categorical_columns:
    X[col] = X[col].astype(str)
    category_values[col] = sorted(X[col].unique().tolist())

# 5. Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# Store training feature columns
model_columns = X.columns.tolist()

# 6. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# 8. Train model
model.fit(X_train, y_train)

# 9. Predict test data
y_pred = model.predict(X_test)

# 10. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Model Training Completed!")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 11. Actual vs Predicted results
results = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Predicted Marks": y_pred.round(2)
})

print(results)

# 12. Plot comparison
if __name__ == "__main__":

    results.head(15).plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8
    )

    plt.title("Actual vs Predicted Final Marks")
    plt.xlabel("Student (Test Data)")
    plt.ylabel("Final Marks")
    plt.xticks(rotation=0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
