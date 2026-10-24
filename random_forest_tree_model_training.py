import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

data=pd.read_csv("Cleaned_College_Student_Performance.csv")

X=data[['SID','FinalMarks','CGPA']]
y=data['FinalMarks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100,random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

results = pd.DataFrame({"Actual Marks": y_test.values,"Predicted Marks": y_pred.round(2)})
print(results)

print("Original dataset:", data.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Results:", results.shape)


'''plt.figure(figsize=(8, 5))

sns.scatterplot(x=y_test, y=y_pred)

# Ideal prediction reference line
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
plt.plot([low, high], [low, high], linestyle="--")

plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")
plt.title("Random Forest: Actual vs Predicted Marks")
plt.grid(True)
plt.tight_layout()
plt.show()'''


# Compare marks for the first 15 students
comparison = results

comparison.plot(
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

















