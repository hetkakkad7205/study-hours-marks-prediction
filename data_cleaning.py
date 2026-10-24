import numpy as np
import pandas as pd

#Step1 : Load csv file 
data=pd.read_csv("college_student_performance_analysis.csv")
print(data.head())

# Step 2: Display basic information
print("Original dataset shape:", data.shape)
print("Dataset information:")
data.info()

# Step 3:finding NULL
print("Missing values in each column")
print(data.isnull().sum())

# Step 4: Remove duplicate rows
print("Duplicate rows:", data.duplicated().sum())
data=data.drop_duplicates()

# Step 5: Clean categorical data
data["Gender"] = data["Gender"].astype("string").str.strip().str.title()

# Step 6: Remove records with missing or duplicate student IDs
data = data.dropna(subset=["SID"])
data = data.drop_duplicates(subset=["SID"])

# Step 7: Check the cleaned dataset
print("Cleaned dataset shape:", data.shape)
print("Missing values after cleaning:")
print(data.isnull().sum())

print("\nDuplicate rows after cleaning:", data.duplicated().sum())

# Step 8: Save the cleaned dataset
data.to_csv("Cleaned_College_Student_Performance.csv", index=False)

print("Data cleaning completed successfully!")
print("Cleaned file saved successfully.")
