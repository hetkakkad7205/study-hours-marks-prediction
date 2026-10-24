import pandas as pd
import numpy as np

df=pd.read_csv("college_student_performance_analysis.csv")
print(df.head())# this will print first top 5 records
print(df.info())# this will print info about datatype
print(df.shape)
