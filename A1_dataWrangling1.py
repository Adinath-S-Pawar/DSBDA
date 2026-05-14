"""Data Wrangling, I
Perform the following operations using Python on any open source dataset (e.g., data.csv)
1. Import all the required Python Libraries.
2. Locate an open source data from the web (e.g., https://www.kaggle.com). Provide a clear
 description of the data and its source (i.e., URL of the web site).
3. Load the Dataset into pandas dataframe.
4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe()
function to get some initial statistics. Provide variable descriptions. Types of variables etc.
Check the dimensions of the data frame.
5. Data Formatting and Data Normalization: Summarize the types of variables by checking
the data types (i.e., character, numeric, integer, factor, and logical) of the variables in the
data set. If variables are not in the correct data type, apply proper type conversions.
6. Turn categorical variables into quantitative variables in Python"""

print("Script started...")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder #package->module->class

df = pd.read_csv('Iris.csv') #df now contains all rows and columns of the Iris dataset. dataset should be downloaded in same dir

""" without downloading dataset - 
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)"""

print(df.head()) 	#Prints those first five rows to the terminal or output screen.

#Data Preprocessing

#check missing values
print("\nMissing Values in each column:")
#print(df.isnull().sum())

#Get Initial Statistics
print("\nStatistical Summary:")
#print(df.describe())

#Dataset Dimensions
print("\nDataset Shape (Rows, Columns):")
#print(df.shape) #It shows the number of rows and columns in the dataset.

#Data Formatting and Data Normalization

#Check Data Types
print("\nData Types:")
#print(df.dtypes)

#Type Conversion
#df['Id'] = df['Id'].astype(int)
# eg - df['sepal_length'] = df['sepal_length'].astype(float)

#Turning Categorical Variables into Quantitative Variables
"""The Species column is categorical.
We convert it into numeric quantitative form using Label Encoding."""

le = LabelEncoder()
df['Species_Encoded'] = le.fit_transform(df['Species'])
print("\nCategorical to Quantitative Conversion:")
print(df[['Species','Species_Encoded']].head())

""" 
df['Species']
Selects the column containing flower species names:
Iris-setosa
Iris-versicolor
Iris-virginica

le.fit()
The LabelEncoder scans all unique species names and assigns each a unique numeric label.

le.transform()
Converts each species name into its assigned numeric value.

fit_transform()
Performs both fitting and transformation in one step.

df['Species_Encoded']
Stores the newly generated numeric labels in a new column.
"""
