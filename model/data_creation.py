# Colection of Data and Data processing .

#Importing required modules 
import pandas as pd  
import numpy as np 
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

def data_load_preprocess(csv_file):
  train_DataSet = pd.read_csv(csv_file)  #"ML_Assignment_2.csv")
  
  #Check for any invalid data for Categorical values
  print("\n\n#########################################################")
  print("-----------## Checking Invalida data (Training Data Set) ##----------")
  print("#########################################################")
  for column in train_DataSet.columns:
    if(train_DataSet[column].dtype == 'object'):
      # Replace '?' with mode for a specific column
      mode_value = train_DataSet[column].replace(' ?', pd.NA).mode()[0]
      train_DataSet[column] = train_DataSet[column].replace(' ?', mode_value)
      #z Feature Scaling : Encode categorical variables 
  
  ## Since multipl features are Ordinal values converting them to numerical values by using Label encoding
  target_encoder = LabelEncoder()  
  train_DataSet['Salary'] = target_encoder.fit_transform(train_DataSet['Salary'])

  #Feature selection
  #Removing Duplicate features (education)
  # Droping actual original category columns
  # Seperating the features (x) and traget(y)
  X = train_DataSet.drop(['Salary', 'education-num'], axis=1)    #['WorkClass', 'marital-status', 'occupation', 'relationship','race','sex','native-country', 'education', 'Salary', 'Salary_encoded'], axis=1)
  y = train_DataSet['Salary']

  print("Training Data:\n", X)
  print("Training output:\n", y)

  #Since features are at different scale using Min Max scaler
  numerical_columns = ['Age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
  scaler_minmax = MinMaxScaler()
  X[numerical_columns] = scaler_minmax.fit_transform(X[numerical_columns])

  #Performing one-Hot encoding for Categorical values
  #X = pd.get_dummies(X, columns=['workclass_encoded', 'marital-status_encoded', 'occupation_encoded', 'relationship_encoded','race_encoded','sex_encoded','native-country_encoded'])
  X = pd.get_dummies(X, columns=['WorkClass', 'marital-status', 'occupation', 'relationship','race','sex','native-country', 'education'])
  #print("Dummies:\n", X.columns)
  
  # Split data into training and validation sets (80% train, 20% validation)
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify = y)

  print("\nOutput Data:\n",y_train.shape, y_val.shape )

  return (X_train, X_val, y_train, y_val, scaler_minmax, target_encoder, X.columns )


  
