import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
def create_matrix(): # function to create  matrix 
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data") #derive data from excel sheet 
    X=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].to_numpy() # input feature matrix 
    return df,X
def classify(df): # function to classify customers 
    df['Class']=df['Payment (Rs)'].apply(lambda x: 'RICH' if x>200 else 'POOR') # threshold based classification 
    return df[['Customer','Payment (Rs)', 'Class']]
def train_model(X,y):
   ''' X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
    model=LogisticRegression()
    model.fit(X_train,y_train)'''
   model=LogisticRegression() # create model using logistic regression
   model.fit(X,y) # train model
   y_pred=model.predict(X) # predict rich or poor
   return y,y_pred
df,X = create_matrix()
print(classify(df))
y=df['Class'].to_numpy()#target class
y_test,y_pred=train_model(X,y)
print("Actual classes:")
print(y_test)
print("Predicted classes:")
print(y_pred)