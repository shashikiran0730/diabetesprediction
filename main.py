from flask.templating import render_template
import sklearn 
import os
import numpy as np
import pandas as pd
import pickle
data=pd.read_csv("C:\DATASETS\EXCEL\diabetes.csv")
df=pd.DataFrame(data)
#print(df)
df.drop(df[df["Insulin"]==0].index)
df.drop(df[df['Age']==0].index,axis=1)
#print(df)
df.drop(df[df["BloodPressure"]==0].index)
df.drop(df[df["SkinThickness"]==0].index)
#import matplotlib.pyplot as plt
#plt.scatter(df["Glucose"],df["Outcome"])
import seaborn as sns
#sns.pairplot(data=df,hue="Outcome")
import seaborn as sns
#sns.heatmap(df.corr(), annot=True)
#plt.show()
x=df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction"]]
y=df[['Outcome']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
cs3=DecisionTreeClassifier()
cs3.fit(x_train,y_train)
#print(cs3.predict(x_test))
#cs3.score(x_test,y_test)
filename = 'diabatesfinalized_model'
pickle.dump(cs3, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(x_test, y_test)
r=loaded_model.predict([[1,2,3,4,5,6,7]])