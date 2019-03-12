# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:10:45 2019

@author: Chandan
"""


import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"DATA_TRAIN - Copy.csv")


# In[2]:


df.head()


# In[3]:


def prepareData(train):
    train["Date_of_Journey"] = pd.to_datetime(train["Date_of_Journey"],format="%d-%m-%Y")
    train["Arrival_Date"] = pd.to_datetime(train["Arrival_Date"],format="%d-%m-%Y")
    train["Dep_Time"] = train["Dep_Time"].apply(lambda x:datetime.strptime(x, '%H:%M').time())
    train["Arrival_Time"] = train["Arrival_Time"].apply(lambda x:datetime.strptime(x, '%H:%M %p').time())
    train["Dep_hour"] = train["Dep_Time"].apply(lambda x:x.hour)
    train["Day_of_week_of_Journey"] = train["Date_of_Journey"].apply(lambda x:x.weekday())
    train["Month_of_Journey"] = train["Date_of_Journey"].apply(lambda x:x.month)
    train["Day_of_Journey"] = train["Date_of_Journey"].apply(lambda x:x.day)
    train["Dep_Time_early_morning"]= train["Dep_Time"].apply(lambda x: x<datetime.strptime('6:00', '%H:%M').time())

    cat_col = ['Airline','Source', 'Destination','Stop 1', 'Stop 2', 'Stop 3', 'Stop 4','Additional_Info']
    date_time_col = ['Date_of_Journey','Arrival_Date', 'Arrival_Time', 'Dep_Time', 'Arrival_Details']
    col_to_drop = ['Route','Price_Bucket']

    train_cat_col = pd.get_dummies(train, drop_first=True, columns=cat_col)
    train_cat_col.drop(date_time_col,axis=1,inplace=True)
    try:
        train_cat_col.drop(col_to_drop,axis=1,inplace=True)
    except:
        pass
    return train_cat_col


# In[4]:


train_cat_col = prepareData(df)


# In[5]:


# Apply feature scaling.
y = train_cat_col.Price
train_cat_col_noPrice = train_cat_col.drop(["Price"],axis=1)

scale = StandardScaler()
df_scaled = scale.fit_transform(train_cat_col_noPrice)

X = pd.DataFrame(df_scaled,columns=train_cat_col_noPrice.columns)





# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 42)


#neurons = 10


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
Regression = Sequential()

# Adding the input layer and the first hidden layer
Regression.add(Dense(units = 128, activation = 'relu', input_dim = (X_train.shape[1])))

Regression.add(Dense(units = 56, activation = 'relu'))
Regression.add(Dropout(p = 0.2))

Regression.add(Dense(units = 32, activation = 'relu'))
Regression.add(Dropout(p = 0.2))

Regression.add(Dense(units = 28, activation = 'relu'))
Regression.add(Dropout(p = 0.2))



Regression.add(Dense(units = 1, activation = 'relu'))

# Compiling the ANN
Regression.compile(optimizer = 'adam', loss = 'mean_absolute_percentage_error', metrics = ['mse'])

# Fitting the ANN to the Training set
Regression.fit(X_train, y_train, batch_size = 10, epochs = 500)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
predicted_value = Regression.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

r2_score(list(y_test), predicted_value)