#Selection of libraries for our Project
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

from logging import StreamHandler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Intialize thr models
lr=LinearRegression()
dtr=DecisionTreeRegressor()
rfr=RandomForestRegressor()

from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

seoul_df=pd.read_csv('SeoulBikeData.csv', encoding='latin-1')

seoul_df.head()

#TITLE OF THE APP


st.title("Predicting Bicycle demand in Seoul: A machine Learning Model")

#data overview
st.header("data Overview for first 10 row")
st.write(seoul_df.head(10))   

st.header("Dataset Metadata")

# Number of rows and columns
num_rows, num_columns = seoul_df.shape
st.write(f"**Number of rows:** {num_rows}")
st.write(f"**Number of columns:** {num_columns}")

# Overview of column names and types
st.write("**Column Names and Data Types:**")
st.write(seoul_df.dtypes)

st.header("Statistical Summary")

# Display descriptive statistics
st.write("**Descriptive Statistics of the Dataset:**")
st.write(seoul_df.describe())

#new code

seoul_df['Holiday']=le.fit_transform(seoul_df['Holiday'])
seoul_df['Functioning Day']=le.fit_transform(seoul_df['Functioning Day'])

ohe=OneHotEncoder()
encoder=ohe.fit_transform(seoul_df[['Seasons']])
encoder_arr=encoder.toarray()
df_seasons=pd.DataFrame(encoder_arr, columns=ohe.get_feature_names_out(['Seasons']), dtype=int)

seoul_fin_df=pd.concat([seoul_df, df_seasons], axis=1)
seoul_fin_df.drop('Seasons', axis=1, inplace=True)
num_cols=seoul_fin_df[['Rented Bike Count','Hour','Temperature(ï¿½C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Dew point temperature(ï¿½C)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']]
cat_cols=seoul_fin_df.drop(num_cols, axis=1)

seoul_fin_df.drop('Date', axis=1, inplace=True)

#Split the data into 2 parts: input and output
X=seoul_fin_df.drop('Rented Bike Count', axis=1) #input - features

y=seoul_fin_df['Rented Bike Count'] #Output - Target variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

num_cols_train=X_train[['Hour','Temperature(ï¿½C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Dew point temperature(ï¿½C)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']]
cat_cols_train=X_train.drop(num_cols_train, axis=1)
num_cols_test=X_test[['Hour','Temperature(ï¿½C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Dew point temperature(ï¿½C)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']]
cat_cols_test=X_test.drop(num_cols_test, axis=1)

num_cols_train=ss.fit_transform(num_cols_train)
num_cols_test=ss.transform(num_cols_test)

#Rebuild X_train and X_test
#Rebuild X_train and X_test
X_train_1 = pd.concat([pd.DataFrame(num_cols_train, index=cat_cols_train.index, columns=X_train.columns[0:9]), cat_cols_train], axis=1)
X_test_1 = pd.concat([pd.DataFrame(num_cols_test, index=cat_cols_test.index, columns=X_test.columns[0:9]), cat_cols_test], axis=1)

# Intialize thr models
lr=LinearRegression()
dtr=DecisionTreeRegressor()
rfr=RandomForestRegressor()

# model training
lr.fit(X_train_1, y_train) # model training for linear regression
y_pred_lr=lr.predict(X_test_1)
dtr.fit(X_train_1, y_train) # model training for decision tree
rfr.fit(X_train_1, y_train) # model training for random forest

mod=st.selectbox("Select a model", ("Linear Regression", "Random Forest", "Decision tree"))


models={
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Decision tree":DecisionTreeRegressor()
}

#train the Model
selected_model=models[mod] #initializing the selected model

# train the selected model
selected_model.fit(X_train_1,y_train)

#make predictions
y_pred=selected_model.predict(X_test_1)

#model evaluation
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

#display results
st.write(f"r2 Score: {r2}")
st.write(f"Mean Square Error:{mse}")
st.write(f"Mean Absolute Error:{mae}")



















