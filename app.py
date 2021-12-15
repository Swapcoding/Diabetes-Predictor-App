import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.title('Diabetes Prediction')

data = pd.read_csv('E:\Datasets\diabetes.csv')

x = data.iloc[: , :-1]
y = data.iloc[: , 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = LogisticRegression()
model.fit(x_train,y_train)


Pregnancies = st.text_input('Enter number of pregnancies:')
Glucose = st.text_input('Glucose Level:')
BloodPressure = st.text_input('BP:')
Skin_Thickness = st.text_input('Skin Thickness:')
Insulin = st.text_input('Insulin Levels:')
BMI = st.text_input('Enter BMI:')
Pedigree_Function = st.text_input('Enter DPF:')
Age = st.text_input('Enter Age :')

final_data = pd.Series([Pregnancies,Glucose,BloodPressure,Skin_Thickness,Insulin,BMI,Pedigree_Function,Age],['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

arr = np.array([final_data])
try:
    if(st.button("Predict")):
        if(model.predict(arr) == 1):
            st.write("You are most likely having Diabetes")
        else:
            st.write("No!! You do not have diabetes")
except:
    st.write('Processing......')







