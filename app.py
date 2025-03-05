import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load the model
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as f:
    label_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Customer Churn Predictor')

# Create a form to enter the customer details
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Convert geography to one hot encoding

Geography_France = 1 if geography == 'France' else 0
Geography_Germany = 1 if geography == 'Germany' else 0
Geography_Spain = 1 if geography == 'Spain' else 0

# Convert to pandas Dataframe

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography_France': [Geography_France],
    'Geography_Germany': [Geography_Germany],
    'Geography_Spain': [Geography_Spain],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Gender':[gender]
})

new_order = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
       'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France',
       'Geography_Germany', 'Geography_Spain']
input_data = input_data[new_order]

input_data['Gender'] = label_encoder.transform(input_data['Gender'])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# add churn probability to the app
st.write('Churn Probability:', prediction_proba)

if prediction_proba > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')