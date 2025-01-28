# app.py
import streamlit as st
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Load and set up the page
image = Image.open('./img/funding.png')
st.set_page_config(page_title='Loan Prediction', page_icon=image)
st.title("Loan Repayment Prediction")

st.write("""This application automates the loan eligibility process for customers across urban, semi-urban, 
and rural areas.The application provides managment with decision support capabilities to validate the 
eligibility of each customer for a loan""")  # Truncated for brevity

imgs = Image.open("./img/funding.png")
st.sidebar.image(imgs)
st.sidebar.write('This AI classification model predicts loan repayment capability of customers based on predetermined criteria')

st.sidebar.button("Learn More")

# Load data
data = pd.read_csv(r"C:\Users\HP\Desktop\DATALAB\pythonclasses\Home_Loan_Prediction\loan_data_clean.csv")

# Define ordinal features and the label encoder
ordinal_features = ["Gender", 'Married', 'Education', 'Self_Employed', 'Property_Area']
label_encoder = LabelEncoder()

# Encode the target variable
data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])

# Ordinal Encoding for other categorical features
ordinal_encoder = OrdinalEncoder()
data[ordinal_features] = ordinal_encoder.fit_transform(data[ordinal_features])

# Initialize different classifiers
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Classifier': SVC(),
}

# Create a select box for choosing the model
model_selector = st.selectbox ('Choose a model', models.keys())

# Input fields for user input
Gender = st.selectbox('Gender', ['Male', 'Female'])
Married = st.selectbox('Married', ['Yes', 'No'])
Dependents = st.number_input('Dependents', min_value=0, step=1)
Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox('Self Employed', ['Yes', 'No'])
ApplicantIncome = st.number_input('Applicant Income', min_value=0)
CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
LoanAmount = st.number_input('Loan Amount', min_value=0)
Loan_Amount_Term = st.number_input('Loan Amount Term', min_value=0)
Credit_History = st.number_input('Credit History', min_value=0, max_value=1)
Property_Area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

# Function to prepare input data
def input_data(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    Gender = 1 if Gender == 'Male' else 0
    Married = 1 if Married == 'Yes' else 0
    Dependents = 3 if Dependents > 2 else int(Dependents)
    Education = 0 if Education == 'Graduate' else 1
    Self_Employed = 1 if Self_Employed == 'Yes' else 0
    Property_Area = {'Urban': 2, 'Rural': 0, 'Semiurban': 1}[Property_Area]

    Features = [Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
                CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]
    return np.array(Features).reshape(1, -1)

# Prediction button
if st.button('Predict'):
    input_features = input_data(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area)

    # Prepare data for model
    X = data.drop(columns=['Loan_ID', 'Loan_Status'])  # Exclude these columns
    y = data['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = models[model_selector]
    model.fit(X_train, y_train)

    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)

    result = "Repaid" if prediction[0] == 1 else "Defaulted"
    st.write(f'The predicted repayment status is: {result}')
    st.write(f"Confusion Matrix:\n{confusion_matrix(y_test, model.predict(X_test))}")
    st.write(f"Classification Report:\n{classification_report(y_test, model.predict(X_test))}")