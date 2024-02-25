import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the model
model = load('model_saved_new.joblib')

def main():
    st.title('Loan defaulters Prediction App')
    st.subheader('Enter Customer Details Below to Predict Loan Defaulting Status')
    st.image('loan_image.jpg', use_column_width=True)
    st.divider()

    # Numerical Fields
    agecategory = ['<20', '20-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '>60']
    creditscorecategory =['<300', '300-400', '401-500', '501-600', '601-700', '701-800', '>800']

    Income = st.sidebar.number_input('Income')
    LoanAmount = st.sidebar.number_input('Loan Amount')
    MonthsEmployed = st.sidebar.number_input('Months Employed')
    NumCreditLines = st.sidebar.number_input('Number of Credit Lines')
    InterestRate = st.sidebar.number_input('Interest Rate')
    LoanTerm = st.sidebar.number_input('Loan Term')
    DTIRatio = st.sidebar.number_input('DTI Ratio')
    age_category = st.sidebar.selectbox('Select Age Category', agecategory)
    creditscore_category = st.sidebar.selectbox('Select Credit Score Category', creditscorecategory)

    # Categorical inputs
    Education = st.selectbox('Education', ['Select', 'High School', "Bachelor's", 'Master', 'PhD'])
    EmploymentType = st.selectbox('Employment Type', ['Select', 'Part-time', 'Full-time', 'Self-employed', 'Unemployed'])
    MaritalStatus = st.selectbox('Marital Status', ['Select', 'Single', 'Married', 'Divorced'])
    HasMortgage = st.radio('Has Mortgage', ['Select', 'Yes', 'No'])
    HasDependents = st.radio('Has Dependents', ['Select', 'Yes', 'No'])
    LoanPurpose = st.selectbox('Loan Purpose', ['Select', 'Personal', 'Education', 'Home', 'Car', 'Other'])
    HasCoSigner = st.radio('Has CoSigner', ['Select', 'Yes', 'No'])

    # Prepare input data for prediction
    data = {
        'Age': age_category, 'Income': Income, 'LoanAmount': LoanAmount, 'CreditScore': creditscore_category,
        'MonthsEmployed': MonthsEmployed, 'NumCreditLines': NumCreditLines, 'InterestRate': InterestRate,
        'LoanTerm': LoanTerm, 'DTIRatio': DTIRatio,
        # Handle the 'Select' option for categorical inputs; they should not contribute to dummy variables
        **{f'Education_{Education}': 1 if Education != 'Select' else 0},
        **{f'EmploymentType_{EmploymentType}': 1 if EmploymentType != 'Select' else 0},
        **{f'MaritalStatus_{MaritalStatus}': 1 if MaritalStatus != 'Select' else 0},
        **{f'HasMortgage_{HasMortgage}': 1 if HasMortgage == 'Yes' else 0},
        **{f'HasDependents_{HasDependents}': 1 if HasDependents == 'Yes' else 0},
        **{f'LoanPurpose_{LoanPurpose}': 1 if LoanPurpose != 'Select' else 0},
        **{f'HasCoSigner_{HasCoSigner}': 1 if HasCoSigner == 'Yes' else 0},
    }

    #Define function for preprocessing input data        
    def preprocess_input(data):
        #Combine user inputs into a DataFrame
        df = pd.DataFrame(data, index=[0])
        #Perform one-hot encoding for categorical variables
        categorical_cols = ['AgeCategory', 'CreditScoreCategory', 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
        input_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype='int')
        return input_encoded

    #Make prediction function
    def predict_defaulters(data):
        # Preprocess input features
        processed_input=preprocess_input(data)

        # Make prediction
        prediction = model.predict(processed_input)
        return prediction

    # Display prediction result
    if st.button('Predict'):
        prediction = predict_defaulters(data)
        st.write(f"Predicted Loan Default Status: {prediction}")

if __name__ == "__main__":
    main()
