import streamlit as st, joblib, numpy as np

st.title("Salary Estimation App")

st.divider()

years_at_company = st.number_input("Enter years at company", min_value= 0.0, max_value=20.0)
satisfaction_level = st.number_input("Satisfaction Level", min_value= 0.0)
average_monthly_hours = st.number_input("Average Monthly Hours", min_value=120.0, max_value=400.0)

X = [years_at_company, satisfaction_level, average_monthly_hours]

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

predict_button = st.button('Press for predicting the salary')

st.divider()

if predict_button:
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    prediction = model.predict(X_array)[0]
    st.write(f'Salary prediction is {prediction}')

else:
    st.write('Please enter the values and press to the predict button')
