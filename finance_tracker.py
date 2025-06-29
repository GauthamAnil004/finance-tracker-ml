import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------ Data Simulation ------------------------
st.title("Personal Finance Tracker with ML")
st.subheader("Simulated Transaction Data")

data = {
    'Date': pd.date_range(start='2023-01-01', periods=20, freq='M'),
    'Amount': [200, 500, 100, 450, 700, 150, 250, 300, 500, 650, 230, 120,200,150,180,900,1000,500,500,600],
    'Description': [
        'Grocery Store', 'Electric Bill', 'Coffee Shop', 'Clothing',
        'Hospital', 'Bus Ticket', 'Online Shopping', 'Fuel',
        'Gym Membership', 'Restaurant', 'Medicine', 'Train Ticket'
        ,'Biriyani','Fried Rice','Noodles','KFC','Pizza','Uber ride','Taxi','Pills'
        
    ],
    'Category': [
        'Food', 'Bills', 'Food', 'Shopping', 'Health', 'Transport',
        'Shopping', 'Transport', 'Health', 'Food', 'Health', 'Transport',
        'Food','Food','Food','Food','Food','Transport','Transport','Health'
    ]
}
df = pd.DataFrame(data)
df['Month'] = df['Date'].dt.month

# ------------------------ Linear Regression (Expense Prediction) ------------------------
X = df[['Month']]
y = df['Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
predicted_expense = lr.predict([[13]])[0]  # Predict for next month

# ------------------------ Random Forest (Categorization) ------------------------
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Category'])
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(df['Description'])
rf = RandomForestClassifier()
rf.fit(X_text, df['Label'])

# ------------------------ Isolation Forest (Anomaly Detection) ------------------------
iso = IsolationForest(contamination=0.2)
df['Anomaly'] = iso.fit_predict(df[['Amount']])

# ------------------------ Streamlit Dashboard ------------------------
st.subheader("Transaction Table")
st.dataframe(df[['Date', 'Description', 'Amount', 'Category']])

st.subheader("Predicted Expense for Next Month")
st.success(f"₹ {predicted_expense:.2f}")

st.subheader("Anomalies Detected")
anomalies = df[df['Anomaly'] == -1]
st.write(anomalies[['Date', 'Description', 'Amount']])

st.subheader("Test Transaction Categorization")
desc = st.text_input("Enter a transaction description:")
if st.button("Categorize"):
    if desc.strip():
        X_new = vectorizer.transform([desc])
        label = rf.predict(X_new)
        st.info(f"Predicted Category: {le.inverse_transform(label)[0]}")
