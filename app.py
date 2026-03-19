import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("housing.csv")

df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
df = pd.get_dummies(df, columns=["ocean_proximity"])

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

model = RandomForestRegressor()
model.fit(X, y)

st.title("House Price Prediction")

income = st.number_input("Median Income")
rooms = st.number_input("Total Rooms")

if st.button("Predict"):
    sample = X.iloc[0].copy()
    sample["median_income"] = income
    sample["total_rooms"] = rooms

    prediction = model.predict(sample.values.reshape(1, -1))
    st.write("Estimated Price:", prediction[0])