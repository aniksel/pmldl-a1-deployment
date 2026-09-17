# web app: form -> request to the API -> predicted price

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("California Housing")
st.write("Предсказание медианной стоимости жилья в районе Калифорнии.")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Долгота", value=-122.23, format="%.2f")
    latitude = st.number_input("Широта", value=37.88, format="%.2f")
    housing_median_age = st.number_input("Медианный возраст домов", value=41.0)
    total_rooms = st.number_input("Всего комнат", value=880.0)
    total_bedrooms = st.number_input("Всего спален", value=129.0)

with col2:
    population = st.number_input("Население", value=322.0)
    households = st.number_input("Домохозяйств", value=126.0)
    median_income = st.number_input("Медианный доход (десятки тысяч $)", value=8.33)
    ocean_proximity = st.selectbox(
        "Близость к океану",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
    )

if st.button("Предсказать", type="primary"):
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        price = response.json()["predicted_price"]
        st.success(f"Предсказанная стоимость: ${price:,.0f}")
    except requests.RequestException as e:
        st.error(f"API недоступен: {e}")
