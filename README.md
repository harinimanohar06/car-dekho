# 🚗 Car Price Prediction Streamlit Application

A Streamlit-based machine learning web application that predicts the estimated selling price of a used car based on multiple features such as fuel type, body type, manufacturer, model, mileage, engine capacity, city, and more.

---

## 📌 Project Overview
This project uses a **Random Forest Regression model** along with several **Label Encoders** and **One‑Hot Encoders** to process categorical inputs.  
The app provides an interactive UI built with **Streamlit**, allowing users to select car attributes and instantly get a predicted price.

---

## ✨ Features
- User-friendly Streamlit interface  
- Dropdown selections for fuel type, body type, manufacturer, model, insurance, and city  
- Numeric inputs for mileage, engine CC, car age, kilometers driven, etc.  
- Encodes categorical variables using pre-trained encoders  
- Predicts car price using a trained Random Forest model  
- Displays results instantly in the UI  

---

## 🧠 Machine Learning Components

The app loads the following pre-trained components:

| Component | Purpose |
|----------|---------|
| `fuel_encoder.pkl` | Encodes fuel type |
| `body_encoder.pkl` | Encodes body type |
| `transmission_encoder.pkl` | Encodes transmission |
| `label_manufacturer_encoder.pkl` | Encodes manufacturer |
| `car_model_encoder.pkl` | Encodes car model |
| `insurance_encoder.pkl` | Encodes insurance type |
| `ohe_encoder.pkl` | One‑hot encodes city |
| `rf_model.pkl` | Random Forest model for price prediction |

---

## 🛠️ Tech Stack
- Python 3.x  
- Streamlit  
- Pandas / NumPy  
- Scikit‑learn  
- Pickle  

---





