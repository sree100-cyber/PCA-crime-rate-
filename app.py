import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Communities Crime Prediction App")

with open("rf_model.pkl","rb") as f:
    model = pickle.load(f)
with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)
with open("pca.pkl","rb") as f:
    pca = pickle.load(f)
with open("kmeans.pkl","rb") as f:
    kmeans = pickle.load(f)
with open("feature_columns.pkl","rb") as f:
    feature_columns = pickle.load(f)


st.header("Enter feature values:")
user_input = {}
for col in feature_columns:
    
    if '_' in col:  
        continue
    user_input[col] = st.text_input(f"{col}", "0")


input_df = pd.DataFrame([user_input])


input_encoded = pd.get_dummies(input_df)
for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_columns]


X_scaled = scaler.transform(input_encoded)
X_pca = pca.transform(X_scaled)
cluster_label = kmeans.predict(X_pca)[0]
X_final = pd.DataFrame(X_pca, columns=['PC1','PC2'])
X_final['Cluster'] = cluster_label

prediction = model.predict(X_final)[0]
st.success(f"Predicted ViolentCrimesPerPop: {prediction:.4f}")
