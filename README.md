## Communities Crime Prediction
## Project Overview
app:https://mm5akwq69sqqs8txclfhdv.streamlit.app/

This project predicts the Violent Crime Rate per Population (ViolentCrimesPerPop) for communities in the United States using the UCI Communities and Crime dataset. It leverages a RandomForest regression model with preprocessing techniques, PCA, and clustering to improve predictions.

## Dataset

Source: UCI Machine Learning Repository – Communities and Crime

Description: The dataset contains 128 attributes for 1994 communities in the U.S., including demographic, socioeconomic, law enforcement, and crime data.

Features: 127 features (numeric & categorical) + 1 target (ViolentCrimesPerPop)

## Project Structure
project/
│
├─ train_model.py       # Script to preprocess data, train model, and save objects
├─ app.py               # Streamlit app to make predictions
├─ communities.data     # Dataset file
├─ communities.names    # Dataset feature names
├─ rf_model.pkl         # Trained RandomForest model (saved after training)
├─ scaler.pkl           # StandardScaler object (saved after training)
├─ pca.pkl              # PCA object (saved after training)
├─ kmeans.pkl           # KMeans clustering object (saved after training)
├─ feature_columns.pkl  # List of columns after one-hot encoding
└─ README.md            # Project documentation

## Preprocessing Steps

Missing Values: Columns with >30% missing values are removed; remaining missing values filled with median.

Categorical Encoding: All categorical features are one-hot encoded.

Outlier Handling: Numeric features capped using IQR method.

Skew Reduction: Numeric features with high skewness are transformed using Yeo-Johnson power transform.

Scaling: StandardScaler applied to all features.

Dimensionality Reduction: PCA reduces features to 2 principal components.

Clustering: KMeans clustering (3 clusters) added as a feature.

## Model Training

Algorithm: RandomForest Regressor

Hyperparameter Tuning: GridSearchCV with 5-fold cross-validation

Saved Objects:

RandomForest model (rf_model.pkl)

Scaler (scaler.pkl)

PCA (pca.pkl)

## Conclusion

This project demonstrates how to build a predictive model for violent crime rates in U.S. communities using the Communities and Crime dataset. By combining data preprocessing, one-hot encoding for categorical features, outlier handling, skew reduction, scaling, PCA, and clustering, we were able to effectively prepare the dataset for modeling.

The RandomForest Regressor was trained with hyperparameter tuning and achieved good predictive performance on unseen data. The use of PCA and KMeans clustering helped reduce dimensionality and capture underlying patterns in the dataset.

The Streamlit app provides an interactive interface where users can input community features and receive an instant prediction for ViolentCrimesPerPop.

KMeans (kmeans.pkl)

Feature columns (feature_columns.pkl)
