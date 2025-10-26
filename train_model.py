import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle


columns = [
    "state", "county", "community", "communityname", "fold",
    "population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp",
    "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up",
    "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc",
    "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc",
    "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap",
    "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore",
    "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu",
    "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv",
    "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par",
    "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg",
    "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10",
    "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10",
    "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup",
    "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
    "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR",
    "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos",
    "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart",
    "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ",
    "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg",
    "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85",
    "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop",
    "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop",
    "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",
    "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor",
    "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked",
    "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg",
    "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn",
    "PolicBudgPerPop", "ViolentCrimesPerPop"
]


df = pd.read_csv("communities.data", names=columns, na_values='?')
print(f"Dataset shape: {df.shape}")
print(df.columns.tolist())

df = df.dropna(axis=1, thresh=int(0.7*len(df)))  # drop cols with >30% NaN
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

X = df.drop('ViolentCrimesPerPop', axis=1)
y = df['ViolentCrimesPerPop']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()


X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


feature_columns = X_encoded.columns.tolist()
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)


for col in numeric_cols:
    if col in X_encoded.columns:
        Q1 = X_encoded[col].quantile(0.25)
        Q3 = X_encoded[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        X_encoded[col] = np.clip(X_encoded[col], lower, upper)


skewed_cols = X_encoded[numeric_cols].skew()[abs(X_encoded[numeric_cols].skew())>0.5].index
pt = PowerTransformer(method='yeo-johnson')
X_encoded[skewed_cols] = pt.fit_transform(X_encoded[skewed_cols])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

X_final = pd.DataFrame(X_pca, columns=['PC1','PC2'])
X_final['Cluster'] = clusters


X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)


param_grid = {'n_estimators':[100,200], 'max_depth':[None,10,20], 'min_samples_split':[2,5]}
rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(" Best R² (CV):", grid.best_score_)
print(" Best Params:", grid.best_params_)
print(" Test R²:", best_model.score(X_test, y_test))


with open("rf_model.pkl","wb") as f:
    pickle.dump(best_model,f)
with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)
with open("pca.pkl","wb") as f:
    pickle.dump(pca,f)
with open("kmeans.pkl","wb") as f:
    pickle.dump(kmeans,f)

print(" Training complete and objects saved!")







