import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. data loading ---
file_path = "customer_dataset/marketing_campaign.csv" # change the path if nessasary
try:
    data = pd.read_csv(file_path, sep='\t')
    print(f"✅ Loaded successfully，data: {len(data)}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# --- 2. Data cleansing and handling missing value ---
# income imputation
data['Income'] = data['Income'].fillna(data['Income'].median())

data = data.dropna()

# --- 3. Feature Engineering ---
#  (Customer_For)
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
reference_date = data['Dt_Customer'].max()
data["Customer_For"] = (reference_date - data['Dt_Customer']).dt.days

# Calculate age
data["Age"] = reference_date.year - data["Year_Birth"]

# (Spent)
spent_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
data["Spent"] = data[spent_cols].sum(axis=1)

# Simplify Marrital status
def living_with(status):
    return 'Couple' if status in ['Married', 'Together'] else 'Alone' if status in ['Single', 'Widow', 'Divorced', 'Alone'] else 'Other'

data["Living_With"] = data["Marital_Status"].apply(living_with)
data["Children"] = data["Kidhome"] + data["Teenhome"]
data["Family_Size"] = data["Children"] + data["Marital_Status"].apply(lambda x: 2 if x in ['Married', 'Together'] else 1)
data["Is_Parent"] = np.where(data["Children"] > 0, 1, 0)

# Simplify edu level
def simplify_education(ed):
    return 'Undergraduate' if ed in ['Basic', '2n Cycle'] else 'Graduate'
data["Education_Simplified"] = data["Education"].apply(simplify_education)

# --- 4. label and normalize ---

le = LabelEncoder()
data["Living_With_Encoded"] = le.fit_transform(data["Living_With"])
data["Education_Encoded"] = le.fit_transform(data["Education_Simplified"])

# feature selection and normalize
features_to_scale = ["Income", "Age", "Spent", "Customer_For", "Children", "Family_Size", "Recency"]
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# --- 5. (Isolation Forest) ---

iso_forest = IsolationForest(contamination=0.01, random_state=42)
data["Outlier_IF"] = iso_forest.fit_predict(data_scaled[features_to_scale])

remove outlier
data_clean = data[data["Outlier_IF"] == 1]
print(f"🚀 After removed : {len(data_clean)} (removed {len(data) - len(data_clean)} samples)")

# save into gihub repo
data_clean.to_csv("customer_dataset/cleaned_customer_data.csv", index=False)
print("CSV saved。")