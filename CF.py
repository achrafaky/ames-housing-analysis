import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration visuelle
plt.style.use("seaborn-v0_8-whitegrid")

# 1. CHARGEMENT SÉCURISÉ
try:
    df = pd.read_csv("ames.csv")
    print("✅ Dataset chargé avec succès.")
except FileNotFoundError:
    print("❌ Erreur : 'ames.csv' introuvable dans le dossier.")
    exit()

# --- DÉTECTEUR DE COLONNES (Le "Fix" Magique) ---
# On crée un dictionnaire pour mapper les noms Kaggle vers les noms réels du CSV
mapping = {
    "FirstFlrSF": ["1stFlrSF", "First_Flr_SF", "FirstFlrSF"],
    "SecondFlrSF": ["2ndFlrSF", "Second_Flr_SF", "SecondFlrSF"],
    "ThreeSsnPorch": ["3SsnPorch", "Three_Ssn_Porch", "3-Ssn_Porch"]
}

def get_real_col(df, target_names):
    for name in target_names:
        if name in df.columns:
            return name
    return None

# On renomme pour être sûr
col_1st = get_real_col(df, mapping["FirstFlrSF"])
col_2nd = get_real_col(df, mapping["SecondFlrSF"])
col_3ssn = get_real_col(df, mapping["ThreeSsnPorch"])

X = df.copy()
y = X.pop('SalePrice')

# 2. CRÉATION DES FEATURES
print("🚀 Création des features stratégiques...")

X_1 = pd.DataFrame()
X_1["LivLotRatio"] = X["GrLivArea"] / X["LotArea"]

# Utilisation des colonnes détectées dynamiquement
if col_1st and col_2nd:
    X_1["Spaciousness"] = (X[col_1st] + X[col_2nd]) / X["TotRmsAbvGrd"]

# Gestion sécurisée du Porch
porch_list = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch"]
if col_3ssn: porch_list.append(col_3ssn)

X_1["TotalOutsideSF"] = X[porch_list].sum(axis=1)

# Le reste du code (Interactions, Grouping...)
X_2 = pd.get_dummies(X["BldgType"], prefix="Bldg").mul(X["GrLivArea"], axis=0)
X_3 = pd.DataFrame()
X_3["PorchTypes"] = X[porch_list].gt(0).sum(axis=1)

# On combine tout
X_new = X.join([X_1, X_2, X_3])

print(f"🎯 Terminé ! Total features : {X_new.shape[1]}")
X_new.to_csv('ames_engineered.csv', index=False)