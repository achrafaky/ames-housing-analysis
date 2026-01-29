import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# --- ÉTAPE 1 : CHARGEMENT DES DONNÉES ---
# Assure-toi que ton fichier s'appelle bien 'ames_housing.csv' et qu'il est dans le même dossier
try:
    data = pd.read_csv('ames.csv')
    print("✅ Fichier chargé avec succès.")
except FileNotFoundError:
    print("❌ ERREUR : Le fichier 'ames_housing.csv' est introuvable.")
    print("Assure-toi que le fichier est bien dans le dossier ouvert dans VS Code.")
    exit() # On arrête le script si pas de fichier

# --- ÉTAPE 2 : NETTOYAGE (CLEANING) ---
# On enlève les lignes où il n'y a pas de prix (Target manquante)
data = data.dropna(subset=['SalePrice'])

# Séparation des Features (X) et de la Cible (y)
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Remplissage des trous (Missing Values)
# Pour les colonnes numériques -> on met la médiane
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Pour les colonnes texte -> on met le mode (la valeur la plus fréquente)
cat_cols = X.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

print("✅ Nettoyage terminé (trous remplis).")

# --- ÉTAPE 3 : ENCODAGE (TEXTE -> NOMBRES) ---
# C'est l'étape cruciale pour que le MI fonctionne sur les catégories
print("🔄 Encodage des catégories en cours...")
for colname in X.select_dtypes(include=['object']):
    X[colname], _ = X[colname].factorize()

# --- ÉTAPE 4 : CALCUL DE L'INFORMATION MUTUELLE (MI) ---
# On identifie les colonnes qui sont des entiers (discrètes)
discrete_features = X.dtypes == int

def make_mi_scores(X, y, discrete_features):
    # Le calcul mathématique se fait ici
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

print("⏳ Calcul des scores MI en cours (ça peut prendre quelques secondes)...")
mi_scores = make_mi_scores(X, y, discrete_features)

print("\n🏆 TOP 10 DES MEILLEURES FEATURES :")
print(mi_scores.head(10))

# --- ÉTAPE 5 : VISUALISATION ---
def plot_mi_scores(scores):
    plt.figure(figsize=(10, 8))
    # On trie pour l'affichage graphique
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    
    # Création du bar plot horizontal
    plt.barh(width, scores, color="#1f77b4")
    plt.yticks(width, ticks)
    plt.title("Scores d'Information Mutuelle (Importance des Variables)")
    plt.xlabel("Score MI")

# On affiche seulement le Top 20 pour que le graphique soit lisible
plot_mi_scores(mi_scores.head(20))
plt.show()

