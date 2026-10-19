# ===== CELLULE À AJOUTER À LA FIN DU NOTEBOOK (Colab) =====
import json, joblib, sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from google.colab import files

# 1) Variables finales (sans discharge_disposition_id)
Xf = X_sans_sortie.copy()

cols_cat_voulues = ["gender", "race", "admission_type_id", "admission_source_id",
                    "medical_specialty", "diag_1_family", "diag_2_family", "diag_3_family"]
cols_cat = [c for c in Xf.columns
            if c in cols_cat_voulues or not pd.api.types.is_numeric_dtype(Xf[c])]
cols_num = [c for c in Xf.columns if c not in cols_cat]
# Tout en texte pour que le formulaire et le modèle parlent le même langage
Xf[cols_cat] = Xf[cols_cat].fillna("Inconnu").astype(str)

# 2) Préprocesseur reconstruit sur les bonnes colonnes
prep = ColumnTransformer([
    ("numerique", Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler())]), cols_num),
    ("categoriel", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cols_cat),
])

modele_final = Pipeline([("preparation", prep),
                         ("classification", LogisticRegression(max_iter=2000))])

Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.20, random_state=42, stratify=y)
modele_final.fit(Xtr, ytr)
print("Accuracy test :", round(modele_final.score(Xte, yte), 4))

# 3) Sauvegarde du modèle + infos pour construire le formulaire
meta = {
    "ordre_colonnes": list(Xf.columns),
    "numeriques": {c: {"min": float(Xf[c].min()), "max": float(Xf[c].max()),
                       "mediane": float(Xf[c].median())} for c in cols_num},
    "categorielles": {c: sorted(Xf[c].dropna().astype(str).unique().tolist()) for c in cols_cat},
}
joblib.dump(modele_final, "modele_final.joblib")
with open("meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Version scikit-learn :", sklearn.__version__)
files.download("modele_final.joblib")
files.download("meta.json")
