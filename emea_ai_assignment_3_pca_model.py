

#Original file is located at
   # https://colab.research.google.com/drive/1LnL0wVw2jwCpJLO5kAxOWpjCpexp66aH

 %pip install -q ucimlrepo pandas numpy scikit-learn matplotlib seaborn

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

RANDOM_STATE = 42
sns.set_theme()
pd.set_option("display.max_columns", 30)
print("Libraries imported successfully.")

#Download the Wine Quality dataset
wine_repo = fetch_ucirepo(id=186)

X_all = wine_repo.data.features.copy()
y_all = wine_repo.data.targets.copy()

# The UCI dataset has a 'color' feature for red/white wine.
if "color" in X_all.columns:
    red_mask = X_all["color"].astype(str).str.lower().eq("red")
    X_all = X_all.loc[red_mask].drop(columns=["color"]).reset_index(drop=True)
    y_all = y_all.loc[red_mask].reset_index(drop=True)

# Normalize column names and coerce values to numeric where appropriate.
X_all.columns = [str(c).strip().replace(" ", "_") for c in X_all.columns]
X = X_all.apply(pd.to_numeric, errors="coerce")
quality_col = y_all.columns[0]
quality = pd.to_numeric(y_all[quality_col], errors="coerce")

valid = quality.notna()
X = X.loc[valid].reset_index(drop=True)
quality = quality.loc[valid].reset_index(drop=True)

# Binary target: 1 = quality >= 6, 0 = quality < 6
y = (quality >= 6).astype("int64")

print("Dataset shape:", X.shape)
print("Target counts (0 = quality < 6, 1 = quality >= 6):")
display(y.value_counts().sort_index().rename_axis("class").to_frame("count"))
display(X.head())

print("Missing values by feature:")
display(X.isna().sum().to_frame("missing_count"))

print("Feature summary:")
display(X.describe().T)

plt.figure(figsize=(6, 4))
sns.countplot(x=y.map({0: "Quality < 6", 1: "Quality >= 6"}))
plt.title("Wine Quality Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Impute and scale first, fitted only on training data.
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

pca_full = PCA()
pca_full.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)

print("Original feature count:", X_train.shape[1])
print("Components needed for >=95% training variance:", n_components_95)
print("Cumulative explained variance:", round(cumulative_variance[n_components_95 - 1], 4))

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.axhline(0.95, linestyle="--", label="95% variance threshold")
plt.axvline(n_components_95, linestyle="--", label=f"{n_components_95} components")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Explained Variance (Training Data)")
plt.legend()
plt.tight_layout()
plt.show()

pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("PCA training shape:", X_train_pca.shape)
print("PCA test shape:", X_test_pca.shape)

#training
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gaussian Naive Bayes": GaussianNB()
}

def score_model(model, Xtr, Xte, ytr, yte, model_name, feature_set):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "Model": model_name,
        "Features": feature_set,
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, zero_division=0),
        "Recall": recall_score(yte, pred, zero_division=0),
        "F1-score": f1_score(yte, pred, zero_division=0)
    }

results = []
fitted_models = {}

for name, estimator in models.items():
    # Standardized original features
    model_original = estimator
    results.append(score_model(
        model_original, X_train_scaled, X_test_scaled,
        y_train, y_test, name, "Standardized original"
    ))
    fitted_models[(name, "Standardized original")] = model_original

    # PCA features; use a fresh estimator for this run
    if name == "Logistic Regression":
        pca_estimator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    elif name == "KNN":
        pca_estimator = KNeighborsClassifier(n_neighbors=5)
    else:
        pca_estimator = GaussianNB()

    results.append(score_model(
        pca_estimator, X_train_pca, X_test_pca,
        y_train, y_test, name, "PCA"
    ))
    fitted_models[(name, "PCA")] = pca_estimator

results_df = pd.DataFrame(results).sort_values(
    ["Features", "F1-score"], ascending=[True, False]
).reset_index(drop=True)
display(results_df.round(4))

#Visualizing
for metric in ["Accuracy", "F1-score"]:
    pivot = results_df.pivot(index="Model", columns="Features", values=metric)
    ax = pivot.plot(kind="bar", figsize=(9, 4))
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Classifier")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.show()

#Confusion matrix and classification report
best_row = results_df.loc[results_df["F1-score"].idxmax()]
best_name = best_row["Model"]
best_feature_set = best_row["Features"]
best_model = fitted_models[(best_name, best_feature_set)]

if best_feature_set == "PCA":
    X_eval = X_test_pca
else:
    X_eval = X_test_scaled

best_pred = best_model.predict(X_eval)

print("Best configuration by test F1-score in this run:")
display(best_row.to_frame("value"))

print("\nClassification report:")
print(classification_report(
    y_test, best_pred,
    labels=[0, 1],
    target_names=["Quality < 6", "Quality >= 6"],
    zero_division=0
))

ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred,
    display_labels=["Quality < 6", "Quality >= 6"],
    cmap="Blues",
    values_format="d"
)
plt.title(f"Confusion Matrix — {best_name} ({best_feature_set})")
plt.tight_layout()
plt.show()
