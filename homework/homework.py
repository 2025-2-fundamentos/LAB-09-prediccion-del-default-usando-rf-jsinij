# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/" (o en "files/grading/" según el entorno).
#

import os
import json
import gzip
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def _load_data():
    """
    Carga x_train, y_train, x_test, y_test desde files/grading/ o files/input/
    dependiendo de qué exista.
    """
    if os.path.exists("files/grading/x_train.pkl"):
        base_dir = "files/grading"
    else:
        base_dir = "files/input"

    x_train = pd.read_pickle(os.path.join(base_dir, "x_train.pkl"))
    y_train = pd.read_pickle(os.path.join(base_dir, "y_train.pkl"))
    x_test = pd.read_pickle(os.path.join(base_dir, "x_test.pkl"))
    y_test = pd.read_pickle(os.path.join(base_dir, "y_test.pkl"))

    return x_train, y_train, x_test, y_test


def pregunta_01():
    """
    Entrena un modelo de Random Forest, lo guarda en
    files/models/model.pkl.gz y genera files/output/metrics.json
    con métricas y matrices de confusión.
    """

    # ---------- Paso 1: cargar datos ----------
    df_x_tr, df_y_tr, df_x_te, df_y_te = _load_data()

    # Normalizar nombre de la columna target en y_train
    if isinstance(df_y_tr, pd.DataFrame):
        if "default payment next month" in df_y_tr.columns:
            df_y_tr = df_y_tr.rename(
                columns={"default payment next month": "default"}
            )
            df_y_tr = df_y_tr["default"]
    else:
        df_y_tr.name = "default"

    # Normalizar nombre de la columna target en y_test
    if isinstance(df_y_te, pd.DataFrame):
        if "default payment next month" in df_y_te.columns:
            df_y_te = df_y_te.rename(
                columns={"default payment next month": "default"}
            )
            df_y_te = df_y_te["default"]
    else:
        df_y_te.name = "default"

    # Quitar columna target duplicada en X si viene allí
    if "default payment next month" in df_x_tr.columns:
        df_x_tr = df_x_tr.drop(columns=["default payment next month"])
    if "default payment next month" in df_x_te.columns:
        df_x_te = df_x_te.drop(columns=["default payment next month"])

    # Quitar ID
    if "ID" in df_x_tr.columns:
        df_x_tr = df_x_tr.drop(columns=["ID"])
    if "ID" in df_x_te.columns:
        df_x_te = df_x_te.drop(columns=["ID"])

    # EDUCATION > 4 -> 4 ("others")
    if "EDUCATION" in df_x_tr.columns:
        df_x_tr["EDUCATION"] = df_x_tr["EDUCATION"].apply(
            lambda n: 4 if n > 4 else n
        )
    if "EDUCATION" in df_x_te.columns:
        df_x_te["EDUCATION"] = df_x_te["EDUCATION"].apply(
            lambda n: 4 if n > 4 else n
        )

    # Unir X e y y eliminar filas con nulos
    df_train = pd.concat([df_x_tr, df_y_tr], axis=1).dropna()
    df_test = pd.concat([df_x_te, df_y_te], axis=1).dropna()

    X_tr = df_train.drop(columns=["default"])
    y_tr = df_train["default"]

    X_te = df_test.drop(columns=["default"])
    y_te = df_test["default"]

    # ---------- Paso 3: pipeline con preprocesador + RandomForest ----------

    # Variables categóricas
    cat_cols = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X_tr.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough",
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(random_state=58082)),
        ]
    )

    # ---------- Paso 4: GridSearchCV con balanced_accuracy y cv=10 ----------

    param_grid = {
        "model__n_estimators": [500],
        "model__max_depth": [22],
        "model__min_samples_split": [2],
        "model__min_samples_leaf": [1],
        "model__class_weight": [None],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    grid.fit(X_tr, y_tr)

    # ---------- Paso 5: guardar el modelo comprimido ----------

    model_dir = "files/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl.gz")

    with gzip.open(model_path, "wb") as f:
        pickle.dump(grid, f)

    # ---------- Paso 6: métricas de train y test ----------

    os.makedirs("files/output", exist_ok=True)
    results = []

    def compute_metrics(X, y, dataset_name):
        preds = grid.predict(X)
        return {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": float(precision_score(y, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "recall": float(recall_score(y, preds)),
            "f1_score": float(f1_score(y, preds)),
        }

    results.append(compute_metrics(X_tr, y_tr, "train"))
    results.append(compute_metrics(X_te, y_te, "test"))

    # ---------- Paso 7: matrices de confusión ----------

    def cm_to_dict(cm, dataset_name):
        return {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {
                "predicted_0": int(cm[0, 0]),
                "predicted_1": int(cm[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm[1, 0]),
                "predicted_1": int(cm[1, 1]),
            },
        }

    cm_train = confusion_matrix(y_tr, grid.predict(X_tr))
    cm_test = confusion_matrix(y_te, grid.predict(X_te))

    results.append(cm_to_dict(cm_train, "train"))
    results.append(cm_to_dict(cm_test, "test"))

    # Guardar metrics.json, una línea por dict
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


# Ejecutar automáticamente al importar el módulo
pregunta_01()
