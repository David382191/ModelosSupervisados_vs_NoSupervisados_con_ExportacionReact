# streamlit_app.py
"""
Streamlit app para la tarea:
- Supervisado: Bernoulli Naive Bayes (trabaja con features binarizadas).
- No supervisado: MiniBatchKMeans.
- Exporta JSON con métricas y predicción actual.
- Guarda modelos .pkl y permite descargar.
- Dataset por defecto: Iris/Wine/Breast Cancer (se binarizan para BernoulliNB).
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import json
from io import BytesIO

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt


#################################3 Esas fueron las importanciones que necesitaremeos más tarde.

####### Ahora vamos con algo más.

st.set_page_config("Tarea ML - BernoulliNB + MiniBatchKMeans", layout="wide")

# ---------- Helpers ----------
def save_pickle(obj):
    return pickle.dumps(obj)

def save_to_disk(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def bytes_for_download(obj_bytes, filename):
    return obj_bytes

def export_json_to_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

# ---------- Sidebar: configuración ----------
st.sidebar.title("Configuración")
mode = st.sidebar.radio("Modo", ["Supervisado (BernoulliNB)", "No Supervisado (MiniBatchKMeans)"])
dataset_choice = st.sidebar.selectbox("Dataset ejemplo", ["Iris", "Wine", "Breast Cancer"])
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))
test_size_pct = int(st.sidebar.slider("Test size (%) (supervisado)", 10, 50, 20))
n_clusters = int(st.sidebar.slider("Número de clusters (no supervisado)", 2, 8, 3))
batch_size = int(st.sidebar.number_input("MiniBatch size (no supervisado)", value=32, min_value=8, max_value=1024, step=8))

# ---------- Load dataset ----------
def load_dataset(name):
    if name == "Iris":
        d = datasets.load_iris(as_frame=True)
    elif name == "Wine":
        d = datasets.load_wine(as_frame=True)
    else:
        d = datasets.load_breast_cancer(as_frame=True)
    X = d.data
    y = d.target
    feature_names = list(X.columns)
    target_names = list(map(str, d.target_names))
    return X, y, feature_names, target_names

X_raw, y_raw, feature_names, target_names = load_dataset(dataset_choice)

st.title("Tarea: Bernoulli Naive Bayes + MiniBatchKMeans")
st.markdown("App lista para entrenar, validar, exportar JSON y descargar modelos `.pkl`.")


################################
# show data preview
with st.expander("Ver dataset (primeras filas)"):
    df_preview = X_raw.copy()
    df_preview["target"] = y_raw
    st.dataframe(df_preview.head())

os.makedirs("artifacts", exist_ok=True)

################################
# ---------- Supervisado ----------
if mode.startswith("Supervisado"):
    st.header("Modo Supervisado — Bernoulli Naive Bayes")
    st.write("Dataset:", dataset_choice)
    st.markdown("**Nota:** BernoulliNB trabaja con datos binarios. Aquí binarizamos cada feature por su mediana por defecto (opcional ajustar umbral).")

    # choose binarization strategy
    median_threshold = st.checkbox("Binarizar por mediana (recomendado)", value=True)
    custom_thresh = None
    if not median_threshold:
        # allow entering thresholds
        st.write("Puedes introducir umbrales por feature (opcional):")
        custom_thresh = {}
        for f in feature_names:
            v = st.number_input(f"Umbral para '{f}'", value=float(X_raw[f].median()))
            custom_thresh[f] = v

    # split
    if st.button("Entrenar BernoulliNB"):
        X = X_raw.copy()
        if median_threshold:
            threshold_vec = X.median().values.reshape(1, -1)
            bin = Binarizer(threshold=0.0)  # we will compare manually to medians
            X_bin = (X.values > threshold_vec).astype(int)
        else:
            # use custom thresholds
            thresh_arr = np.array([custom_thresh[f] for f in feature_names]).reshape(1, -1)
            X_bin = (X.values > thresh_arr).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X_bin, y_raw.values, test_size=test_size_pct/100.0, random_state=random_state, stratify=y_raw)
        clf = BernoulliNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        st.subheader("Métricas")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision (weighted)", f"{prec:.4f}")
        col3.metric("Recall (weighted)", f"{rec:.4f}")
        col4.metric("F1-score (weighted)", f"{f1:.4f}")

        # interactive prediction (manual input)
        st.subheader("Prueba interactiva (entrada binaria por feature)")
        st.write("Marca los checkboxes para indicar presencia (1) o ausencia (0) de cada feature en el ejemplo a predecir.")
        user_input = []
        cols = st.columns(3)
        for i, f in enumerate(feature_names):
            c = cols[i % 3]
            val = c.checkbox(f"presente: {f}", value=False, key=f"ui_{f}")
            user_input.append(1 if val else 0)
        pred_class = int(clf.predict([user_input])[0])
        pred_label = target_names[pred_class]
        st.success(f"Clase predicha: {pred_class} -> {pred_label}")

        # save models
        model_bytes = save_pickle(clf)
        model_filename = f"artifacts/bernoulli_{dataset_choice.lower()}.pkl"
        save_to_disk(clf, model_filename)

        # prepare JSON
        results = {
            "model_type": "Supervised",
            "model_name": "Bernoulli Naive Bayes",
            "dataset": dataset_choice,
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            },
            "current_prediction": {
                "input": user_input,
                "output_class": pred_class,
                "output_label": pred_label
            },
            "model_file": os.path.basename(model_filename)
        }
        json_bytes = export_json_to_bytes(results)
        st.download_button("Descargar resultados JSON", data=json_bytes, file_name=f"classification_results_{dataset_choice.lower()}.json", mime="application/json")
        st.download_button("Descargar modelo (.pkl)", data=model_bytes, file_name=os.path.basename(model_filename), mime="application/octet-stream")
        st.write("Modelo guardado en:", model_filename)



# ---------- No Supervisado ----------
else:
    st.header("Modo No Supervisado — MiniBatchKMeans")
    st.write("Dataset:", dataset_choice)
    st.markdown("Se escalan las features y se aplica MiniBatchKMeans. Visualización mediante PCA 2D.")

    if st.button("Entrenar MiniBatchKMeans"):
        X = X_raw.copy()
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
        labels = kmeans.fit_predict(X_s)

        # metrics
        sil = float(silhouette_score(X_s, labels)) if len(set(labels)) > 1 else None
        db = float(davies_bouldin_score(X_s, labels)) if len(set(labels)) > 1 else None

        st.subheader("Métricas de clustering")
        col1, col2 = st.columns(2)
        col1.metric("Silhouette score", f"{sil:.4f}" if sil is not None else "N/A")
        col2.metric("Davies-Bouldin", f"{db:.4f}" if db is not None else "N/A")

        # save model & scaler
        model_filename = f"artifacts/minibatchkmeans_{dataset_choice.lower()}.pkl"
        scaler_filename = f"artifacts/scaler_{dataset_choice.lower()}.pkl"
        save_to_disk(kmeans, model_filename)
        save_to_disk(scaler, scaler_filename)
        st.write("Modelo guardado:", model_filename)

        # prepare JSON
        results = {
            "model_type": "Unsupervised",
            "algorithm": "MiniBatchKMeans",
            "dataset": dataset_choice,
            "parameters": {
                "n_clusters": int(n_clusters),
                "batch_size": int(batch_size)
            },
            "metrics": {
                "silhouette_score": sil,
                "davies_bouldin": db
            },
            "cluster_labels": labels.tolist(),
            "model_file": os.path.basename(model_filename)
        }
        json_bytes = export_json_to_bytes(results)
        st.download_button("Descargar resultados JSON", data=json_bytes, file_name=f"clustering_results_{dataset_choice.lower()}.json", mime="application/json")

        # visualization: PCA 2D
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(X_s)
        fig, ax = plt.subplots()
        scatter = ax.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab10', s=40)
        ax.set_title("MiniBatchKMeans - PCA 2D")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)

        # download model
        model_bytes = save_pickle(kmeans)
        st.download_button("Descargar modelo (.pkl)", data=model_bytes, file_name=os.path.basename(model_filename), mime="application/octet-stream")

# ---------- Footer ----------
st.sidebar.markdown("---")
st.sidebar.write("Archivos guardados en la carpeta `./artifacts`.")
st.write("Hecho para la tarea: BernoulliNB (supervisado) + MiniBatchKMeans (no supervisado).")
