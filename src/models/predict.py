import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from src.config.config import DATABASE_URL

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/preprocessing/preprocessed_data.csv"
TABLE_NAME = "predictions"


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    sk_id = df["SK_ID_CURR"].astype(int)
    target = df["TARGET"] if "TARGET" in df.columns else None

    drop_cols = [c for c in ["SK_ID_CURR", "TARGET"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    probas = model.predict_proba(X)
    classes = model.predict(X)

    result = pd.DataFrame({
        "sk_id_curr": sk_id.values,
        "predicted_class": classes,
        "proba_class_0": probas[:, 0],
        "proba_class_1": probas[:, 1],
    })

    if target is not None:
        result["true_class"] = target.values

    return result


def predict_onnx(onnx_path: str, df: pd.DataFrame) -> pd.DataFrame:
    import onnxruntime as ort

    sk_id = df["SK_ID_CURR"].astype(int)
    target = df["TARGET"] if "TARGET" in df.columns else None
    drop_cols = [c for c in ["SK_ID_CURR", "TARGET"] if c in df.columns]
    X = df.drop(columns=drop_cols).values.astype(np.float32)

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X})

    labels = outputs[0]
    prob_maps = outputs[1]
    proba_0 = np.array([m[0] for m in prob_maps])
    proba_1 = np.array([m[1] for m in prob_maps])

    result = pd.DataFrame({
        "sk_id_curr": sk_id.values,
        "predicted_class": labels,
        "proba_class_0": proba_0,
        "proba_class_1": proba_1,
    })

    if target is not None:
        result["true_class"] = target.values

    return result


def save_to_database(df: pd.DataFrame, table: str = TABLE_NAME) -> None:
    engine = create_engine(DATABASE_URL)
    df.to_sql(table, engine, if_exists="replace", index=False)
    print(f"{len(df)} prédictions insérées dans la table '{table}'.")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Calcul des prédictions de crédit")
    parser.add_argument(
        "--engine",
        choices=["catboost", "onnx"],
        default="catboost",
        help="Moteur d'inférence à utiliser (défaut : catboost)",
    )
    args = parser.parse_args()

    print("Chargement des données préprocessées...")
    df = pd.read_csv(DATA_PATH)

    if args.engine == "onnx":
        ONNX_PATH = "models/model.onnx"
        print(f"Calcul des prédictions avec ONNX Runtime ({ONNX_PATH})...")
        predictions = predict_onnx(ONNX_PATH, df)
    else:
        print("Chargement du modèle CatBoost...")
        model = load_model(MODEL_PATH)
        print("Calcul des prédictions avec CatBoost...")
        predictions = predict(model, df)

    print(predictions.head())
    print("Sauvegarde dans Supabase...")
    save_to_database(predictions)
