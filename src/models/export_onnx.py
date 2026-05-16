import pickle
from typing import Any


def _find_savable_model(obj: Any):
    """Return an object that implements `save_model`, searching inside
    sklearn-style pipelines if necessary."""
    if hasattr(obj, "save_model"):
        return obj

    try:
        from sklearn.pipeline import Pipeline
    except Exception:
        Pipeline = None

    if Pipeline is not None and isinstance(obj, Pipeline):
        final = obj.steps[-1][1]
        if hasattr(final, "save_model"):
            return final

    for name in ("estimator", "final_estimator", "clf", "model"):
        candidate = getattr(obj, name, None)
        if candidate is not None and hasattr(candidate, "save_model"):
            return candidate

    return None


def export_to_onnx(pkl_path: str, onnx_path: str) -> None:
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    savable = _find_savable_model(model)
    if savable is None:
        raise AttributeError(
            "No object with `save_model` found in the loaded pickle. "
            "If your model is a scikit-learn Pipeline, ensure the final "
            "estimator is a CatBoost model (has `save_model`)."
        )

    savable.save_model(onnx_path, format="onnx")
    print(f"Modèle exporté vers {onnx_path}")


if __name__ == "__main__":
    export_to_onnx("models/model.pkl", "models/model.onnx")
