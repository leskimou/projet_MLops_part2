import pickle
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from src.models.predict import predict, load_model, save_to_database


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0])
    model.predict_proba.return_value = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1],
    ])
    return model


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "SK_ID_CURR": [100001, 100002, 100003],
        "feature_a": [1.0, 2.0, 3.0],
        "feature_b": [4.0, 5.0, 6.0],
    })


def test_predict_returns_correct_columns(mock_model, sample_df):
    result = predict(mock_model, sample_df)
    assert set(result.columns) == {"sk_id_curr", "predicted_class", "proba_class_0", "proba_class_1"}


def test_predict_correct_class_values(mock_model, sample_df):
    result = predict(mock_model, sample_df)
    assert list(result["predicted_class"]) == [0, 1, 0]


def test_predict_preserves_sk_id(mock_model, sample_df):
    result = predict(mock_model, sample_df)
    assert list(result["sk_id_curr"]) == [100001, 100002, 100003]


def test_predict_probas_sum_to_one(mock_model, sample_df):
    result = predict(mock_model, sample_df)
    sums = result["proba_class_0"] + result["proba_class_1"]
    assert all(abs(s - 1.0) < 1e-6 for s in sums)


def test_predict_with_target_includes_true_class(mock_model):
    df = pd.DataFrame({
        "SK_ID_CURR": [100001, 100002],
        "TARGET": [0, 1],
        "feature_a": [1.0, 2.0],
    })
    mock_model.predict.return_value = np.array([0, 1])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

    result = predict(mock_model, df)

    assert "true_class" in result.columns
    assert list(result["true_class"]) == [0, 1]


def test_predict_without_target_excludes_true_class(mock_model, sample_df):
    result = predict(mock_model, sample_df)
    assert "true_class" not in result.columns


def test_predict_drops_sk_id_from_features(mock_model, sample_df):
    predict(mock_model, sample_df)
    features_passed = mock_model.predict.call_args[0][0]
    assert "SK_ID_CURR" not in features_passed.columns


def test_load_model_charge_le_fichier(tmp_path):
    model_obj = {"type": "fake_model"}
    model_file = tmp_path / "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model_obj, f)

    result = load_model(str(model_file))

    assert result == model_obj


def test_save_to_database_appelle_to_sql(sample_df):
    mock_engine = MagicMock()
    with patch("src.models.predict.create_engine", return_value=mock_engine):
        with patch.object(sample_df.__class__, "to_sql") as mock_to_sql:
            save_to_database(sample_df, table="predictions")
            mock_to_sql.assert_called_once_with(
                "predictions", mock_engine, if_exists="replace", index=False
            )


def test_predict_drops_target_from_features(mock_model):
    df = pd.DataFrame({
        "SK_ID_CURR": [100001],
        "TARGET": [0],
        "feature_a": [1.0],
    })
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

    predict(mock_model, df)

    features_passed = mock_model.predict.call_args[0][0]
    assert "TARGET" not in features_passed.columns


from src.models.export_onnx import export_to_onnx


def test_export_to_onnx_cree_un_fichier_onnx(tmp_path):
    from catboost import CatBoostClassifier
    import numpy as np

    X = np.random.rand(50, 3).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    model = CatBoostClassifier(iterations=5, verbose=0)
    model.fit(X, y)

    pkl_path = str(tmp_path / "model.pkl")
    onnx_path = str(tmp_path / "model.onnx")

    import pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    export_to_onnx(pkl_path, onnx_path)

    assert (tmp_path / "model.onnx").exists()
    assert (tmp_path / "model.onnx").stat().st_size > 0


from src.models.predict import predict_onnx


def test_predict_onnx_retourne_les_memes_colonnes(tmp_path):
    from catboost import CatBoostClassifier
    import numpy as np
    import pickle
    from src.models.export_onnx import export_to_onnx

    X = np.random.rand(30, 3).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    model = CatBoostClassifier(iterations=5, verbose=0)
    model.fit(X, y)

    pkl_path = str(tmp_path / "model.pkl")
    onnx_path = str(tmp_path / "model.onnx")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    export_to_onnx(pkl_path, onnx_path)

    df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    df.insert(0, "SK_ID_CURR", range(30))

    result = predict_onnx(onnx_path, df)

    assert set(result.columns) == {"sk_id_curr", "predicted_class", "proba_class_0", "proba_class_1"}


def test_predict_onnx_probabilites_proches_de_catboost(tmp_path):
    from catboost import CatBoostClassifier
    import numpy as np
    import pickle
    from src.models.export_onnx import export_to_onnx
    from src.models.predict import predict

    np.random.seed(42)
    X = np.random.rand(30, 3).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    model = CatBoostClassifier(iterations=5, verbose=0)
    model.fit(X, y)

    pkl_path = str(tmp_path / "model.pkl")
    onnx_path = str(tmp_path / "model.onnx")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    export_to_onnx(pkl_path, onnx_path)

    df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    df.insert(0, "SK_ID_CURR", range(30))

    result_cb = predict(model, df)
    result_onnx = predict_onnx(onnx_path, df)

    np.testing.assert_allclose(
        result_cb["proba_class_1"].values,
        result_onnx["proba_class_1"].values,
        atol=1e-5,
        err_msg="Les probabilités ONNX diffèrent de CatBoost de plus de 1e-5",
    )
