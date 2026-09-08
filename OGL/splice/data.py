"""Load and split the official LIBSVM splice data."""

import hashlib

import numpy as np

try:
    from libsvmdata import fetch_dataset, fetch_libsvm
except ImportError:
    fetch_dataset = None
    fetch_libsvm = None


class Data:
    def __init__(self):
        self.X_train = None
        self.X_validate = None
        self.X_test = None
        self.y_train = None
        self.y_validate = None
        self.y_test = None


class SpliceSettings:
    def __init__(
        self,
        num_train,
        num_validate,
        num_test,
        num_features,
        dataset=None,
    ):
        self.num_train = num_train
        self.num_validate = num_validate
        self.num_test = num_test
        self.num_features = num_features
        self.dataset = dataset


def _fetch_libsvm_dataset(dataset_name):
    if fetch_dataset is not None:
        return fetch_dataset(dataset_name)
    if fetch_libsvm is not None:
        return fetch_libsvm(dataset_name)
    raise ImportError("libsvmdata is required for the splice experiment")


def _dense_float_array(value):
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=float)


def _binary_pm_one_labels(y_train, y_test):
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    classes = np.unique(y_train)
    if classes.size != 2:
        raise ValueError("splice must have exactly two classes")
    if not np.all(np.isin(np.unique(y_test), classes)):
        raise ValueError("splice test labels do not match the training labels")
    mapping = {float(classes[0]): -1.0, float(classes[1]): 1.0}
    train_pm = np.array([mapping[float(value)] for value in y_train], dtype=float)
    test_pm = np.array([mapping[float(value)] for value in y_test], dtype=float)
    return train_pm, test_pm, classes


def load_splice(settings):
    """Use 80%/20% of the official training set and all official test data."""

    X_pool, y_pool = _fetch_libsvm_dataset("splice")
    X_test, y_test = _fetch_libsvm_dataset("splice_test")
    X_pool = _dense_float_array(X_pool)
    X_test = _dense_float_array(X_test)
    y_pool, y_test, classes = _binary_pm_one_labels(y_pool, y_test)
    if X_pool.ndim != 2 or X_test.ndim != 2 or X_pool.shape[1] != X_test.shape[1]:
        raise ValueError("splice training and test feature dimensions do not match")

    seed = int(getattr(settings, "seed", 2026))
    validation_fraction = float(getattr(settings, "validation_fraction", 0.2))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    order = np.random.default_rng(seed).permutation(X_pool.shape[0])
    num_validate = int(round(validation_fraction * X_pool.shape[0]))
    num_validate = min(max(num_validate, 1), X_pool.shape[0] - 1)
    num_train = X_pool.shape[0] - num_validate
    train_indices = order[:num_train]
    validation_indices = order[num_train:]

    X_train = X_pool[train_indices].copy()
    X_validate = X_pool[validation_indices].copy()
    X_test = X_test.copy()
    if bool(getattr(settings, "standardize", False)):
        feature_mean = np.mean(X_train, axis=0)
        feature_scale = np.std(X_train, axis=0)
        feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
        X_train = (X_train - feature_mean) / feature_scale
        X_validate = (X_validate - feature_mean) / feature_scale
        X_test = (X_test - feature_mean) / feature_scale
    else:
        feature_mean = np.zeros(X_train.shape[1], dtype=float)
        feature_scale = np.ones(X_train.shape[1], dtype=float)

    settings.dataset = "splice"
    settings.test_dataset = "splice_test"
    settings.data_source = "libsvm_official_split"
    settings.num_train = num_train
    settings.num_validate = num_validate
    settings.num_test = X_test.shape[0]
    settings.num_features = X_train.shape[1]
    settings.class_labels = classes.copy()
    settings.split_hash = hashlib.sha256(
        np.asarray(order, dtype=np.int64).tobytes()
    ).hexdigest()
    fingerprint = hashlib.sha256()
    for array in (X_pool, y_pool, X_test, y_test):
        contiguous = np.ascontiguousarray(array)
        fingerprint.update(str(contiguous.shape).encode("ascii"))
        fingerprint.update(contiguous.tobytes())
    settings.dataset_fingerprint = fingerprint.hexdigest()

    data = Data()
    data.X_train = X_train
    data.X_validate = X_validate
    data.X_test = X_test
    data.y_train = y_pool[train_indices]
    data.y_validate = y_pool[validation_indices]
    data.y_test = y_test
    data.train_indices = train_indices.copy()
    data.validation_indices = validation_indices.copy()
    data.feature_mean = feature_mean
    data.feature_scale = feature_scale
    return data
