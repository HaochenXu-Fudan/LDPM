import hashlib
import numpy as np
import math

try:
    from libsvmdata import fetch_dataset, fetch_libsvm
except ImportError:
    fetch_dataset = None
    fetch_libsvm = None

class Data():
    def __init__(self):
        self.X_train = 0
        self.X_validate = 0
        self.X_test = 0
        self.y_train = 0
        self.y_validate = 0
        self.y_test = 0

def matrix_data(data):
    m_data = Data()
    m_data.X_train, m_data.X_validate, m_data.X_test = np.matrix(data.X_train), np.matrix(data.X_validate), np.matrix(data.X_test)
    m_data.y_train, m_data.y_validate, m_data.y_test = np.matrix(data.y_train).T, np.matrix(data.y_validate).T, np.matrix(data.y_test).T
    return m_data

class Data_with_Info():
    def __init__(self, data, settings, data_index = 0) -> None:
        self.data = data 
        self.settings = settings 
        self.data_index = data_index

def Data_Generator_Wrapper(generator, settings, data_index=0):
    #重要的是这个data_info含有train test 以及分割方式
    data_info = Data_with_Info(generator(settings), settings)
    ##这个index也没什么用
    data_info.data_index = data_index
    return data_info 

class ElasticNet_Setting():
    def __init__(self, num_train, num_validate, num_test, num_features,dataset=None):
        self.num_train = num_train
        self.num_validate = num_validate
        self.num_test = num_test
        self.num_features = num_features
        self.dataset=dataset

def ElasticNet_Data_Generator(settings):
    num_train, num_validate, num_test = settings.num_train, settings.num_validate, settings.num_test
    num_features = settings.num_features
    num_samples = num_train + num_validate + num_test

    num_nonzero_features = 15

    # Create data
    correlation_matrix = [  [np.power(0.5, abs(i - j)) for i in range(0, num_features)] 
        for j in range(0, num_features)]
    X = np.random.randn(num_samples, num_features) @ np.linalg.cholesky(correlation_matrix).T

    # X = normalize(X, norm='l2', axis=1)

    # beta real is a shuffled array of zeros and iid std normal values
    beta_real = np.concatenate((
            np.ones((num_nonzero_features)),
            np.zeros((num_features - num_nonzero_features))
        ))

    np.random.shuffle(beta_real)

    y_true = X @ beta_real

    # add noise
    snr = 2
    epsilon = np.random.randn(num_samples)
    SNR_factor = snr / np.linalg.norm(y_true) * np.linalg.norm(epsilon)
    y = y_true + 1.0 / SNR_factor * epsilon

    data = Data()
    # split data
    data.X_train, data.X_validate, data.X_test = X[0:num_train], X[num_train:num_train + num_validate], X[num_train + num_validate:]
    data.y_train, data.y_validate, data.y_test = y[0:num_train], y[num_train:num_train + num_validate], y[num_train + num_validate:]
    return data


def OverlapGroupLasso_Data_Generator(settings):
    """Synthetic overlapping group Lasso data from overlap_group_lasso_data_generation.md."""

    num_train = int(settings.num_train)
    num_validate = int(settings.num_validate)
    num_test = int(settings.num_test)
    num_features = int(settings.num_features)
    num_samples = num_train + num_validate + num_test
    snr = float(getattr(settings, "snr", 2.0))
    active_fraction = float(getattr(settings, "active_group_fraction", 0.05))
    groups = getattr(settings, "overlap_groups", None)
    if groups is None:
        group_size = int(getattr(settings, "overlap_group_size", 60))
        num_groups = int(getattr(settings, "num_experiment_groups", max(1, num_features // group_size)))
        groups = [
            np.unique(np.random.choice(num_features, size=group_size, replace=True)).astype(int)
            for _ in range(num_groups)
        ]
        settings.overlap_groups = groups
    groups = [np.asarray(group, dtype=int).reshape(-1) for group in groups]
    num_groups = len(groups)
    num_active = int(getattr(settings, "num_active_groups", max(1, round(active_fraction * num_groups))))
    num_active = min(max(num_active, 1), num_groups)

    X = np.random.randn(num_samples, num_features)
    active_groups = np.random.choice(num_groups, size=num_active, replace=False)
    beta_real = np.zeros(num_features, dtype=float)
    latent_beta = []
    for group_index, group in enumerate(groups):
        values = np.zeros(group.size, dtype=float)
        if group_index in active_groups:
            values = np.random.randn(group.size)
            np.add.at(beta_real, group, values)
        latent_beta.append(values)

    y_signal = X @ beta_real
    if np.linalg.norm(y_signal) <= 1e-12:
        raise ValueError("generated zero signal; increase active groups or check overlap groups")
    epsilon = np.random.randn(num_samples)
    sigma = np.linalg.norm(y_signal) / max(np.sqrt(snr) * np.linalg.norm(epsilon), 1e-12)
    y = y_signal + sigma * epsilon

    data = Data()
    data.X_train = X[:num_train]
    data.X_validate = X[num_train : num_train + num_validate]
    data.X_test = X[num_train + num_validate :]
    data.y_train = y[:num_train]
    data.y_validate = y[num_train : num_train + num_validate]
    data.y_test = y[num_train + num_validate :]
    data.true_beta = beta_real
    data.latent_beta = latent_beta
    data.active_groups = active_groups
    data.snr = snr
    return data


def _fetch_libsvm_dataset(dataset_name):
    if fetch_dataset is not None:
        return fetch_dataset(dataset_name)
    if fetch_libsvm is not None:
        return fetch_libsvm(dataset_name)
    raise ImportError("libsvmdata is required for real-data experiments.")


def _dense_float_array(value):
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=float)


def _binary_pm_one_labels(y_train, y_test):
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    classes = np.unique(y_train)
    if classes.size != 2:
        raise ValueError("splice must have exactly two classes; got %r" % classes.tolist())
    if not np.all(np.isin(np.unique(y_test), classes)):
        raise ValueError("splice test labels do not match the training labels")
    mapping = {float(classes[0]): -1.0, float(classes[1]): 1.0}
    train_pm = np.array([mapping[float(value)] for value in y_train], dtype=float)
    test_pm = np.array([mapping[float(value)] for value in y_test], dtype=float)
    return train_pm, test_pm, classes


def Splice_Data_Generator(settings):
    """Load the official LIBSVM splice split for OGL least-squares experiments.

    The 1,000 official training examples are deterministically split into
    train/validation subsets.  The 2,175 examples in ``splice_test`` remain an
    untouched test set.  Feature standardization, when enabled, is fitted on
    the training subset only.
    """

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
        raise ValueError("validation_fraction must be strictly between zero and one")
    order = np.random.default_rng(seed).permutation(X_pool.shape[0])
    num_validate = int(round(validation_fraction * X_pool.shape[0]))
    num_validate = min(max(num_validate, 1), X_pool.shape[0] - 1)
    num_train = X_pool.shape[0] - num_validate
    train_indices = order[:num_train]
    validation_indices = order[num_train:]

    X_train = X_pool[train_indices].copy()
    X_validate = X_pool[validation_indices].copy()
    X_test = X_test.copy()
    if bool(getattr(settings, "standardize", True)):
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


def ElasticNet_Real_Data_Generator(settings):
    if fetch_libsvm is None:
        raise ImportError("libsvmdata is required for real-data experiments.")
    X, y = fetch_libsvm(settings.dataset)
    if settings.dataset in ['breast-cancer','diabetes','liver-disorders']:
        min_val=np.min(X,0)
        max_val=np.max(X,0)
        X = (X - min_val) / (max_val - min_val)
    num_sample, num_features = X.shape

    random_order = np.arange(num_sample)
    np.random.shuffle(random_order)

    num_train = math.floor(num_sample*.8*.8)
    num_validate = math.floor(num_sample*.8*.2)
    num_test = num_sample-num_train-num_validate

    settings.num_train=num_train
    settings.num_validate=num_validate
    settings.num_test=num_test
    settings.num_features=num_features

    X = X[random_order[:num_train + num_validate + num_test]]
    y = y[random_order[:num_train + num_validate + num_test]]

    data = Data()
    data.X_train = X[:num_train]
    data.X_validate = X[num_train : num_train + num_validate]
    data.X_test = X[num_train + num_validate : num_train + num_validate + num_test]
    data.y_train = y[:num_train]
    data.y_validate = y[num_train : num_train + num_validate]
    data.y_test = y[num_train + num_validate : num_train + num_validate + num_test]

    return data
#%%
class SGL_Setting:
    def __init__(self, num_train=90, num_validate=30, num_test=200, num_features=600, num_experiment_groups=30):
        self.num_train = num_train
        self.num_validate = num_validate
        self.num_test = num_test
        self.num_features = num_features
        self.num_experiment_groups = num_experiment_groups
        self.num_true_groups = 3
        # num_true_groups is defined in data_generator

def SGL_Data_Generator(settings):
    num_train, num_validate, num_test = settings.num_train, settings.num_validate, settings.num_test
    num_features = settings.num_features
    num_samples = num_train + num_validate + num_test

    num_true_groups = settings.num_true_groups

    group_feature_sizes = [num_features//num_true_groups] * num_true_groups
    base_nonzero_coeff = np.array([1, 2, 3, 4, 5])
    num_nonzero_features = len(base_nonzero_coeff)

    # correlation_matrix = [  [np.power(0.5, abs(i - j)) for i in range(0, num_features)] 
    #     for j in range(0, num_features)]
    # X = np.random.randn(num_samples, num_features) @ np.linalg.cholesky(correlation_matrix).T

    X = np.random.randn(num_samples, num_features)

    beta_real = np.concatenate(
        [np.concatenate( (base_nonzero_coeff, np.zeros(group_feature_size - num_nonzero_features)) )
        for group_feature_size in group_feature_sizes] )

    y_true = X @ beta_real

    # add noise
    snr = 2
    epsilon = np.random.randn(num_samples)
    SNR_factor = snr / np.linalg.norm(y_true) * np.linalg.norm(epsilon)
    y = y_true + 1.0 / SNR_factor * epsilon

    data = Data()
    # split data
    data.X_train, data.X_validate, data.X_test = X[0:num_train], X[num_train:num_train + num_validate], X[num_train + num_validate:]
    data.y_train, data.y_validate, data.y_test = y[0:num_train], y[num_train:num_train + num_validate], y[num_train + num_validate:]

    return data

#%%
class SPAM_Setting:
    def __init__(self, num_train = 100, num_validate = 50, num_test = 50, smooth_fcn_list=[lambda x: 0], feat_range = [0, 5]):
        self.num_train = num_train
        self.num_validate = num_validate
        self.num_test = num_test
        self.num_samples = num_train + num_validate + num_test
        self.smooth_fcn_list = smooth_fcn_list
        self.num_features = len(smooth_fcn_list)
        self.feat_range = feat_range
        # num_true_groups is defined in data_generator

class SPAM_Data(Data):
    def __init__(self, data, settings):
        super().__init__()
        self.X_train = data.X_train
        self.X_validate = data.X_validate
        self.X_test = data.X_test
        self.y_train = data.y_train
        self.y_validate = data.y_validate
        self.y_test = data.y_test
        self.X_full = np.vstack((self.X_train, self.X_validate, self.X_test))
        self.train_idx = np.arange(0, settings.num_train)
        self.validate_idx = np.arange(settings.num_train, settings.num_train + settings.num_validate)
        self.test_idx = np.arange(settings.num_train + settings.num_validate, settings.num_train + settings.num_validate + settings.num_test)

def SPAM_Data_Generator_0(settings):
    num_features = len(settings.smooth_fcn_list)
    all_Xs = map(lambda x: _make_shuffled_uniform_X(), range(num_features))
    X_smooth = np.column_stack(list(all_Xs))

    y_smooth = 0
    for idx, fcn in enumerate(settings.smooth_fcn_list):
        y_smooth += fcn(X_smooth[:, idx]).reshape(settings.num_samples, 1)

    def _make_shuffled_uniform_X(eps=0.0001):
        step_size = (settings.feat_range[1] - settings.feat_range[0] + eps)/settings.num_samples
        # start the uniformly spaced X at a different start point, jitter by about 1/20 of the step size
        jitter = np.random.uniform(0, 1) * step_size/10
        equal_spaced_X = np.arange(settings.feat_range[0] + jitter, settings.feat_range[1] + jitter, step_size)
        np.random.shuffle(equal_spaced_X)
        return equal_spaced_X

def SPAM_Data_Generator(settings):
    num_train = settings.num_train
    num_validate = settings.num_validate
    num_features = len(settings.smooth_fcn_list)

    def _make_shuffled_uniform_X(eps=0.0001):
        step_size = (settings.feat_range[1] - settings.feat_range[0] + eps)/settings.num_samples
        # start the uniformly spaced X at a different start point, jitter by about 1/20 of the step size
        jitter = np.random.uniform(0, 1) * step_size/10
        equal_spaced_X = np.arange(settings.feat_range[0] + jitter, settings.feat_range[1] + jitter, step_size)
        np.random.shuffle(equal_spaced_X)
        return equal_spaced_X

    all_Xs = [_make_shuffled_uniform_X() for _ in range(num_features)]
    X_smooth = np.column_stack(all_Xs)

    y_smooth = 0
    for idx, fcn in enumerate(settings.smooth_fcn_list):
        y_smooth += fcn(X_smooth[:, idx])

    # add noise
    snr = 2
    epsilon = np.random.randn(settings.num_samples)
    SNR_factor = snr / np.linalg.norm(y_smooth) * np.linalg.norm(epsilon)
    y = y_smooth + 1.0 / SNR_factor * epsilon

    data = Data()
    # split data
    data.X_train, data.X_validate, data.X_test = X_smooth[0:num_train], X_smooth[num_train:num_train + num_validate], X_smooth[num_train + num_validate:]
    data.y_train, data.y_validate, data.y_test = y[0:num_train], y[num_train:num_train + num_validate], y[num_train + num_validate:]

    data = SPAM_Data(data, settings)

    return data

# %%
class Real_Data_Setting():
    def __init__(self, dataset_name="australian_scale", num_train = 100, num_validate = 100, num_test = 100, print_flag = False) -> None:     
        self.num_train = num_train
        self.num_validate = num_validate
        self.num_test = num_test

        self.num_samples = num_train + num_validate + num_test
        self.dataset_name = dataset_name 
        self.print_flag = print_flag
        self.num_features = 0

def Real_Data_Fetcher(settings):
    if fetch_libsvm is None:
        raise ImportError("libsvmdata is required for real-data experiments.")
    X, y = fetch_libsvm(settings.dataset_name)
    num_sample, num_features = X.shape
    settings.num_features = num_features

    if settings.print_flag: 
        print(
            "%15s: (%d, %d) " % (settings.dataset_name, num_sample, num_features) 
            + "num_train: %d num_validate: %d num_test: %d" %
            (settings.num_train, settings.num_validate, settings.num_test)
        )

    random_order = np.arange(num_sample)
    np.random.shuffle(random_order)

    num_train = settings.num_train
    num_validate = settings.num_validate
    num_test = settings.num_test
    if num_test == 0:
        num_test = num_sample - num_train - num_validate 
        settings.num_samples = num_sample 
        settings.num_test = num_test 

    X = X[random_order[:num_train + num_validate + num_test]]
    y = y[random_order[:num_train + num_validate + num_test]]

    data = Data()
    data.X_train = X[:num_train]
    data.X_validate = X[num_train : num_train + num_validate]
    data.X_test = X[num_train + num_validate : num_train + num_validate + num_test]
    data.y_train = y[:num_train]
    data.y_validate = y[num_train : num_train + num_validate]
    data.y_test = y[num_train + num_validate : num_train + num_validate + num_test]

    return data

#%%
class CV_Data():
    def __init__(self):
        self.X = 0
        self.X_test = 0
        self.y = 0
        self.y_test = 0
        self.iTr = []
        self.iVal = []

class CV_Real_Data_Setting():
    def __init__(self, dataset_name="australian_scale", num_one_fold = 0, num_CV = 3, num_test = 0, print_flag = False) -> None:
        self.num_one_fold = num_one_fold
        self.num_CV = num_CV
        
        self.num_train = num_one_fold * (num_CV - 1) * num_CV
        self.num_validate = num_one_fold * num_CV
        self.num_test = num_test

        self.num_CV_samples = num_one_fold * num_CV
        self.dataset_name = dataset_name 
        self.print_flag = print_flag
        self.num_features = 0

def CV_Real_Data_Fetcher(settings):
    X, y = fetch_libsvm(settings.dataset_name)
    yc = np.unique(y)
    assert len(yc) == 2
    y += 100
    yc = np.unique(y)
    y[y == yc[0]] = -1
    y[y == yc[1]] = 1

    num_sample, num_features = X.shape
    settings.num_features = num_features

    if settings.num_one_fold == 0:
        if settings.num_test == 0:
            settings.num_one_fold = num_sample // (2 * settings.num_CV)
            settings.num_test = num_sample - settings.num_one_fold * settings.num_CV
            settings.num_train = settings.num_one_fold * (settings.num_CV - 1) * settings.num_CV
            settings.num_validate = settings.num_one_fold * settings.num_CV
        else:
            settings.num_one_fold = (num_sample - settings.num_test) // settings.num_CV
            settings.num_train = settings.num_one_fold * (settings.num_CV - 1) * settings.num_CV
            settings.num_validate = settings.num_one_fold * settings.num_CV
        settings.num_CV_samples = settings.num_validate

    if settings.print_flag: 
        print(
            "%15s: (%d, %d) " % (settings.dataset_name, num_sample, num_features) 
            + "%d folds, %d per fold, num_test: %d" %
            (settings.num_CV, settings.num_one_fold, settings.num_test)
        )

    random_order = np.arange(num_sample) #排好
    np.random.shuffle(random_order) #打乱

    index_CV = [
        np.arange( i*settings.num_one_fold, (i+1)*settings.num_one_fold ).tolist()
        for i in range(settings.num_CV) # 每个fold里排好的index
    ]
    tmp_list = np.arange(settings.num_CV).tolist() #[0,1,2]
    tmp_list = tmp_list[1:] + tmp_list #[1,2,0,1,2]

    index_train = []
    index_validate = []
    for i in range(settings.num_CV):
        train_tmp = []
        for j in tmp_list[i:i+(settings.num_CV-1)]:
            train_tmp += index_CV[j] #3*[2*num_one_fold]
        index_train.append(train_tmp)
        index_validate.append(index_CV[i]) #3*[num_one_fold]

    num_test = settings.num_test

    X = X[random_order[:settings.num_CV_samples + num_test]]
    y = y[random_order[:settings.num_CV_samples + num_test]] # CV * num_one_fold + test

    data = CV_Data()
    data.X = X[:settings.num_CV_samples] # CV*num_one_fold
    data.X_test = X[settings.num_CV_samples : settings.num_CV_samples + num_test]
    data.y = y[:settings.num_CV_samples]
    data.y_test = y[settings.num_CV_samples : settings.num_CV_samples + num_test]

    data.iTr = index_train
    data.iVal = index_validate

    return data

#%%
class SVM_Data_Setting():
    def __init__(self, dataset_name="australian_scale", print_flag = False) -> None:     
        self.num_train = 0
        self.num_validate = 0
        self.num_test = 0

        self.num_samples = 0 + 0 + 0
        self.dataset_name = dataset_name 
        self.print_flag = print_flag
        self.num_features = 0

def SVM_Data_Fetcher(settings):
    X, y = fetch_libsvm(settings.dataset_name)
    num_samples, num_features = X.shape
    settings.num_train = num_samples // 3
    settings.num_validate = num_samples // 3
    settings.num_test = num_samples - num_samples // 3 * 2
    settings.num_samples = num_samples
    settings.num_features = num_features

    if settings.print_flag: 
        print(
            "%15s: (%d, %d) " % (settings.dataset_name, num_samples, num_features) 
            + "num_train: %d num_validate: %d num_test: %d" %
            (settings.num_train, settings.num_validate, settings.num_test)
        )

    random_order = np.arange(num_samples)
    np.random.shuffle(random_order)

    num_train = settings.num_train
    num_validate = settings.num_validate
    num_test = settings.num_test

    yc = np.unique(y)
    assert len(yc) == 2
    y += 100
    yc = np.unique(y)
    y[y == yc[0]] = -1
    y[y == yc[1]] = 1
    X = X[random_order[:num_train + num_validate + num_test]]
    y = y[random_order[:num_train + num_validate + num_test]]

    data = Data()
    data.X_train = X[:num_train]
    data.X_validate = X[num_train : num_train + num_validate]
    data.X_test = X[num_train + num_validate : num_train + num_validate + num_test]
    data.y_train = y[:num_train]
    data.y_validate = y[num_train : num_train + num_validate]
    data.y_test = y[num_train + num_validate : num_train + num_validate + num_test]

    return data

#%%
class MatrixGroupsObservedData:
    # special data structure for storing matrix-value observations
    def __init__(self, row_features, col_features, train_idx, validate_idx, test_idx, observed_matrix, alphas, betas, gamma, real_matrix):
        self.num_rows = real_matrix.shape[0]
        self.num_cols = real_matrix.shape[1]

        self.num_alphas = len(alphas)
        self.num_betas = len(betas)

        self.row_features = row_features
        self.col_features = col_features
        self.train_idx = train_idx
        self.validate_idx = validate_idx
        self.test_idx = test_idx
        self.observed_matrix = observed_matrix

        self.real_matrix = real_matrix
        self.real_alphas = alphas
        self.real_betas = betas
        self.real_gamma = gamma

class MCG_data:
    def __init__(self, matrix_data):
        self.num_rows = matrix_data.num_rows
        self.num_cols = matrix_data.num_cols

        self.row_features = np.hstack(matrix_data.row_features)
        self.col_features = np.hstack(matrix_data.col_features)

        self.train_idx = matrix_data.train_idx
        self.validate_idx = matrix_data.validate_idx
        self.test_idx = matrix_data.test_idx
        self.observed_matrix = matrix_data.observed_matrix

        self.real_matrix = matrix_data.real_matrix
        self.real_alphas = np.vstack(matrix_data.real_alphas)
        self.real_betas = np.vstack(matrix_data.real_betas)
        self.real_gamma = matrix_data.real_gamma

        self.num_alphas = matrix_data.num_alphas
        self.num_betas = matrix_data.num_betas

def get_matrix_completion_groups_fitted_values(row_features, col_features, alphas, betas, gamma):
    m = 0
    if len(row_features) > 0:
        m += np.hstack(row_features) * np.vstack(alphas) * np.ones(gamma.shape[1]).T
    if len(col_features) > 0:
        m += (np.hstack(col_features) * np.vstack(betas) * np.ones(gamma.shape[0]).T).T
    return gamma + m

class MCG_DataGenerator:
    def __init__(self, settings):
        self.settings = settings
        self.snr = settings.snr

    def matrix_completion_groups(self):
        gamma_to_row_col_m = self.settings.gamma_to_row_col_m
        feat_factor = self.settings.feat_factor
        matrix_shape = (self.settings.num_rows, self.settings.num_cols)

        def _make_feature_vec(num_features, num_nonzero_groups, num_total_groups, feat_factor):
            return (
                [(i + 1) * feat_factor * np.matrix(np.ones(num_features)).T for i in range(num_nonzero_groups)]
                + [np.matrix(np.zeros(num_features)).T] * (num_total_groups - num_nonzero_groups)
            )

        def _create_feature_matrix(num_samples, num_features):
            return np.matrix(np.random.randn(num_samples, num_features))

        alphas = _make_feature_vec(
            self.settings.num_row_features,
            self.settings.num_nonzero_row_groups,
            self.settings.num_row_groups,
            feat_factor=feat_factor
        )

        betas = _make_feature_vec(
            self.settings.num_col_features,
            self.settings.num_nonzero_col_groups,
            self.settings.num_col_groups,
            feat_factor=feat_factor
        )

        row_features = [
            _create_feature_matrix(self.settings.num_rows, self.settings.num_row_features)
            for i in range(self.settings.num_row_groups)
        ]
        col_features = [
            _create_feature_matrix(self.settings.num_cols, self.settings.num_col_features)
            for i in range(self.settings.num_col_groups)
        ]

        gamma = 0
        for i in range(self.settings.num_nonzero_s):
            u = np.random.randn(self.settings.num_rows)
            v = np.random.randn(self.settings.num_cols)
            gamma += np.matrix(u).T * np.matrix(v)

        only_row_col = get_matrix_completion_groups_fitted_values(
            row_features,
            col_features,
            alphas,
            betas,
            np.zeros(gamma.shape),
        )
        xz_feat_factor = 1.0/gamma_to_row_col_m * 1/np.linalg.norm(only_row_col, ord="fro") * np.linalg.norm(gamma, ord="fro")
        row_features = [xz_feat_factor * m for m in row_features]
        col_features = [xz_feat_factor * m for m in col_features]

        true_matrix = get_matrix_completion_groups_fitted_values(
            row_features,
            col_features,
            alphas,
            betas,
            gamma,
        )

        epsilon = np.random.randn(matrix_shape[0], matrix_shape[1])
        SNR_factor = self._make_snr_factor(np.linalg.norm(true_matrix, ord="fro"), np.linalg.norm(epsilon, ord="fro"))
        observed_matrix = true_matrix + 1.0 / SNR_factor * epsilon

        # sample out parts of each column and parts of each row
        train_indices = set()
        validate_indices = set()
        # num_train_sample = max(int(self.settings.train_perc * matrix_shape[0]), 1)
        # num_val_sample = max(int(self.settings.validate_perc * matrix_shape[0]), 1)
        num_train_sample = 2
        num_val_sample = 1
        # sample from each column
        for i in range(matrix_shape[1]):
            shuffled_idx = i * matrix_shape[1] + np.random.permutation(matrix_shape[0])
            train_indices.update(shuffled_idx[:num_train_sample])
            validate_indices.update(shuffled_idx[num_train_sample:num_train_sample + num_val_sample])

        # sample from each row
        for j in range(matrix_shape[0]):
            shuffled_idx = j + matrix_shape[1] * np.random.permutation(matrix_shape[0])
            train_indices.update(shuffled_idx[:num_train_sample])
            validate_indices.update(shuffled_idx[num_train_sample:num_train_sample + num_val_sample])
        validate_indices.difference_update(train_indices)
        test_indices = set(range(matrix_shape[0] * matrix_shape[1]))
        test_indices.difference_update(train_indices)
        test_indices.difference_update(validate_indices)

        train_indices = np.array(list(train_indices), dtype=int)
        validate_indices = np.array(list(validate_indices), dtype=int)
        test_indices = np.array(list(test_indices), dtype=int)

        return MatrixGroupsObservedData(
            row_features,
            col_features,
            train_indices,
            validate_indices,
            test_indices,
            observed_matrix,
            alphas,
            betas,
            gamma,
            true_matrix
        )

    def _make_snr_factor(self, true_sig_norm, noise_norm):
        return self.snr / true_sig_norm * noise_norm
