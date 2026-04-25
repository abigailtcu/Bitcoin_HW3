import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import skew
from gensim.models import Word2Vec

class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []
        self.pt = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self
        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()
        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(X_copy)
        if self.skewed_cols:
            X_copy[self.skewed_cols] = self.pt.transform(X_copy[self.skewed_cols])
        return X_copy

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.3, corr_threshold=0.03, cardinality_threshold=0.9):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.cardinality_threshold = cardinality_threshold
        self.features_to_keep = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered = X[cols_low_missing]
        cat_cols = X_filtered.select_dtypes(exclude='number').columns
        cols_to_drop = []
        for col in cat_cols:
            uniqueness_ratio = X_filtered[col].nunique() / len(X_filtered)
            if uniqueness_ratio > self.cardinality_threshold:
                cols_to_drop.append(col)
        remaining_cats = [c for c in cat_cols if c not in cols_to_drop]
        numeric_X = X_filtered.select_dtypes(include='number')
        if y is not None and not numeric_X.empty:
            temp_df = numeric_X.copy()
            temp_df['target'] = y
            correlations = temp_df.corr()['target'].abs().drop('target')
            numeric_to_keep = correlations[correlations >= self.corr_threshold].index.tolist()
        else:
            numeric_to_keep = numeric_X.columns.tolist()
        self.features_to_keep = numeric_to_keep + remaining_cats
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.features_to_keep]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, windows=[5, 10, 20]):
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        data = X_df.squeeze()
        X_out = pd.DataFrame(index=X_df.index)
        for w in self.windows:
            X_out[f'EMA_{w}'] = data.ewm(span=w, min_periods=w).mean()
            X_out[f'ROC_{w}'] = (data.diff(w - 1) / data.shift(w - 1)) * 100
            X_out[f'MOM_{w}'] = data.diff(w)
            delta = data.diff()
            u = pd.Series(np.where(delta > 0, delta, 0), index=delta.index)
            d = pd.Series(np.where(delta < 0, -delta, 0), index=delta.index)
            rs = u.ewm(com=w - 1, adjust=False).mean() / d.ewm(com=w - 1, adjust=False).mean()
            X_out[f'RSI_{w}'] = 100 - (100 / (1 + rs))
            X_out[f'MA_{w}'] = data.rolling(w, min_periods=w).mean()
        return X_out

class PairFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, window=60):
        self.window = window
        self.is_fitted_ = False

    def fit(self, X, y=None):
        if len(X) < self.window:
            raise ValueError("Data length is less than window size")
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Must be fitted before transform.")
        df = pd.DataFrame(X, columns=['price_a', 'price_b']) if isinstance(X, np.ndarray) else X.copy()
        df.columns = ['price_a', 'price_b']
        df[['spread', 'beta']] = self._compute_rolling_regression(df)
        df['z_score'] = (df['spread'] - df['spread'].rolling(self.window).mean()) / df['spread'].rolling(self.window).std()
        return df

    def _compute_rolling_regression(self, df):
        spreads, betas = np.full(len(df), np.nan), np.full(len(df), np.nan)
        for i in range(self.window, len(df)):
            y, x = df['price_a'].values[i-self.window:i], sm.add_constant(df['price_b'].values[i-self.window:i])
            model = sm.OLS(y, x).fit()
            betas[i] = model.params[1]
            spreads[i] = df['price_a'].iloc[i] - (betas[i] * df['price_b'].iloc[i] + model.params[0])
        return pd.DataFrame({'spread': spreads, 'beta': betas}, index=df.index)

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size, self.window, self.min_count = vector_size, window, min_count

    def fit(self, X, y=None):
        sentences = [str(row[0]).split() for row in X]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count)
        return self

    def transform(self, X):
        def get_mean_vector(text):
            words = [w for w in str(text).split() if w in self.model.wv]
            return np.mean([self.model.wv[w] for w in words], axis=0) if words else np.zeros(self.vector_size)
        return np.array([get_mean_vector(row[0]) for row in X])

def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")
    return df

def recode_text_fields(df):
    df = df.copy()
    if "term" in df.columns: df["term"] = df["term"].astype(str).str.extract(r"(\d+)", expand=False)
    if "emp_length" in df.columns: df["emp_length"] = df["emp_length"].astype(str).str.replace("10+ years", "10", regex=False).str.replace("< 1 year", "0", regex=False).str.extract(r"(\d+)", expand=False)
    for pct_col in ["int_rate", "revol_util"]:
        if pct_col in df.columns: df[pct_col] = df[pct_col].astype(str).str.replace("%", "", regex=False).str.strip()
    if "earliest_cr_line" in df.columns: df["earliest_cr_line"] = df["earliest_cr_line"].astype(str).str.extract(r"(\d{4})", expand=False)
    return df

def adjust_dtypes(df):
    df = df.copy()
    num_cols = ["loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs", "revol_bal", "fico_range_low"]
    for col in num_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fill_missing_values(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns: df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude=[np.number]).columns: df[col] = df[col].fillna("missing")
    return df

def clip_numeric_outliers(df, clip_lower=0.01, clip_upper=0.99):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].clip(lower=df[col].quantile(clip_lower), upper=df[col].quantile(clip_upper))
    return df

class LoanDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, clip_lower=0.01, clip_upper=0.99):
        self.clip_lower, self.clip_upper = clip_lower, clip_upper
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = clean_column_names(X); X = recode_text_fields(X); X = adjust_dtypes(X)
        X = fill_missing_values(X); return clip_numeric_outliers(X, self.clip_lower, self.clip_upper)

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X # Add actual feature engineering logic here

def find_high_missing_columns(df, threshold):
    return [c for c in df.columns if df[c].isna().mean() > threshold]

def find_constant_columns(df):
    nunique = df.nunique(dropna=False)
    return nunique[nunique <= 1].index.tolist()

def find_high_cardinality_columns(df, threshold):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [c for c in cat_cols if df[c].nunique(dropna=True) > threshold]

def find_low_variance_numeric_columns(df, threshold):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols: return []
    temp_num = df[num_cols].fillna(df[num_cols].median())
    selector = VarianceThreshold(threshold=threshold).fit(temp_num)
    keep_cols = temp_num.columns[selector.get_support()].tolist()
    return [c for c in num_cols if c not in keep_cols]

class PipelineFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.40, cardinality_threshold=50, variance_threshold=0.0):
        self.missing_threshold, self.cardinality_threshold, self.variance_threshold = missing_threshold, cardinality_threshold, variance_threshold

    def fit(self, X, y=None):
        self.drop_missing_cols_ = find_high_missing_columns(X, self.missing_threshold)
        self.drop_constant_cols_ = find_constant_columns(X.drop(columns=self.drop_missing_cols_, errors="ignore"))
        self.drop_high_cardinality_cols_ = find_high_cardinality_columns(X, self.cardinality_threshold)
        self.drop_low_variance_cols_ = find_low_variance_numeric_columns(X, self.variance_threshold)
        return self

    def transform(self, X):
        cols_to_drop = self.drop_missing_cols_ + self.drop_constant_cols_ + self.drop_high_cardinality_cols_ + self.drop_low_variance_cols_
        return X.drop(columns=list(set(cols_to_drop)), errors="ignore")
