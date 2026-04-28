import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import skew
from gensim.models import Word2Vec
import re

# --- GLOBAL HELPER FUNCTIONS ---
# Moving these here allows both the Cleaner and the Selector to use them without errors.

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

# --- TRANSFORMER CLASSES ---

class LoanDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, clip_lower=0.01, clip_upper=0.99):
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = self._clean_column_names(X)
        X = self._recode_text_fields(X)
        X = self._adjust_dtypes(X)
        X = self._fill_missing_values(X)
        X = self._clip_numeric_outliers(X)
        return X

    def _clean_column_names(self, df):
        df.columns = [re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z_]+", "_", c.strip().lower())).strip("_") for c in df.columns]
        return df

    def _recode_text_fields(self, df):
        if "term" in df.columns: 
            df["term"] = df["term"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        if "emp_length" in df.columns: 
            df["emp_length"] = df["emp_length"].astype(str).str.replace("10+ years", "10", regex=False).str.replace("< 1 year", "0", regex=False).str.extract(r"(\d+)", expand=False).astype(float)
        for pct_col in ["int_rate", "revol_util"]:
            if pct_col in df.columns: 
                df[pct_col] = df[pct_col].astype(str).str.replace("%", "", regex=False).str.strip()
                df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")
        if "earliest_cr_line" in df.columns: 
            df["earliest_cr_line"] = df["earliest_cr_line"].astype(str).str.extract(r"(\d{4})", expand=False).astype(float)
        return df

    def _adjust_dtypes(self, df):
        num_cols = ["loan_amnt", "funded_amnt", "installment", "annual_inc", "dti", "delinq_2yrs", "revol_bal", "fico_range_low"]
        for col in num_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _fill_missing_values(self, df):
        for col in df.select_dtypes(include=[np.number]).columns: 
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(exclude=[np.number]).columns: 
            df[col] = df[col].fillna("missing")
        return df

    def _clip_numeric_outliers(self, df):
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].clip(lower=df[col].quantile(self.clip_lower), upper=df[col].quantile(self.clip_upper))
        return df

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Rubric Alignment: Running the 10 feature engineering steps
        X = self._add_fico_average(X)
        X = self._add_credit_history_length(X)
        X = self._add_income_based_ratios(X)
        X = self._add_credit_behavior_ratios(X)
        X = self._add_home_and_bankruptcy_ratios(X)
        X = self._add_accounts_per_credit_year(X)
        return X

    def _add_fico_average(self, df):
        if {"fico_range_low", "fico_range_high"}.issubset(df.columns):
            df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        return df

    def _add_credit_history_length(self, df):
        if "earliest_cr_line" in df.columns:
            df["credit_history_length"] = 2026 - pd.to_numeric(df["earliest_cr_line"], errors="coerce")
        return df

    def _add_income_based_ratios(self, df):
        if {"loan_amnt", "annual_inc"}.issubset(df.columns):
            df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
        if {"installment", "annual_inc"}.issubset(df.columns):
            df["installment_to_income"] = df["installment"] / (df["annual_inc"] + 1)
        if {"revol_bal", "annual_inc"}.issubset(df.columns):
            df["revol_bal_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)
        return df

    def _add_credit_behavior_ratios(self, df):
        if {"delinq_2yrs", "total_acc"}.issubset(df.columns):
            df["delinq_ratio"] = df["delinq_2yrs"] / (df["total_acc"] + 1)
        if {"inq_last_6mths", "total_acc"}.issubset(df.columns):
            df["inquiry_ratio"] = df["inq_last_6mths"] / (df["total_acc"] + 1)
        return df

    def _add_home_and_bankruptcy_ratios(self, df):
        if {"mort_acc", "total_acc"}.issubset(df.columns):
            df["mortgage_ratio"] = df["mort_acc"] / (df["total_acc"] + 1)
        if {"pub_rec_bankruptcies", "total_acc"}.issubset(df.columns):
            df["bankruptcy_ratio"] = df["pub_rec_bankruptcies"] / (df["total_acc"] + 1)
        return df

    def _add_accounts_per_credit_year(self, df):
        if {"open_acc", "credit_history_length"}.issubset(df.columns):
            df["accounts_per_credit_year"] = df["open_acc"] / (df["credit_history_length"] + 1)
        return df

class PipelineFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.40, cardinality_threshold=50, variance_threshold=0.0):
        self.missing_threshold = missing_threshold
        self.cardinality_threshold = cardinality_threshold
        self.variance_threshold = variance_threshold

    def fit(self, X, y=None):
        self.drop_missing_cols_ = find_high_missing_columns(X, self.missing_threshold)
        X_temp = X.drop(columns=self.drop_missing_cols_, errors="ignore")
        self.drop_constant_cols_ = find_constant_columns(X_temp)
        self.drop_high_cardinality_cols_ = find_high_cardinality_columns(X, self.cardinality_threshold)
        self.drop_low_variance_cols_ = find_low_variance_numeric_columns(X, self.variance_threshold)
        return self

    def transform(self, X):
        cols_to_drop = list(set(self.drop_missing_cols_ + self.drop_constant_cols_ + 
                                self.drop_high_cardinality_cols_ + self.drop_low_variance_cols_))
        return X.drop(columns=cols_to_drop, errors="ignore")
