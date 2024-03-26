from typing import Optional, Any, Union, List
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder

standard_column_names = {"cohort": "cohort", "period": "period", "cohort_age": "cohort_age"}

class Data:
  def __init__(self, data: pd.DataFrame) -> None:
    self.data = data
    self.features: list[str] = ["age", "cohort_age", "month"]

  def period_encoder(self): return LabelEncoder()
  def cohort_encoder(self): return LabelEncoder()
  def age_scaler(self): return MaxAbsScaler()
  def cohort_age_scaler(self): return MaxAbsScaler()

  @property
  def customer_count(self): return self.data["customer_count"].to_numpy()
  @property
  def active_customer_count(self): return self.data["active_customer_count"].to_numpy()
  @property
  def retention(self): return self.data["retention"].to_numpy()
  @property
  def retention_logit(self): return logit(self.retention)
  @property
  def revenue(self): return self.data["revenue"].to_numpy()
  @property
  def revenue_per_user(self): return self.revenue / self.active_customer_count
  @property
  def cohort(self): return self.data["cohort"].to_numpy()
  @property
  def cohort_idx(self): return self.cohort_encoder.fit_transform(self.cohort).flatten()
  @property
  def period(self): return self.data["period"].to_numpy()
  @property
  def period_idx(self): return self.period_encoder.fit_transform(self.period).flatten()
  @property
  def age(self): return self.data["age"].to_numpy()
  @property
  def cohort_age(self): return self.data["cohort_age"].to_numpy()
  @property
  def cohort_age_scaled(self): return self.cohort_age_scaler.fit_transform(self.cohort_age.reshape(-1, 1)).flatten()
  @property
  def age_scaled(self): return self.age_scaler.fit_transform(self.age.reshape(-1, 1)).flatten()
  @property
  def train_filtered(self): return self.data[self.features]
  @property
  def index(self): return self.data.index.to_numpy()

  def __repr__(self):
    return self.data

  def shape(self):
    return self.data.shape

class DataLoad:
  def __init__(self, path, columns_names: Optional[dict] = None) -> None:
    self.path = path
    self.load()
    if columns_names != None:
      self.column_rename = {v: k for k, v in columns_names.items()}
      self.data.columns.rename(self.column_rename, inplace=True)

  def load(self) -> None:
    self.data = pd.read_csv(self.path, parse_dates=["cohort", "period"])

  def prepare(self, split_date: str) -> List[Data]:
    self.add_features()
    return self.splitter(split_date=split_date)

  def add_features(self):
    self.data["month"] = self.data["period"].dt.strftime("%m").astype(int)
    self.data["cohort_month"] = (self.data["cohort"].dt.strftime("%m").astype(int))
    self.data["period_month"] = (self.data["period"].dt.strftime("%m").astype(int))

  def splitter(self, split_date: str) -> List[Data]:
    self.data = self.data[self.data['cohort_age'] > 0].reset_index(drop=True)
    train = self.data[self.data['period'] <= split_date]
    test = self.data[self.data['period'] > split_date]
    test = test[test["cohort"].isin(train["cohort"].unique())]
    return Data(train), Data(test)
