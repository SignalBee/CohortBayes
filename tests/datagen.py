import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import List, Optional, Tuple, Type, TypeVar
from scipy.special import expit

class CohortDataGenerator:
  def __init__(
      self,
      rng: np.random.Generator,
      start_cohort: str,
      n_cohorts,
      user_base: int = 10_000,
  ) -> None:
    self.rng = rng
    self.start_cohort = start_cohort
    self.n_cohorts = n_cohorts
    self.user_base = user_base

  def _generate_cohort_labels(self) -> pd.DatetimeIndex:
    return pd.period_range(start="2022-01-01", periods=self.n_cohorts, freq="M").to_timestamp()

  def _generate_cohort_sizes(self) -> npt.NDArray[np.int_]:
    ones = np.ones(shape=self.n_cohorts)
    trend = ones.cumsum() / ones.sum()
    return ((self.user_base * trend * self.rng.gamma(shape=1, scale=1, size=self.n_cohorts)).round().astype(int))

  def _generate_dataset_base(self) -> pd.DataFrame:
    cohorts = self._generate_cohort_labels()
    customer_count = self._generate_cohort_sizes()
    data_df = pd.merge(
        left=pd.DataFrame(data={
            "cohort": cohorts,
            "customer_count": customer_count
        }),
        right=pd.DataFrame(data={"period": cohorts}),
        how="cross",
    )
    data_df["age"] = (data_df["period"].max() - data_df["cohort"]).dt.days
    data_df["cohort_age"] = (data_df["period"] - data_df["cohort"]).dt.days
    data_df = data_df.query("cohort_age >= 0")
    return data_df

  def _generate_retention_rates(self, data_df: pd.DataFrame) -> pd.DataFrame:
    data_df["retention_true_mu"] = (-data_df["cohort_age"] / (data_df["age"] + 1) + 0.8 * np.cos(2 * np.pi * data_df["period"].dt.dayofyear / 365) +
                                    0.5 * np.sin(2 * 3 * np.pi * data_df["period"].dt.dayofyear / 365) - 0.5 * np.log1p(data_df["age"]) + 1.0)
    data_df["retention_true"] = expit(data_df["retention_true_mu"])
    return data_df

  def _generate_user_history(self, data_df: pd.DataFrame) -> pd.DataFrame:
    data_df["active_customer_count"] = self.rng.binomial(n=data_df["customer_count"], p=data_df["retention_true"])
    data_df["active_customer_count"] = np.where(data_df["cohort_age"] == 0, data_df["customer_count"], data_df["active_customer_count"])
    return data_df

  def run(self,) -> pd.DataFrame:
    return (self._generate_dataset_base().pipe(self._generate_retention_rates).pipe(self._generate_user_history))

if __name__ == "__main__":
  seed: int = sum(map(ord, "retention"))
  rng: np.random.Generator = np.random.default_rng(seed=seed)

  start_cohort: str = "2020-01-01"
  n_cohorts: int = 50

  cohort_generator = CohortDataGenerator(rng=rng, start_cohort=start_cohort, n_cohorts=n_cohorts)
  data_df = cohort_generator.run()

  # calculate retention rates
  data_df["retention"] = data_df["active_customer_count"] / data_df["customer_count"]
  data_df.to_csv('eval_data.csv', index=False)
