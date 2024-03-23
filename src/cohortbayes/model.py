from typing import Optional
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
from pymc_bart.split_rules import ContinuousSplitRule, SubsetSplitRule

from .diagnostics import ModelDiagnostic
from .dataops import Data


class BayesModel(ModelDiagnostic):
  def __init__(self, training_data: Data, add_eps: bool, rng: Optional[int] = None):
    self.training_data = training_data
    self.model = pm.Model(coords={"feature": self.training_data.features})
    self.add_eps = add_eps
    self.rng = rng if rng is not None else np.random.default_rng()
    super().__init__(model=self.model)

  def build(self):
    with self.model as model:
      model.add_coord(name="obs", values=self.training_data.index, mutable=True)
      age_scaled = pm.MutableData(name="age_scaled", value=self.training_data.age_scaled, dims="obs")
      cohort_age_scaled = pm.MutableData(name="cohort_age_scaled", value=self.training_data.cohort_age_scaled, dims="obs")
      x = pm.MutableData(name="x", value=self.training_data.train_filtered, dims=("obs", "feature"))
      revenue = pm.MutableData(name="revenue", value=self.training_data.revenue, dims="obs")
      customer_count = pm.MutableData(name="customer_count", value=self.training_data.customer_count, dims="obs")
      active_customer_count = pm.MutableData(name="active_customer_count", value=self.training_data.active_customer_count, dims="obs")

      # PRIORS
      intercept = pm.Normal(name="intercept", mu=0, sigma=1)
      prior_age_scaled = pm.Normal(name="prior_age_scaled", mu=0, sigma=1)
      prior_cohort_age_scaled = pm.Normal(name="prior_cohort_age_scaled", mu=0, sigma=1)
      prior_age_cohort_age_interaction = pm.Normal(name="prior_age_cohort_age_interaction", mu=0, sigma=1)

      # PARAMETERIZATION
      mu = pmb.BART(name="mu", X=x, Y=self.training_data.retention_logit, m=100, response="mix", split_rules=[ContinuousSplitRule(), ContinuousSplitRule(), SubsetSplitRule()], dims="obs",)
      p = pm.Deterministic(name="p", var=pm.math.invlogit(mu), dims="obs")
      if self.add_eps:
        eps = np.finfo(float).eps
        p = np.where(p == 0, eps, p)
        p = np.where(p == 1, 1 - eps, p)
      lam_log = pm.Deterministic(
        name="lam_log",
        var=intercept + prior_age_scaled * age_scaled + prior_cohort_age_scaled * cohort_age_scaled + prior_age_cohort_age_interaction * age_scaled * cohort_age_scaled,
        dims="obs",
      )
      lam = pm.Deterministic(name="lam", var=pm.math.exp(lam_log), dims="obs")

      # LIKELIHOOD
      active_customer_count_estimated = pm.Binomial( name="n_active_users_estimated", n=customer_count, p=p, observed=active_customer_count, dims="obs",)
      x = pm.Gamma(name="revenue_estimated", alpha=active_customer_count_estimated + eps, beta=lam, observed=revenue, dims="obs",)
      mean_revenue_per_active_user = pm.Deterministic(name="mean_revenue_per_active_user", var=(1 / lam), dims="obs")
      pm.Deterministic(name="mean_revenue_per_user", var=p * mean_revenue_per_active_user, dims="obs")

  def visulaize(self):
    pm.model_to_graphviz(model=self.model)

  def train(self):
    return
