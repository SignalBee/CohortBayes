import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

class ModelDiagnostic:
  def __init__(self, model):
    self.model = model

  def posteriod_predictive(self):
    return

  def in_smaple_fit_means(self):
    return

  def uncertainty(self, variable: str):
    assert variable in [self.model.estimated_vars], f"variable must be one of the model estimators [{', '.join(self.model.estimated_vars)}]"
    self.hdi = ''
