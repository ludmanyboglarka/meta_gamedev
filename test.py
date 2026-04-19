import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nodes import apply_outliers
from nodes import fit_models

# data
data = pd.read_csv("raw_1.csv")

# outlier filtering
filtered_test = apply_outliers(data, method = "sd", threshold = 3)

models_test = fit_models(filtered_test, model_type = "intercept")