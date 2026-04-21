import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nodes import apply_outliers
from nodes import fit_models
from nodes import transform_rt

# data
data = pd.read_csv("raw_1.csv").query("rt > 150")
print(data)

# outlier filtering
filtered_test = apply_outliers(data, method = "sd", threshold = 3)

transform_rt(data = filtered_test, method = "raw")

model_test = fit_models(data = filtered_test, model_type = "lmm-full")
summary = model_test.summary()

print(summary)