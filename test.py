import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nodes import apply_outliers

# data
data = pd.read_csv("raw_1.csv")

# outlier filtering
filtered_test = apply_outliers(data, method = "accuracy", threshold = 0.7)

# drop any rows with missing values 
filtered_test = filtered_test.dropna(
    subset=["rt", "congruency", "prev_congruency", "subj_code"]
)

# reset index
filtered_test = filtered_test.reset_index(drop = True)

# ensure categorical values are treated as factors
filtered_test["congruency"] = filtered_test["congruency"].astype("category")
filtered_test["prev_congruency"] = filtered_test["prev_congruency"].astype("category")

# print(len(filtered_test))

# fit model 
md = smf.mixedlm(
    "rt ~ congruency * prev_congruency", 
    data = filtered_test, 
    groups = filtered_test['subj_code']
)

mdf = md.fit(method="lbfgs") # still not sure what lbfgs does better

print(mdf.summary())
