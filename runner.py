import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nodes import apply_outliers
from nodes import fit_models
from nodes import transform_rt
from nodes import extract_results

data = pd.read_csv("raw_1.csv").query("rt > 150")

outlier_options = [
    ("sd", 2),
    ("sd", 3),
    ("accuracy", 0.7),
    ("accuracy", 0.8),
    ("no-filter", None)
]
transform = ["log", "raw"]
model_types = ["lmm-full", "lmm-intercept", "lmm-intercept-slope"]

results = []

for method, threshold in outlier_options: 
    for t in transform: 
        for model in model_types:

            d1 = apply_outliers(data, method, threshold)
            d2 = transform_rt(d1, t)
            fit = fit_models(d2, model)

            res = extract_results(fit)


            results.append({
                "outlier_method": method, 
                "threshold": threshold, 
                "transformed": t,
                "model": model,
                **res
            })

for r in results: 
    coef = r['coeff']
    p = r['p_value']

    print(
        f"{r['outlier_method']}, {r['threshold']}, {r['transformed']}, {r['model']} | "
        f"coef = {coef if coef is None else round(coef, 4)}, "
        f"p = {p if p is None else f'{p:.4g}'}, "
        f"n = {r['n_obs']}"
    )

## TODO: STORE RESULTS IN A DESIGNATED OBJECT - 