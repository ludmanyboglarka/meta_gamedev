import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

raw_data = pd.read_csv("raw_1.csv")
print(raw_data.head())

#' ----------------------------------------------
#' NODE 1: OUTLER EXCLUSION
#' Parameters: method can be based on standard deviation or accuracy
#' Threshold: in case of accuracy-based filtering, can be 0.7 or 0.8, in case of standard deviation-based filtering, +-3 or +-2 SD from the conditional mean
#' Return: two possible outcomes
#' ----------------------------------------------
def apply_outliers(data, method, threshold):
    if method == "sd": 

        method = method.strip() # removes space/new lines

        ## if method is based on sd, only the correct trials are needed
        valid_sd_data = data[
            (data["correct"] != 0) & (data["prev_correct"] != 0)
        ]
        participant_summary = ( ## calculate participant-level summary statistics
            valid_sd_data
            .groupby(['subj_code', 'congruency', 'prev_congruency'])['rt']
            .agg(participant_mean_rt = 'mean', participant_sd_rt = 'std')
            .reset_index()
            )
        processed_sd_data = ( ## join summary and calculate z scores
            valid_sd_data
            .merge(participant_summary, on = ['subj_code', 'congruency', 'prev_congruency'], how = 'left') ## left joining the summaries by row, by condition
            .assign(rt_z_score=lambda df:
                     (df['rt'] - df['participant_mean_rt']) / df['participant_sd_rt']) ## calculating z-score by participant - necessary for sd filtering 
        )
        processed_sd_data = processed_sd_data[
            processed_sd_data["rt_z_score"].abs() < threshold 
        ] ## keeping rows where the fits the threshold requirements
        return processed_sd_data 
    elif method == "accuracy":
        accuracy_sum = (
            data
            .groupby(['subj_code'])['correct']
            .mean()
            .reset_index(name = "all_accuracy")
        )
        accuracy_data = (
            data
            .merge(accuracy_sum, on = ['subj_code'], how = 'left')
        )
        accuracy_data = accuracy_data[
            accuracy_data["all_accuracy"] > threshold
        ]
        return accuracy_data
    elif method == "no-filter":
        return data.copy()
    else:
        raise ValueError(f"Unknown outlier method: '{method}'")

#' ----------------------------------------------
#' NODE 2: LOG TRANSFORM 
#' ----------------------------------------------
def transform_rt(data, method):

    transform_data = data.copy()

    transform_data = transform_data.dropna(subset=["rt"])

    if method == "raw":
        transform_data["rt_used"] = transform_data["rt"]
        return transform_data

    elif method == "log":
        if (transform_data["rt"] <= 0).any():
            raise ValueError("RT contains non-positive values; cannot take log.")
        log_data = transform_data
        log_data["rt_used"] = np.log(log_data["rt"])

        return log_data
    else:
        raise ValueError(f"Unknown method: {method}")

#' ----------------------------------------------
#' NODE 3: STATISTICAL MODELS
#' Model type: random intercept and random slope / only random intercept
#' ----------------------------------------------
def fit_models(data, model_type):

    model_type = model_type.strip() # removes space/new lines

    ## ensure that data is in the appropriate format
    # drop na values
    model_data = data.dropna(
                subset = ["rt", "congruency", "prev_congruency", "subj_code"]
                ).copy() # make sure to create a true copy
    # reset index
    model_data = model_data.reset_index(drop = True)
    # ensure categorical values are treated as factors
    model_data["congruency"] = model_data["congruency"].astype("category")
    model_data["prev_congruency"] = model_data["prev_congruency"].astype("category")

    # rt_used: 
    formula = "rt_used ~ congruency * prev_congruency"

    if model_type == "lmm-intercept":
        # model 
        lmm_intercept = smf.mixedlm(
            formula, 
            data = model_data, 
            groups = model_data["subj_code"]
        )

        # fit the model
        intercept_fit = lmm_intercept.fit(method = "lbfgs") # still not sure what lbfgs does better
        
        return intercept_fit
    
    elif model_type == "lmm-intercept-slope":
        # model
        lmm_intercept_slope = smf.mixedlm(
            formula, 
            data = model_data, 
            groups = model_data["subj_code"], 
            re_formula = "~ congruency" # random slope is the currect trial congruency
        )

        # fit the model
        intercept_slope_fit = lmm_intercept_slope.fit(method = "lbfgs")
        
        return intercept_slope_fit
    
    elif model_type == "lmm-full":
        # model: full random effects model, the random effect includes the cong * prev cong interaction
        lmm_full = smf.mixedlm(
            formula, 
            data = model_data, 
            groups = model_data["subj_code"], 
            re_formula = "~ congruency * prev_congruency" # random slope is the currect trial congruency
        )
        # fit full lmm model
        lmm_full_fit = lmm_full.fit(method = "lbfgs")
    
        # return only the fitted object
        return lmm_full_fit
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    

# TODO: ERROR HANDLING
# nem fittel a modell - dokumentalni kell, hogy milyen jellegu a nem fittelodes. nem konvergal, vagy singular fit lesz belole
# csak azt irja ki amit kiir az error message - vagy csak elmenti az adatba 

#' ----------------------------------------------
#' NODE 5: EXTRACT RESULTS 
#' ----------------------------------------------
def extract_results(model):

    # if model failed, there was an error
    if model is None: 
        return {
            "coeff": None, 
            "p_value": None, 
            "n_obs": None
        }

    params = model.params
    p_values = model.pvalues

    # find CSE - interaction term
    interaction = None
    for name in params.index: 
        if "congruency" in name and "prev_congruency" in name and ":" in name:
            interaction = name
            break # finishes for loop if interaction is found

    # if interaction is not found
    if interaction is None:
        return {
            "coeff": None, 
            "p_value": None, 
            "n_obs": model.nobs
        }
    return {
        "coeff": params[interaction], 
        "p_value": p_values[interaction], 
        "n_obs": model.nobs
    }
