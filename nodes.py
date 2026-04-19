import pandas as pd
import numpy as np
import statsmodels as sm

raw_data = pd.read_csv("raw_1.csv")
print(raw_data.head())


#' NODE 1: OUTLER EXCLUSION
#' Parameters: method can be based on standard deviation or accuracy
#' Threshold: in case of accuracy-based filtering, can be 0.7 or 0.8, in case of standard deviation-based filtering, +-3 or +-2 SD from the conditional mean
#' Return: two possible outcomes
def apply_outliers(data, method, threshold):
    if method == "sd": 
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
            .assign(rt_z_score=lambda df: (df['rt'] - df['participant_mean_rt']) / df['participant_sd_rt']) ## calculating z-score by participant - necessary for sd filtering 
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
    
#' NODE 2: STATISTICAL MODELS
#' Model type: random intercept and random slope / only random intercept
# def fit_models(data, model_type):
#    if model_type == "intercept-slope":
        
        