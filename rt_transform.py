import pandas as pd
import numpy as np
#import streamlit as st #UI stuff
#import matplotlib.pyplot as plt #visualization stuff

#dataset gets imported here. there should be a prompt for the player to choose what dataset they want to use in the future
clean104 = pd.read_csv(r"C:\Users\haran\metalab\gamedev\104\raw_1.csv").query("rt > 150") #importing + filtering rt below 150

def transform_rt(clean104, method):
    clean104 = clean104.copy()

    clean104 = clean104.dropna(subset=["rt"])


    if method == "raw":
        clean104["rt_used"] = clean104["rt"]

    elif method == "log":
        if (clean104["rt"] <= 0).any():
            raise ValueError("RT contains non-positive values; cannot take log.")

        clean104["rt_used"] = np.log(clean104["rt"])

    else:
        raise ValueError(f"Unknown method: {method}")

    return clean104

clean104_transformed = transform_rt(clean104, "log")  #set to "raw" or "log", duh