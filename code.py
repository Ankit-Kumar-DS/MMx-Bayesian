# this is python code for applying bayesian MMx

# importing libraries

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Loading the channels cost/engagement and prescription data

path = r'C:\D\MMX\data\November 25\Overall'
df_eng = pd.read_csv(path+"\cost_overall.csv")
df = df_eng.copy()
print(df.head())
df_cols = df.columns.tolist()
print(df_cols)
channels = df_cols[1:-1]
channels
