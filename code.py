# this is python code for applying bayesian MMx

# importing libraries

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Loading the channels cost/engagement and prescription data

df_eng = pd.read_csv(r"data.csv")
df = df_eng.copy()
df_cols = df.columns.tolist()
channels = df_cols[1:-1]
channels

# adding log transformation to dependent variable (TRx)
y = np.log1p(df["TRx"].values)

# Step 3: Adstock transformation function u
def adstock(series, decay=0.5):
    result = []
    carryover = 0
    for val in series:
        carryover = val + carryover * decay
        result.append(carryover)
    return np.array(result)

# Apply adstock for each channel
for ch in channels:
    df[ch + "_adstock"] = adstock(df[ch].values, decay=0.5)


# Prepare predictors (log-transform to capture saturation)
X = np.log1p(df[[ch + "_adstock" for ch in channels]].values)


# Step 4: Bayesian Model (defining priors and likelihood func)

with pm.Model() as bayes_mmm:
    #priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    betas = pm.HalfNormal("betas", sigma=1, shape=len(channels))
    sigma = pm.HalfNormal("sigma", sigma=1)

    # linear predictor (alpha+beta*X) finding the predicted mean 
    mu = alpha + pm.math.dot(X, betas)

    #liklihood function
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # sampling from posterior distribution
    if __name__ == "__main__":
        trace = pm.sample(1000, tune=500, target_accept=0.9, cores = 1)


# Step 5: summarize results

summary = az.summary(trace, var_names=["alpha", "betas", "sigma"], hdi_prob=0.95)
print(summary)

# Step 6: Plotting  posterior distributions for priors

az.plot_trace(trace, var_names=["alpha","betas"])
plt.show()

# Step 7: plotting Posterior distributions for betas with confidence interval

az.plot_posterior(trace, var_names=["betas"], hdi_prob=0.95)
plt.show()