import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Load Data directly from your GitHub repository URL to guarantee access
repo_url = "https://raw.githubusercontent.com/jimswatt/marketing-roi-regression/main/marketing_sales_data.csv"
df = pd.read_csv(repo_url)
df = df.dropna().drop_duplicates()

print("--- Data Loaded and Cleaned ---")
print(df.head())

# 2. Multicollinearity Check (VIF)
X_vif = df[['TV', 'Radio', 'Social_Media']]
X_vif_const = sm.add_constant(X_vif)
vif_df = pd.DataFrame()
vif_df["Feature"] = X_vif_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape)]
print("\n--- VIF Scores ---")
print(vif_df[vif_df['Feature'] != 'const'])

# 3. Multiple Linear Regression Model
Y = df['Sales']
X = df[['TV', 'Radio', 'Social_Media']]
X_model_const = sm.add_constant(X)
mlr_model = sm.OLS(Y, X_model_const).fit()
print("\n--- OLS Regression Summary ---")
print(mlr_model.summary())

# 4. Residual Diagnostics
residuals = mlr_model.resid
fitted_values = mlr_model.fittedvalues
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(x=fitted_values, y=residuals, ax=axes, color='blue')
axes.axhline(y=0, color='red', linestyle='--')
axes.set_title('Residuals vs Fitted')

sns.histplot(residuals, kde=True, ax=axes, color='green')
axes.set_title('Residual Normality')

sm.qqplot(residuals, line='45', fit=True, ax=axes)
axes.set_title('Normal Q-Q Plot')
plt.tight_layout()
plt.show()

# 5. Extract statistics using explicit bracket properties
adj_r_squared = mlr_model.rsquared_adj
f_stat = mlr_model.fvalue
b0 = mlr_model.params.iloc[0]
b1 = mlr_model.params.iloc[1]
b2 = mlr_model.params.iloc[2]
b3 = mlr_model.params.iloc[3]

print(f"\nModel Equation: Sales = {b0:.4f} + ({b1:.4f}*TV) + ({b2:.4f}*Radio) + ({b3:.4f}*Social_Media)")
print(f"Adjusted R-Squared: {adj_r_squared*100:.2f}%")
print(f"Overall F-Statistic: {f_stat:.4f}")
