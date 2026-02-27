import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel("MentalHealthSurvey.xlsx")

# Target Creation

RISK_FEATURES = ["depression", "anxiety", "isolation", "future_insecurity"]

df["risk_score"] = df[RISK_FEATURES].mean(axis=1) # applied horizontly
df["at_risk"] = (df["risk_score"] >= 3.5).astype(int)

# 1. Class Distribution

print("Class Distribution:")
print(df["at_risk"].value_counts())

plt.figure()
counts = df["at_risk"].value_counts()
plt.bar(["Low Risk", "High Risk"], counts)
plt.title("Class Distribution (Low vs High Risk)")
plt.ylabel("Number of Students")
plt.show()

# 2. Boxplot


plt.figure(figsize=(6,4))
sns.boxplot(x="at_risk", y="financial_concerns", data=df)

plt.title("Financial Concerns by Risk Level")
plt.xlabel("0 = Low Risk, 1 = High Risk")
plt.ylabel("Financial Concerns (1-5)")
plt.show()

# 3. Correlation Heatmap

numeric_df = df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 4. Correlation with Target

correlation = corr["at_risk"].sort_values()

plt.figure()
correlation.plot(kind="barh")
plt.title("Correlation of Features with Target (at_risk)")
plt.xlabel("Correlation Value")
plt.show()