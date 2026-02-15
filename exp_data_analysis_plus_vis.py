import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("marketing.csv")
df = df.dropna()
df = df[df["Income"] < 500000]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
print(df.head())
print(df.info())
print(df.describe())

sns.countplot(x="Conversion", data=df)
plt.title("Conversion Distribution")
plt.show()

conversion_rate = df["Conversion"].mean()
print(f"Overall Conversion Rate: {conversion_rate:.2%}")
print(df.describe(include='all'))
num_cols = [
    "Age", "AdSpend",
    "WebsiteVisits",
    "PreviousPurchases", "LoyaltyPoints"
]

for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


cat_cols = ["Gender", "CampaignChannel", "CampaignType"]

for col in cat_cols:
    plt.figure()
    sns.countplot(x=col, data=df)
    plt.title(f"{col} Distribution")
    plt.xticks(rotation=45)
    plt.show()


for col in num_cols:
    plt.figure()
    sns.boxplot(x="Conversion", y=col, data=df)
    plt.title(f"{col} vs Conversion")
    plt.show()


for col in cat_cols:
    conversion_by_cat = df.groupby(col)["Conversion"].mean().sort_values()
    
    plt.figure()
    conversion_by_cat.plot(kind="bar")
    plt.title(f"Conversion Rate by {col}")
    plt.ylabel("Conversion Rate")
    plt.xticks(rotation=45)
    plt.show()

plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)

sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()


engagement_cols = [
    "WebsiteVisits",
    "PagesPerVisit",
    "TimeOnSite",
    "EmailOpens",
    "EmailClicks",
    "SocialShares"
]

for col in engagement_cols:
    plt.figure()
    sns.boxplot(x="Conversion", y=col, data=df)
    plt.title(f"{col} vs Conversion")
    plt.show()


sns.scatterplot(
    x="TimeOnSite",
    y="PagesPerVisit",
    hue="Conversion",
    data=df,
    alpha=0.6
)
plt.title("Time on Site vs Pages per Visit")
plt.show()

# These are mine preliminary thoughts on the data
# This file has a bunch of graphs about the data
# 1. We missed two gigantic values in the income feature, need to make sure we deal with them during preprocessing
# 2. Around 90% of the customers converted, which sounds suspicious. This again raises a question about what does conversion mean, and what is even the commercial reason behind our analysis if they already have such astounding numbers?
# 3. Most of the features follow a type of uniform distribution except for gender, where we have around 5000 women vs 3000 men.
# 4. More adspend seems to result in slightly better conversion rate. The same goes for the most other engagement columns, with higher number, the chance that the customer converted is slightly higher.
# 5. Overall, seems to be an extremely noisy dataset. There's very little doubt about it being simply generated.
# 6. These are just my preliminary thoughts from a basic overview of the data and I'll try more tools, but I doubt that we will find any meaningful correlation between the features.