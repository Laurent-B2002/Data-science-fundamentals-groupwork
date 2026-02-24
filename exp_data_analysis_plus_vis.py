import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("marketing_cleaned Pt1.csv")

# df = df.dropna()
# df = df[df["Income"] < 500000]

print(df.head())
print(df.info())
conversion_rate = df["Conversion"].mean()
print(f"Overall Conversion Rate: {conversion_rate:.2%}")
print(df.describe(include='all'))

num_cols = [
    "Age", "AdSpend",
    "WebsiteVisits",
    "PreviousPurchases",
    "LoyaltyPoints", "Income"
]

cat_cols = ["Gender", "CampaignChannel", "CampaignType"]

engagement_cols = [
    "ClickThroughRate",
    "ConversionRate",
    "WebsiteVisits",
    "PagesPerVisit",
    "TimeOnSite",
    "EmailOpens",
    "EmailClicks",
    "SocialShares"
]

engagement_vars = [
    "WebsiteVisits",
    "PagesPerVisit",
    "TimeOnSite",
    "EmailOpens",
    "EmailClicks"
]

print("Show graphs? y/n")
x = input()
if str(x) == "y":
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (8, 5)

    sns.countplot(x="Conversion", data=df)
    plt.title("Conversion Distribution")
    plt.show()

    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

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

    df["AdSpend_bin"] = pd.qcut(df["AdSpend"], 10)

    df["AdSpend_bin"] = df["AdSpend_bin"].astype(str)

    grouped = (
        df.groupby(["CampaignType", "AdSpend_bin"], observed=True)["Conversion"]
        .mean()
        .reset_index()
    )

    sns.lineplot(
        data=grouped,
        x="AdSpend_bin",
        y="Conversion",
        hue="CampaignType",
        marker="o"
    )

    plt.xticks(rotation=25)
    plt.title("Conversion Rate by AdSpend Level and CampaignType")
    plt.ylabel("Conversion Rate")
    plt.xlabel("AdSpend Bin")
    plt.show()

for col in cat_cols:
    for col_eng in engagement_vars:
        sns.boxplot(x=col, y=col_eng, data=df)
        plt.xticks(rotation=45)
        plt.title(f"{col} by {col_eng}")
        plt.show()

for col in cat_cols:
    print(df.groupby(col, observed=True)["Conversion"].mean())

for col in num_cols:
    print(df.groupby(pd.qcut(df[col], 4), observed=True)["Conversion"].mean())

for col in engagement_cols:
    print(df.groupby(pd.qcut(df[col], 4), observed=True)["Conversion"].mean())

print(df.groupby(["CampaignType", pd.cut(df["AdSpend"], bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])], observed=True)["Conversion"].mean().unstack())
print(df.groupby([pd.cut(df["EmailOpens"], bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])], observed=True)["Conversion"].mean())
print(df.groupby([pd.qcut(df["WebsiteVisits"], 25)], observed=True)["Conversion"].mean())
print(df.groupby([pd.qcut(df["PreviousPurchases"], 9)], observed=True)["Conversion"].mean())
print(df.groupby([pd.qcut(df["LoyaltyPoints"], 10)], observed=True)["Conversion"].mean())
print(df.groupby([pd.qcut(df["PagesPerVisit"], 10)], observed=True)["Conversion"].mean())
print(df.groupby([pd.qcut(df["TimeOnSite"], 10)], observed=True)["Conversion"].mean())
print(df.groupby([pd.qcut(df["EmailClicks"], 9)], observed=True)["Conversion"].mean())

campaign_summary = df.groupby("CampaignType")[engagement_cols].agg(
    ["mean"]
)

print(campaign_summary)
num_cols.append("Conversion")
campaign_summary = df.groupby("CampaignType")[num_cols].agg(
    ["mean"]
)

print(campaign_summary)

corr = df.corr(numeric_only=True)

print(corr[engagement_vars].sort_values(by="WebsiteVisits", ascending=False))
print(df.groupby("CampaignChannel")[engagement_vars].mean())
print(df.groupby("CampaignType")[engagement_vars].mean())

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[engagement_vars])

df["EngagementScore"] = scaled_values.mean(axis=1)
print(df.corr(numeric_only=True)["EngagementScore"].sort_values(ascending=False))
print(df.groupby("CampaignChannel")["EngagementScore"].mean())
print(df.groupby("CampaignType")["EngagementScore"].mean())
drivers = df.corr(numeric_only=True)[engagement_vars]

print(drivers)

# Thoughts
# Alright, so I think I'm done with this part. The strange thing that I noticed is that engagement features aren't really affected by anything, even though they are affecting conversion
# I also created an engagement score var to combine all of the engagement features together, but it didn't really give me anything new.
# Perhaps Laurent will be able to find some non-linear dependencies, but as far as my analysis goes, more AdSpend plus more Engagement score equals better Conversion
# Gender, Income or Age did not have any effect on the engagement score or on the conversion.
# My biggest point is still the fact that conversion CampaignType performs better than others on average, and doesn't require as much AdSpend. This plus the correlation matrices plus distribution plus comparing boxplots should be enough for the presentation.

