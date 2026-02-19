import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Thoughts
# Overall: Conversion CampaignType gives the best conversion rate. Larger AdSpend increases the chance of conversion.
# Conversion CampaignType seems to be performing the best because it doesn't require an lot of AdSpend, it has high conversion even with AdSpend below 1000. Other Campaign Types however, experience a significant jump when AdSpend goes above 5000.
# There's no other noticable difference between the CampaignTypes. So, even though the AdSpend is evenly distributed between the types and customers engage with them the same, Conversion campaign still has an unexpected boost in Conversion.
# Customers with higher WebsiteVisits(starting from 11), PreviousPurchases(more than 1), LoyaltyPoints(more than a thousand), PagesPerVisit(more than 3), TimeOnSite(more than 5), EmailOpens(starting from 6) and EmailClicks(more than 2) have a higher chance of conversion.
# Important note, there's no real benefit in increasing numbers above the ones that I stated above. So Capmpaign with AdSpend of 6000 performs the same as with 10000 (Threshold effect).
# Convertion Rate is a strange feature. I'm not sure if it's worth digging into it. Thoughts?
# Threshold effect might be a useful term during the presentation, as well as Conversion rate lift.

