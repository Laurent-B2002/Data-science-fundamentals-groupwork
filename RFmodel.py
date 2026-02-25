import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load data
df = pd.read_csv("marketing_cleaned Pt1.csv")

X = df.drop(columns=["Conversion", "CustomerID"])
y = df["Conversion"]

X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "class_weight": [None, "balanced", "balanced_subsample"]
}

threshold = 0.5  # Custom classification threshold, actually helps

keys = param_grid.keys()
values = param_grid.values()
combinations = list(itertools.product(*values))

results = []

for combo in combinations:
    
    params = dict(zip(keys, combo))
    
    rf = RandomForestClassifier(
        random_state=42,
        **params
    )
    
    rf.fit(X_train, y_train)
    
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    roc_auc = roc_auc_score(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_1 = precision_score(y_test, y_pred, pos_label=1)

    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)

    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    
    # Store results
    results.append({
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "class_weight": params["class_weight"],
        "ROC_AUC": roc_auc,
        "Balanced_Accuracy": bal_acc,
        "Precision_0": precision_0,
        "Precision_1": precision_1,
        "Recall_0": recall_0,
        "Recall_1": recall_1,
        "F1_0": f1_0,
        "F1_1": f1_1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by ROC-AUC
results_df = results_df.sort_values(by="ROC_AUC", ascending=False)

# Print results
print(results_df)

# Save to CSV
results_df.to_csv("random_forest_results.csv", index=False)

# Random forest for part 4
# I'm uploading a python file with the RandomForest classifier code and two csv file with my results.
# Overall impression is that Random Forest can handle the dataset relatively well, but there are no optimal hyperparameters. They would depend on the company's goals.
# For example, in most cases, Random Forest struggles classifying the customers that didn't convert and if that's important to the company, they should focus on metrics like Balanced_accuracy, Precision_0, Recall_0, etc.
# The hyperparameters that helped with that issue are the balanced class_weight and changing the classification threshold. One of the CSV files was generated using 0.5 threshold and the other 0.7
# The 0.7 one has much better results in terms of Balanced Accuracy.
# The python file is rather simple to modify, so if we need to add additional hyperparameters or change the threshold, it would be easy to do so. Please check which metrics I used when you're testing your own models, because we need to compare them in the end.