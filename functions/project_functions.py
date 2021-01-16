import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, scale


def random_forest_tuning(x_train, y_train):
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 10],
        "n_estimators": [50, 100, 150],
        "min_samples_leaf": [5, 10, 15],
        "max_features": ["auto", "sqrt", "log2"],
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10)
    cv_rfc.fit(x_train, y_train)
    return cv_rfc


def random_forest_model():
    rf = RandomForestClassifier(
        criterion="gini",
        max_depth=5,
        n_estimators=50,
        max_features="auto",
        min_samples_leaf=40,
        class_weight="balanced",
        random_state=123,
    )
    return rf


def feature_importance(clf, x_train):
    df = pd.DataFrame(
        clf.feature_importances_, index=x_train.columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    return df


def clf_results(x_train, x_test, y_train, y_test, clf):
    clf.fit(x_train, y_train)
    prediction_train = clf.predict(x_train)
    prediction_test = clf.predict(x_test)
    print(
        "Confusion matrix and results on train: \n",
        pd.crosstab(
            y_train, prediction_train, rownames=["Actual"], colnames=["Predicted"]
        ),
    )
    print(
        classification_report(
            y_train, prediction_train, target_names=wine_data.target_names
        )
    )
    print(
        "Confusion matrix amd results on test: \n",
        pd.crosstab(
            y_test, prediction_test, rownames=["Actual"], colnames=["Predicted"]
        ),
    )

    print(
        classification_report(
            y_test, prediction_test, target_names=wine_data.target_names
        )
    )


def roc_curve_multi_class(df, target):
    classes = target.nunique()
    target_bin = (target, classes)
    x_train, x_test, y_train, y_test = train_test_split(
        df, target_bin, test_size=0.25, random_state=1
    )
    clf = OneVsRestClassifier(
        RandomForestClassifier(max_depth=3, n_estimator=50, random_state=0)
    )
    clf.fit(x_train, y_train)
    y_prob = clf.predict_proba(x_test)
    precision = dict()
    recall = dict()
    for i in classes:
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_prob[:, 1])
        plt.plot(recall[i], precision[i], lw=2, label=wine_data.target_names)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs recall curve")
    plt.show()


def shap_features_bar_plot(clf, x_train, colour):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values[1], x_train, plot_type="bar", color=colour)


def shap_violin_plot(clf, x_train):
    """green feature values will appear for NaN values and grey for bad SHAP values"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values[1], x_train, plot_type="violin")


def plot_single_shap_values(clf, df):
    """jave script won't work in pycharm"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(df)
    shap.initjs()
    return shap.force_plot(
        explainer.expected_value[1], shap_values[1], df, matplotlib=False
    )


def shap_values_importance_plot(clf, df, sample):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(df.iloc[sample, :])
    shap_df = pd.DataFrame(
        data=shap_values[1], index=df.columns, columns=["Shap Value"]
    ).sort_values(by=["Shap Value"], ascending=False)
    sample_df = pd.DataFrame(df.iloc[sample, :], columns=["Feature Value"])
    output_df = shap_df.join(sample_df, how="inner")
    temp_df = output_df
    temp_df.sort_values(by=["Shap Value"], ascending=True, inplace=True)
    temp_df["positive"] = temp_df["Shap Value"] > 0
    index = np.array(len(temp_df.index))
    plt.barh(
        index,
        temp_df["Shap Value"],
        color=temp_df.positive.map({True: "g", False: "c"}),
    )
    plt.yticks(index, temp_df.index)
    plt.title("Importance of Shap Featrues")
    rtdf = output_df.sort_values(by=["Shap Value"], ascending=False)
    print(rtdf)
