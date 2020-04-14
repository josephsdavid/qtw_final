import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from typing import Dict, Any
from collections import Counter
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix


def custom_loss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    weight = np.array([[0, 10], [500, 0]])
    out = cm * weight
    return out.sum() / cm.sum()  # y_true.shape[0]


slater_loss = make_scorer(custom_loss, greater_is_better=False)

"""
data loading
First we define some helper functions
"""


def load_data(path):
    """
    read a csv in from a dict
    """
    result = {}
    reader = csv.DictReader(open(path))
    for row in reader:
        for k, v in row.items():
            result.setdefault(k, []).append(v)
    return result


def _test_value(x: Any) -> bool:
    """
    test if a column is continuous or not
    """
    try:
        float(x)
        return True
    except ValueError:
        return False


def cleanup(d: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    turn default continuous columns into floats, replace non alphanumeric with np.nan
    """
    res = {}
    for k, v in d.items():
        if _test_value(v[0]):
            res[k] = np.array([float(x) if _test_value(x) else np.nan for x in v])
        else:
            res[k] = np.array(v)
    return res


def convert_dollars_percs(x: np.ndarray) -> np.ndarray:
    """replace dollar signs and percentages, as well as rogue negative signs"""
    out = [
        x[i].replace("$", "").replace("%", "").replace("-", "")
        for i in range(x.shape[0])
    ]
    out = np.array([float(z) if _test_value(z) else np.nan for z in out])
    return out


def impute_cats(
    d: Dict[str, np.ndarray], c: Dict[str, Counter]
) -> Dict[str, np.ndarray]:
    """
    mode impute categorical variables, given a dictionary of counts and a
    dictionary of data
    """
    for k in c.keys():
        d[k][np.isnan(d[k])] = c[k].most_common(1)[0][0]
    return d


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    data = load_data(path)
    data = cleanup(data)
    for k in ["x32", "x37"]:
        data[k] = convert_dollars_percs(data[k])
    # cats are continent, month, day, and percentage. We have those enumerated
    # and mode imputed
    cats = [k for k, v in data.items() if not _test_value(v[0])]
    # percentage is also a categorical variable
    cats.append("x32")
    # give it some nans
    cont_dict = {"": np.nan}
    for idx, k in enumerate(list(set(data["x24"]))[1:]):
        cont_dict[k] = idx
    data["x24"] = np.array([cont_dict[v] for v in data["x24"]])
    day_dict = dict(
        zip(["monday", "tuesday", "wednesday", "thurday", "friday"], range(0, 5))
    )
    day_dict[""] = np.nan
    data["x30"] = np.array([day_dict[v] for v in data["x30"]])
    months = [
        "January",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "July",
        "Aug",
        "sept.",
        "Oct",
        "Nov",
        "Dev",
    ]
    month_dict = dict(zip(months, range(0, 12)))
    month_dict[""] = np.nan
    data["x29"] = np.array([month_dict[v] for v in data["x29"]])
    cat_dict = {k: Counter(data[k]) for k in cats}
    data = impute_cats(data, cat_dict)
    return data


def cyclical(x, period):
    # http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    """
    sine cosine transformation for days and months
    """
    s = np.sin(x * (2.0 * np.pi / period))
    c = np.cos(x * (2.0 * np.pi / period))
    return s, c


x = load_dataset("../data/final_project.csv")
sc = [cyclical(m, 5) for m in x["x30"]]
x["x30s"] = np.stack(sc, axis=0)[:, 0]
x["x30c"] = np.stack(sc, axis=0)[:, 1]
sc = [cyclical(m, 12) for m in x["x29"]]
x["x29s"] = np.stack(sc, axis=0)[:, 0]
x["x29c"] = np.stack(sc, axis=0)[:, 1]

# y variable
y = x.pop("y")


# categorical variables with continent asia or not (doesnt matter bc continent
# is a shitty featyre)
def categorize_with_asia(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    d = d.copy()
    continent = "x24"
    asia = Counter(d[continent]).most_common(1)[0][0]
    d[continent] = np.array([1.0 if x == asia else 0.0 for x in d[continent]])
    perc = "x32"
    n_percs = np.unique(d[perc]).shape[0]
    enum_dict = dict(zip(np.unique(d[perc]), range(np.unique(d[perc]).shape[0])))
    enum_percs = np.array([enum_dict[x] for x in d[perc]])
    d[perc] = np.eye(n_percs)[enum_percs]
    to_drop = ["x2", "x41", "x29", "x30"]
    for k in to_drop:
        d.pop(k, None)
    return d


x_encoded = categorize_with_asia(x)
X = np.column_stack(list(x_encoded.values()))


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


X = SimpleImputer().fit_transform(X)


rfc_1 = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
rfc_1_score = cross_val_score(
    rfc_1, X, y, cv=5, scoring=slater_loss, n_jobs=-1, verbose=1
)
print(rfc_1_score)


# happy matrix
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
rf.fit(X_train, y_train)
ppp = rf.predict(X_test)
print(confusion_matrix(y_test, ppp))


def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for i in range(X_train.shape[1]):
        save = X_train[:, i].copy()
        X_train[:, i] = np.random.permutation(X_train[:, i])
        m = metric(rf, X_train, y_train)
        X_train[:, i] = save
        imp.append(baseline - m)
    return np.array(imp)


imps = permutation_importances(rf, X_train, y_train, slater_loss)

for i in range(imps.shape[0]):
    plt.barh(i, imps[i])
plt.axvline(imps.mean())
plt.show()

keep_vars = [i for i in range(imps.shape[0]) if imps[i] > imps.mean()]


X_small = X[:, keep_vars].copy()


rfc_s = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
rfc_s_score = cross_val_score(
    rfc_s, X_small, y, cv=5, scoring=slater_loss, n_jobs=-1, verbose=1
)
print(rfc_s_score)


# scaled:

rfc_s_scaled = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
rfc_ss_score = cross_val_score(
    rfc_s_scaled,
    StandardScaler().fit_transform(X_small),
    y,
    cv=5,
    scoring=slater_loss,
    n_jobs=-1,
    verbose=1,
)
print(rfc_ss_score)

rfc_scaled = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
rfc_scale_score = cross_val_score(
    rfc_scaled,
    StandardScaler().fit_transform(X),
    y,
    cv=5,
    scoring=slater_loss,
    n_jobs=-1,
    verbose=1,
)
print(rfc_scale_score)
