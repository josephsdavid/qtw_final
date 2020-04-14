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



# logreg lives here
from sklearn.metrics import mean_squared_error, r2_score, recall_score, confusion_matrix, make_scorer,classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

%time df = pd.read_csv("../data/final_project.csv", sep=",", header=0)

#Check ze rows
print(len(df))
print(df.shape)

# %%
df.info()
df.describe()

#%%
#Looking at nulls
df.isnull().sum()

#%%
#  Looking at missing data
t_nulls = df.isnull().sum().sort_values(ascending=False)
perc = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_vals = pd.concat([t_nulls, perc], axis=1, keys=["Total", "Missing Percent"])
missing_vals["Missing Percent"] = missing_vals['Missing Percent'].apply(lambda x: x *100)

print(missing_vals)

#%%

#Looking at categorical.
df_cat = df.describe(include=['object'])
print(df_cat.T)

#renaming columns
df.rename(columns={'x24':'continent', 'x29':'month', 'x30':'day'},inplace=True)

#%%
# Data Cleaning.

df['x37'] = df['x37'].str.replace('$', '').astype(float)
df['x32'] = df['x32'].str.replace('%', '').astype(float)


df['continent'] = df['continent'].str.replace('euorpe','europe')
df['month'] = df['month'].str.replace('Dev','Dec')
df['month'] = df['month'].str.replace('sept.','Sep')
df['day'] = df['day'].str.replace('thurday','thursday')

#Fill NA's with the median
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].fillna(df[col].median())

#%%
# EDA

table=pd.crosstab(df['day'],df['y'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Frequency of Target vs day of the week')
plt.xlabel('Day')
plt.ylabel('Frequency')

table=pd.crosstab(df['month'],df['y'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Frequency of Target vs month')
plt.xlabel('Month')
plt.ylabel('Frequency')



# Check numerical histograms of data
# df.hist(bins=50, figsize = (20,15))


# #%%
# # heatmap
# plt.figure(figsize=(20,10))
# sns.heatmap(df.corr().round(1),vmax=1, annot=True, cmap = 'YlGnBu',annot_kws={"fontsize":10})

# #%%

# df_num = df.select_dtypes(include=['float64','int64'])

#%%
X = df.drop('y', axis = 1)
y = df['y']

# Adding in a random noise componenet to test feature importance
#X['Random'] = np.random.random(size=len(X))

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state = 42)

print("\nChecking shape of test/train data")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



#%%

#Scaling Data and prepping for RF and PCA

X1 = X_train.copy()
y1 = y_train.copy()

# Drop vars
drop_col = ['day','month','continent', 'x2','x41']
# One hot encoding x32

#Grouping Asias vs Other (europe and america) due to america's small class count

#

# Dropping from xtrain and xtest
X1 = X1.drop(drop_col, axis=1)
X_test_sc = X_test.drop(drop_col, axis=1)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X1)
X_test_sc = scaler.transform(X_test_sc)
y1 = np.array(y1)

#Setting up loss function
def custom_loss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    weight = np.array([[0, 10], [500, 0]])
    out = cm * weight
    return out.sum()/cm.sum()


#%%

# Baseline RF

#Fit the model
slater_loss = make_scorer(custom_loss, greater_is_better=True)

#%%
# PCA with no components

pca = PCA().fit(X_train_sc)
pca_trans = pca.transform(X_train_sc)

print("\nThe components are as follows:\n {}".format(pca.components_))
print("\nThe explained variance is :\n {}".format(pca.explained_variance_))


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("PCA Analysis")
plt.xlabel('number of components',fontsize=15)
plt.ylabel('cumulative explained variance', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(linewidth=3, color='r', linestyle='--', x = 36, ymin=0)
plt.axhline(y=0.95, xmin=0, color='r', linestyle='--')
plt.show()

#%%
# PCA with 36 components.  Which retains 95% of the variation
pca = PCA(n_components=36).fit(X_train_sc)
X_train_sc_pca = pca.transform(X_train_sc)
X_test_sc_pca = pca.transform(X_test_sc)

# Now rfc on the reduced data
rfc_2 = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
%time rfc_2.fit(X_train_sc_pca, y1)

y_pred_pca = rfc_2.predict(X_test_sc_pca)

# Custom Loss function
slater_loss= make_scorer(custom_loss, greater_is_better=True)

rfc_2_cf = confusion_matrix(y_test, y_pred_pca)
rfc_2_score = cross_val_score(rfc_2, X_test_sc_pca, y_pred_pca, cv=5, scoring=slater_loss)

print("\nRandom Forest w PCA:")
print('Accuracy of RF w PCA: {:.2f}'.format(rfc_2.score(X_test_sc_pca, y_test)*100),'%')
print("Confusion Matrix:\n",rfc_2_cf )
print("Custom Cross Validation Score:\n", rfc_2_score)
print("Classification Report", classification_report(y_test, y_pred_pca))


#%%
# # Randomized searchCV
# np.random.seed(42)
# n_estimators = [np.random.randint(10,100) for _ in range(10)]
# min_samples_split = [np.random.randint(2,20) for _ in range(10)]
# min_samples_leaf = [np.random.randint(2,20) for _ in range(10)]
# max_depth = [np.random.randint(1,20) for _ in range(10)]
# max_features = ['log2','sqrt']
# bootstrap = [True,False]

# param_dict = {'n_estimators':n_estimators,
# 				'min_samples_split':min_samples_split,
# 				'min_samples_leaf':min_samples_leaf,
# 				'max_depth':max_depth,
# 				'max_features':max_features,
# 				'bootstrap':bootstrap}

# rs = RandomizedSearchCV(rfc_2,
# 						param_dict,
# 						n_iter=50,
# 						cv=3,
# 						verbose = 1,
# 						n_jobs=-1,
# 						random_state = 0)

# %time rs.fit(X_train_sc_pca, y_train)
# rs.best_params_
# # %%
# rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
# rs_df = rs_df.drop([
#             'mean_fit_time',
#             'std_fit_time',
#             'mean_score_time',
#             'std_score_time',
#             'params',
#             'split0_test_score',
#             'split1_test_score',
#             'split2_test_score',
#             'std_test_score'],
#             axis=1)
# rs_df.head(10)

#%%
# Predictions for each model
# y_pred = rfc_1.predict(X_test_sc)
# y_pred_pca = rfc_2.predict(X_test_sc_pca)
# y_pred_rs = rs.best_estimator_.predict(X_test_sc_pca)


#%%
# from sklearn.metrics import make_scorer
# from sklearn.metrics import confusion_matrix

# # slater_loss_base_rf = custom_loss(y_test, y_pred)



# print("Baseline Random Forest:")
# print("Custom Cross Validation Score:\n", rfc_1_score)
# print("\nRandom Forest w PCA:")
# print("Custom Cross Validation Score:\n", rfc_2_score)
#Looking at confusion matrix

# conf_M_base = pd.DataFrame(confusion_matrix(y_test, y_pred),
# 							index = ['Actual 0','Actual 1'],
# 							columns = ['Predicted 0','Predicted 1'])

# conf_M_w_pca = pd.DataFrame(confusion_matrix(y_test, y_pred_pca),
# 							index = ['Actual 0','Actual 1'],
# 							columns = ['Predicted 0','Predicted 1'])

# conf_M_base = pd.DataFrame(confusion_matrix(y_test,y_pred_rs),
# 							index = ['Actual 0','Actual 1'],
# 							columns = ['Predicted 0','Predicted 1'])


# %%
#Log reg with test/train

#Reimporting the dat
X = df.drop('y', axis = 1)
y = df['y']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state = 42)

X1_train = X_train.copy()
y1_train = y_train.copy()


# Drop vars
drop_col = ['day','month','continent', 'x2','x41']

# Dropping from xtrain and xtest
X1_train = X1_train.drop(drop_col, axis=1)
X1_test = X_test.drop(drop_col, axis=1)

#Scaling
scaler = StandardScaler()
X1_train_sc = scaler.fit_transform(X1_train)
X1_test_sc = scaler.transform(X1_test)
y1_train = np.array(y1_train)

def custom_loss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    weight = np.array([[0, 10], [500, 0]])
    out = cm * weight
    return out.sum()/cm.sum()


# Logistic Regression
lr_1 = LogisticRegression(penalty='l2')
lr_1.fit(X1_train_sc, y1_train)
y_pred = lr_1.predict(X1_test_sc)

slater_loss = make_scorer(custom_loss, greater_is_better=True)
lr_1_score = cross_val_score(lr_1,
							StandardScaler().fit_transform(X1),
							y1,
							cv=5,
							scoring = slater_loss,
							n_jobs=-1,
							verbose=1)

lr_confusion = confusion_matrix(y_test, y_pred)



print("Baseline Logistic Regression:")
print('Accuracy of Logistic Regression: {:.2f}'.format(lr_1.score(X1_test_sc, y_test)*100),'%')
print("Confusion Matrix:\n", lr_confusion)
print("Custom Cross Validation Score:\n", lr_1_score)
print("Classification Report", classification_report(y_test, y_pred))


# Extra trees


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
    return out.sum()/cm.sum() #y_true.shape[0]


slater_loss = make_scorer(custom_loss, greater_is_better=False)


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
    """replace dollar signs and percentages"""
    out = [
        x[i].replace("$", "").replace("%", "").replace("-", "")
        for i in range(x.shape[0])
    ]
    out = np.array([float(z) if _test_value(z) else np.nan for z in out])
    return out


def impute_cats(
    d: Dict[str, np.ndarray], c: Dict[str, Counter]
) -> Dict[str, np.ndarray]:
    """mode impute cats"""
    for k in c.keys():
        d[k][np.isnan(d[k])] = c[k].most_common(1)[0][0]
    return d


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    data = load_data(path)
    data = cleanup(data)
    for k in ["x32", "x37"]:
        data[k] = convert_dollars_percs(data[k])
    # cats are continent, month, day
    cats = [k for k, v in data.items() if not _test_value(v[0])]
    cats.append("x32")
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


def plot_corr(df, plot_path="corr_matrix", fig_size=(12, 8)):
    plt.figure(figsize=fig_size)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        np.round(df.corr(), 3),
        cmap=cmap,
        annot=True,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.savefig(plot_path)


# src: https://pbpython.com/currency-cleanup.html
def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return x.replace("$", "").replace(",", "")
    return float(x)


def p2f(x):
    if isinstance(x, str):
        return float(x.strip("%")) / 100
    return float(x) / 100


# http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
def cyclical(x, period):
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

y = x.pop("y")

y


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
    to_drop = ['x2','x41','x29','x30']
    for k in to_drop:
        d.pop(k, None)
    return d


x_encoded = categorize_with_asia(x)

X = np.column_stack(list(x_encoded.values()))

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

### begin model part 1
X = SimpleImputer().fit_transform(X)
y.shape

np.isnan(X).sum()

from sklearn.metrics import accuracy_score

rfc_1 = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
rfc_1_score = cross_val_score(
    rfc_1, X, y, cv=3, scoring=slater_loss, n_jobs=-1, verbose=1
)
print(rfc_1_score)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
rf.fit(X_train, y_train)
ppp = rf.predict(X_test)

print(confusion_matrix(y_test, ppp))


#
#
#
# print({k:np.unique(v).shape[0] for k, v in x.items() if np.unique(v).shape[0] <=50})
#
# Counter(x['x24'])
#
#
#
#
#
# cat_cols = ['x2','x41','x29','x30']
#
# df = pd.DataFrame(x)
# df = df.drop(['x2','x41','x29','x30'], axis=1)
#
# plot_corr(df, plot_path='corr_matrix_processed',fig_size=(40,35))
#
#
# custom_loss(y_true = np.array([1,0,1,0,1,0]), y_pred = np.array([1,1,1,0,0,0]))
#
# Counter(y)

def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for i in range(X_train.shape[1]):
        save = X_train[:,i].copy()
        X_train[:,i] = np.random.permutation(X_train[:,i])
        m = metric(rf, X_train, y_train)
        X_train[:,i] = save
        imp.append(baseline - m)
    return np.array(imp)

imps = permutation_importances(rf, X_train, y_train, slater_loss)

imps.shape

for i in range(imps.shape[0]):
    plt.barh(i, imps[i])
plt.axvline(imps.mean())
plt.show()

keep_vars = [i for i in range(imps.shape[0]) if imps[i] > imps.mean()]


X_small = X[:,keep_vars].copy()


rfc_s = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
rfc_s_score = cross_val_score(
    rfc_s, X_small, y, cv=5, scoring=slater_loss, n_jobs=-1, verbose=1
)
print(rfc_s_score)

erfc_s = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, verbose=2)
erfc_s_score = cross_val_score(
    erfc_s, X_small, y, cv=5, scoring=slater_loss, n_jobs=-1, verbose=1
)
print(erfc_s_score)
