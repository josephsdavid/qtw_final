#%%
from sklearn.metrics import mean_squared_error, r2_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

%time df = pd.read_csv("../Data/final_project.csv", sep=",", header=0)

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

for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].fillna(df[col].median())

#%%
#Check numerical histograms of data
df.hist(bins=50, figsize = (20,15))

#%%

#heatmap
plt.figure(figsize=(20,10))
sns.heatmap(df.corr().round(1),vmax=1, annot=True, cmap = 'YlGnBu',annot_kws={"fontsize":10})


#%%
X = df.drop('y', axis = 1)
y = df['y']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state = 42)

print("\nChecking shape of test/train data")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



#%%

#Scaling Data and prepping for RF and PCA

df_pca = X_train.copy()
y = y_train.copy()
cat_var = ['day','month','continent']
#Dropping categorical for scaling
#Maybe should one-hot encode.

X = df_pca.drop(cat_var, axis=1)
X_test = X_test.drop(cat_var, axis=1)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X)
X_test_sc = scaler.transform(X_test)
y_train = np.array(y_train)

#%%

# Baseline RF

rfc_1 = RandomForestClassifier()
%time rfc_1.fit(X_train_sc, y_train)
display(rfc_1.score(X_train_sc, y_train))
#%%
feats = {}
for feature, importance in zip(df.columns, rfc_1.feature_importances_):
    feats[feature] = importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
importances = importances.sort_values(by='Gini-Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})

sns.set(font_scale = 5)
sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
fig, ax = plt.subplots()
fig.set_size_inches(20,15)
sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
plt.xlabel('Importance', fontsize=25, weight = 'bold')
plt.ylabel('Features', fontsize=25, weight = 'bold')
plt.title('Feature Importance', fontsize=25, weight = 'bold')
display(plt.show())
#display(importances)



#%%
# PCA with no components

pca = PCA().fit(X_train_sc)
pca_trans = pca.transform(X_train_sc)

print("\nThe components are as follows:\n {}".format(pca.components_))
print("\nThe explained variance is :\n {}".format(pca.explained_variance_))


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.axvline(linewidth=3, color='r', linestyle='--', x = 36, ymin=0)
plt.axhline(y=0.95, xmin=0, color='r', linestyle='--')
plt.show()

#%%
# PCA with 36 components.  Which retains 95% of the variation
pca = PCA(n_components=36).fit(X_train_sc)
X_train_sc_pca = pca.transform(X_train_sc)
X_test_sc_pca = pca.transform(X_test_sc)

# Now rfc on the reduced data
rfc_2 = RandomForestClassifier()
%time rfc_2.fit(X_train_sc_pca, y_train)
display(rfc_2.score(X_train_sc_pca, y_train))


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
y_pred = rfc_1.predict(X_test_sc)
y_pred_pca = rfc_2.predict(X_test_sc_pca)
# y_pred_rs = rs.best_estimator_.predict(X_test_sc_pca)


#%%
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
def custom_loss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    weight = np.array([[0, 10], [500, 0]])
    out = cm * weight
    return out.sum()
slater_loss = make_scorer(custom_loss, greater_is_better=False)

#Looking at confusion matrix

conf_M_base = pd.DataFrame(custom_loss(y_test, y_pred),
							index = ['Actual 0','Actual 1'],
							columns = ['Predicted 0','Predicted 1'])

conf_M_w_pca = pd.DataFrame(custom_loss(y_test, y_pred_pca),
							index = ['Actual 0','Actual 1'],
							columns = ['Predicted 0','Predicted 1'])

# conf_M_base = pd.DataFrame(confusion_matrix(y_test,y_pred_rs),
# 							index = ['Actual 0','Actual 1'],
# 							columns = ['Predicted 0','Predicted 1'])

print("Baseline Random Forrest:\n",conf_M_base)
print()
print("Random Forrest w PCA:\n",conf_M_w_pca)



# %%
