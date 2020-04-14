#%%
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
rfc_1 = RandomForestClassifier(n_estimators = 300, n_jobs = -1)
%time rfc_1.fit(X_train_sc, y1)
# display(rfc_1.score(X_train_sc, y_train))
# Get predictions
y_pred = rfc_1.predict(X_test_sc)

#Custom Loss Function
slater_loss = make_scorer(custom_loss, greater_is_better=True)
rfc_1_score = cross_val_score(rfc_1, X_test_sc, y_pred, cv=5, scoring = slater_loss, n_jobs=-1, verbose=1)

rfc_1_cf = confusion_matrix(y_test,y_pred)

print("Baseline Random Forest:")
print('Accuracy of Baseline RF: {:.2f}'.format(rfc_1.score(X_test_sc, y_test)*100),'%')
print("Confusion Matrix:", rfc_1_cf)
print("Custom Cross Validation Score:\n", rfc_1_score)
print("Classification Report", classification_report(y_test, y_pred))


#%%
feats = {}
for feature, importance in zip(X1.columns, rfc_1.feature_importances_):
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
plt.show()
# display(importances)



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
							X1_train_sc, 
							y1_train,
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

#fuck
# ehh work




#WORK GIT


