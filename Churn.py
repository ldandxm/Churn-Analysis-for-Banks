
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
###### Explore the Data #####


# In[3]:


dataset = pd.read_csv("train.csv")


# In[4]:


dataset.head()


# In[5]:


dataset.dtypes


# In[6]:


# Correlation Matrix


# In[7]:


# Correlation of Numerical Features
# Exclude ID and SalesPrice
corr = dataset.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize = (16, 10))
sns.heatmap(corr, vmax = 1, vmin = -1, square = True)


# In[8]:


# Hist of Saleprice

sns.distplot(dataset['SalePrice'])


# In[9]:


# QQ-plot for SalePrice
fig = plt.figure(figsize = (6,6))
stats.probplot(dataset['SalePrice'], plot = plt)


# In[10]:


# Missing Data
total = dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent Missing'])


# In[11]:


missing_data = missing_data[missing_data['Total Missing'] != 0]
missing_data


# In[12]:


dataset[dataset['PoolQC'].notnull()]


# In[13]:


dataset[dataset['PoolArea'] !=0]


# In[14]:


# We have 2 cols about Pool: Pool Area (Numeric) and PoolQC (Categorical)
# There are only 7 houses have pool
# We can combine these 2 predictor into 1 by simply multiply them


# In[15]:


# Encoding PoolQC
dataset['PoolQC'] = dataset['PoolQC'].fillna("None")


# In[16]:


dataset['PoolQC'] = dataset['PoolQC'].map({'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})


# In[17]:


dataset['PoolQC'].value_counts()


# In[18]:


dataset['PoolScore'] = dataset['PoolArea'] * dataset['PoolQC']


# In[19]:


dataset = dataset.drop(columns = ['PoolArea', 'PoolQC'])


# In[20]:


sns.barplot(x = 'MiscFeature', y = 'SalePrice', data = dataset[dataset['MiscFeature'].notnull()][['MiscFeature', 'SalePrice']], palette='Blues_d')


# In[21]:


# MiscFeature looks irrelevant, especially we have column MiscVal, which seems to make more sense. So remove.


# In[22]:


dataset = dataset.drop(columns = ['MiscFeature'])


# In[23]:


# Let's also drop ID
dataset = dataset.drop(columns = ['Id'])


# In[24]:


# Encoding Alley
dataset['Alley'] = dataset['Alley'].fillna("None")
dataset['Alley'] = dataset['Alley'].map({'None': 0, 'Grvl': 1, 'Pave': 2})


# In[25]:


dataset['Alley'].value_counts()


# In[26]:


# Encoding Fence:
dataset['Fence'] = dataset['Fence'].fillna("None")
dataset['Fence'] = dataset['Fence'].map({'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4})


# In[27]:


dataset['Fence'].value_counts()


# In[28]:


# Encoding FireplaceQu
dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna("None")
dataset['FireplaceQu'] = dataset['FireplaceQu'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})


# In[29]:


dataset['FireplaceScore'] = dataset['FireplaceQu'] * dataset['Fireplaces']
dataset = dataset.drop(columns = ['FireplaceQu', 'Fireplaces'])


# In[30]:


# LotFrontage
# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
dataset['LotFrontage'] = dataset.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))


# In[31]:


# Garage
# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None


# In[32]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    dataset[col] = dataset[col].fillna('None')


# In[33]:


# GarageType has no order? Decide later
# GarageFinish
dataset['GarageFinish'] = dataset['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
# GarageQual
dataset['GarageQual'] = dataset['GarageQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
# GarageCond
dataset['GarageCond'] = dataset['GarageCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})


# In[34]:


# GarageYrBlt: Replacing missing data with 0
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)


# In[35]:


# Basement
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2: Replacing missing data with None
# Record 948 showing no bsmt in BsmtExposure but has values in other cols
dataset[dataset['BsmtExposure'].isnull()][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]


# In[36]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    dataset[col] = dataset[col].fillna('None')


# In[37]:


# BsmtQual
dataset['BsmtQual'] = dataset['BsmtQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
# BsmtCond
dataset['BsmtCond'] = dataset['BsmtCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
# BsmtExposure
dataset['BsmtExposure'] = dataset['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
# BsmtFinType1
dataset['BsmtFinType1'] = dataset['BsmtFinType1'].map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
# BsmtFinType2
dataset['BsmtFinType2'] = dataset['BsmtFinType2'].map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})


# In[38]:


# MasVnr
# MasVnrType: replace missing data with none. Seems no order, Hot Encode later
dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')
# MasVnrArea: replace missing data with 0.
dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)


# In[39]:


# Electrical: only 1 missing data, just assign to the mode.
dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])


# In[40]:


# Great! Now we don't have any missing values
dataset.isnull().sum().sum()


# In[41]:


dataset.dtypes[dataset.dtypes == 'object'] 


# In[72]:


# Now let's check if any remaining categorical variables should have ordered label
# We will manually fix those and hot encode the rest
dataset.isnull().sum().sum()


# In[43]:


# Street
dataset['Street'] = dataset['Street'].map({'Grvl': 1, 'Pave': 2})


# In[44]:


# LotShape
dataset['LotShape'] = dataset['LotShape'].map({'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4})


# In[45]:


# Utilities
dataset['Utilities'] = dataset['Utilities'].map({'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4})


# In[46]:


# LandSlope
dataset['LandSlope'] = dataset['LandSlope'].map({'Sev': 1, 'Mod': 2, 'Gtl': 3})


# In[47]:


# ExterQual
dataset['ExterQual'] = dataset['ExterQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex':5})


# In[48]:


# ExterCond
dataset['ExterCond'] = dataset['ExterCond'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex':5})


# In[49]:


# HeatingQC
dataset['HeatingQC'] = dataset['HeatingQC'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex':5})


# In[50]:


# CentralAir
dataset['CentralAir'] = dataset['CentralAir'].map({'N': 0, 'Y': 1})


# In[51]:


# KitchenQual
dataset['KitchenQual'] = dataset['KitchenQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex':5})


# In[52]:


# Functional
dataset['Functional'] = dataset['Functional'].map({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8})


# In[71]:


# GarageType
dataset['GarageType'] = dataset['GarageType'].map({'None': 0, 'Detchd': 1, 'CarPort': 2, 'BuiltIn': 3, 'Basment': 4, 'Attchd': 5, '2Types': 6})


# In[54]:


# PavedDrive
dataset['PavedDrive'] = dataset['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})


# In[73]:


# Changing Some Numericals into categoricals
# Year and Month Sold
dataset['YrSold'] = dataset['YrSold'].astype(str)
dataset['MoSold'] = dataset['MoSold'].astype(str)


# In[74]:


# MSSubClass
dataset['MSSubClass'] = dataset['MSSubClass'].apply(str)


# In[75]:


dataset.dtypes[dataset.dtypes == 'object']


# In[58]:





# In[78]:


##### Finally Model ######
train = pd.get_dummies(dataset, columns=list(dataset.select_dtypes(include=['category','object'])))


# In[81]:


# corr_cleaned = dataset.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
# plt.figure(figsize = (16, 10))
# sns.heatmap(corr_cleaned, vmax = 1, vmin = -1, square = True)


# In[87]:


y_train = train['SalePrice'].values
y_train = np.log1p(y_train)
X_train = train.drop(columns = ['SalePrice']).values


# In[82]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# In[88]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[89]:


# Lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[65]:


# # Elastic Net
# ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# In[66]:


# # Kernel Ridge Regression
# KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# In[90]:


# XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[68]:


# score = rmsle_cv(lasso)
# print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

