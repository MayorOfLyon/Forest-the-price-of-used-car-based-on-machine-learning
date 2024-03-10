import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.metrics import mean_absolute_error, make_scorer
from lightgbm.sklearn import LGBMRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
import os
from scipy import stats
from sklearn.decomposition import PCA

# 1. load the training and test data
current_path = os.path.dirname(os.path.abspath(__file__))
relative_path_train = 'used_car_train_20200313.csv'
relative_path_testB = 'used_car_testB_20200421.csv'
path_train = os.path.join(current_path, relative_path_train)
path_testB = os.path.join(current_path, relative_path_testB)
Test_data = pd.read_csv(path_testB, sep=' ')
Train_data = pd.read_csv(path_train, sep=' ')

# 2. EDA
# # (1) preliminary observation of the data
# merge_train_data = Train_data.head(5).append(Train_data.tail(5))
# print(merge_train_data)
# print('Train data shape:', Train_data.shape)

# merge_test_data = Test_data.head(5).append(Test_data.tail(5))
# print(merge_test_data)
# print('TestB data shape:', Test_data.shape)

# (2) an overview of the data
# train_description = Train_data.describe()
# print(train_description)
# test_description = Test_data.describe()
# print(test_description)

# train_info = Train_data.info()
# print(train_info)
# test_info = Test_data.info()
# print(train_info)

# (3) check the situation of the missing data
# train_missing = Train_data.isnull().sum()
# print(train_missing)
# test_missing = Test_data.isnull().sum()
# print(test_missing)
# train_missing = train_missing[train_missing > 0]
# train_missing.sort_values(inplace=True)
# train_missing.plot.bar()

# msno.bar(Train_data.sample(1000))
# msno.bar(Test_data.sample(1000))
# plt.show()

# print(Train_data['notRepairedDamage'].value_counts())
# print(Train_data['notRepairedDamage'].value_counts())
# print(Train_data.isnull().sum())
# print(Test_data.isnull().sum())

# # (4)check data bias situation
# for col in Train_data.columns:
#     value_count = Train_data[col].value_counts()
#     print(f"Value counts for column '{col}':")
#     print(value_count)
#     print("\n")
# for col in Test_data.columns:
#     value_count = Test_data[col].value_counts()
#     print(f"Value counts for column '{col}':")
#     print(value_count)
#     print("\n")

# offerType and seller exhibit significant data bia ,and we should delete them from the features


# def outliers_proc(data, col_name, scale=3):
#     def box_plot_outliers(data_ser, box_scale):
#         iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
#         val_low = data_ser.quantile(0.25) - iqr
#         val_up = data_ser.quantile(0.75) + iqr
#         rule_low = (data_ser < val_low)
#         rule_up = (data_ser > val_up)
#         return (rule_low, rule_up), (val_low, val_up)
#
#     data_n = data.copy()
#     data_series = data_n[col_name]
#     rule, value = box_plot_outliers(data_series, box_scale=scale)
#     index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
#     # print(col_name)
#     # print("Delete number is: {}".format(len(index)))
#     data_n.loc[rule[0] | rule[1], col_name] = np.nan
#     index_low = np.arange(data_series.shape[0])[rule[0]]
#     outliers = data_series.iloc[index_low]
#     # print("Description of data less than the lower bound is:")
#     # print(pd.Series(outliers).describe())
#     index_up = np.arange(data_series.shape[0])[rule[1]]
#     outliers = data_series.iloc[index_up]
#     # print("Description of data larger than the upper bound is:")
#     # print(pd.Series(outliers).describe())
#
#     # fig, ax = plt.subplots(1, 2, figsize=(10, 7))
#     # sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
#     # sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
#     return data_n


# for feature in ['power', 'kilometer', 'v_0', 'v_1', 'v_2',
#                 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
#                 'v_13', 'v_14']:
#     Train_data = outliers_proc(Train_data, feature, scale=3)

# original_price = Train_data['price']
# transformed_price, lambda_param = stats.boxcox(original_price)
# Train_data['price'] = transformed_price
# uniform
# # report
# pfr = pandas_profiling.ProfileReport(Train_data)
# pfr.to_file("./example.html")

# uniform

Train_data['price'] = np.log1p(Train_data['price'])
Train_data.drop(Train_data[Train_data['price'] < 4].index, inplace=True)

# combine
Train_data['train'] = 1
Test_data['train'] = 0
data = pd.concat([Train_data, Test_data], ignore_index=True)

# handle the missing values
# print(Train_data.columns[Train_data.isnull().any()])
data['model'] = data['model'].fillna(0)
data['gearbox'] = data['gearbox'].fillna(0)
data['bodyType'] = data['bodyType'].fillna(0)
data['fuelType'] = data['fuelType'].fillna(0)

# handle the outliers
data['notRepairedDamage'].replace('-', np.nan, inplace=True)
data['notRepairedDamage'] = data['notRepairedDamage'].fillna(0)
data['power'] = data['power'].apply(lambda x: 600 if x > 600 else x)

# datetime
data['used_time'] = (
        pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') - pd.to_datetime(data['regDate'],
                                                                                             format='%Y%m%d',
                                                                                             errors='coerce')).dt.days
mean_value = data['used_time'].mean()
data['used_time'].fillna(mean_value, inplace=True)

# region
data['city'] = data['regionCode'].apply(lambda x: str(x)[:2])

# binning
bin_edge = np.linspace(0, 300, 31)
data['power_bin'] = pd.cut(data['power'], bin_edge, labels=False)

# delete the useless features
data = data.drop(['regDate', 'creatDate', 'seller', 'offerType', 'name', 'SaleID'], axis=1)

# # features crossover
train_cro = Train_data.groupby("brand")
all_info = {}
for kind, kind_data in train_cro:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})
data = data.merge(brand_fe, how='left', on='brand')

train_cro = Train_data.groupby("regionCode")
all_info = {}
for kind, kind_data in train_cro:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['regionCode_amount'] = len(kind_data)
    info['regionCode_price_max'] = kind_data.price.max()
    info['regionCode_price_median'] = kind_data.price.median()
    info['regionCode_price_min'] = kind_data.price.min()
    info['regionCode_price_sum'] = kind_data.price.sum()
    info['regionCode_price_std'] = kind_data.price.std()
    info['regionCode_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "regionCode"})
data = data.merge(brand_fe, how='left', on='regionCode')

train_cro = Train_data.groupby("model")
all_info = {}
for kind, kind_data in train_cro:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['model_amount'] = len(kind_data)
    info['model_price_max'] = kind_data.price.max()
    info['model_price_median'] = kind_data.price.median()
    info['model_price_min'] = kind_data.price.min()
    info['model_price_sum'] = kind_data.price.sum()
    info['model_price_std'] = kind_data.price.std()
    info['model_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "model"})
data = data.merge(brand_fe, how='left', on='model')

# generate polynomial features
numeric_fea = ['power', 'kilometer', 'v_0', 'v_1', 'v_2',
               'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
               'v_13', 'v_14', 'used_time']
numeric_data = data[numeric_fea]
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
data = data.drop(numeric_fea, axis=1)
numeric_poly = poly.fit_transform(numeric_data)
num_poly_data = pd.DataFrame(numeric_poly, columns=poly.get_feature_names_out())
data = pd.concat([data, num_poly_data], axis=1)

# One code
data = pd.get_dummies(data, columns=['city', 'notRepairedDamage', 'power_bin', 'model', 'fuelType', 'bodyType'])

# divide the data into the training set and the test set
data_train = data[data['train'] == 1]
data_test = data[data['train'] == 0]
data_train_x = data_train.drop('price', axis=1)
data_train_y = data_train['price']

# # # PCA
# dims = range(80, 100)
# scores = []
# for dim in dims:
#     pca = PCA(n_components=dim)
#     preprocessed_data = pca.fit_transform(data_train_x)
#     lgb = LGBMRegressor()
#     score = np.mean(cross_val_score(lgb, X=preprocessed_data, y=data_train_y, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
#     print(score)
#     scores.append(score)
# plt.plot(dims, scores)
# plt.xlabel('Number of PCA components')
# plt.ylabel('Accuracy')
# plt.show()
# print(scores.min())
# # # Select from model
lgb = LGBMRegressor()
selector = SelectFromModel(lgb)
X_sfm_gbdt = selector.fit_transform(data_train_x, data_train_y)
scores = cross_val_score(lgb, X=X_sfm_gbdt, y=data_train_y, verbose=0, cv=5,
                         scoring=make_scorer(mean_absolute_error))
print(np.mean(scores))

# # sfs
# sfs = SFS(LGBMRegressor(),
#           k_features=30,
#           forward=True,
#           floating=False,
#           scoring=make_scorer(mean_absolute_error),
#           cv=0)
# sfs.fit(data_train_x, data_train_y)
# print(sfs.k_feature_names_)
# fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
# plt.grid()
# plt.show()
# scores = cross_val_score(lgb, X=data_train_x[sfs.k_feature_names_], y=data_train_y, verbose=0, cv=5,
#                          scoring=make_scorer(mean_absolute_error))
# print(np.mean(scores))

# export the data
select_features_indices = selector.get_support(indices=True)
select_features_names = data_train_x.columns[select_features_indices]
select_features_names = select_features_names.union(['price', 'train'])
data = pd.concat([data_train, data_test])
data = data[select_features_names]
print(data.shape)
data.to_csv('data_for_tree.csv', index=0)
