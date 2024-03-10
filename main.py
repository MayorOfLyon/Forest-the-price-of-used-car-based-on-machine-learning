import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.metrics import mean_absolute_error, make_scorer
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import datetime
from sklearn.model_selection import GridSearchCV
import catboost
from catboost import CatBoostRegressor
from sklearn import linear_model
from bayes_opt import BayesianOptimization

warnings.filterwarnings('ignore')

learning_curve


# model and tune the parameters
#  (1) preprocess the data
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# nonlinear models
data_for_tree = reduce_mem_usage(pd.read_csv('data_for_tree.csv'))

# we train the model with the training data
test_for_tree = data_for_tree[data_for_tree['train'] == 0]
test_for_tree = test_for_tree.drop(['price', 'train'], axis=1)

train_for_tree = data_for_tree[data_for_tree['train'] == 1]
x_train = train_for_tree.drop(['price', 'train'], axis=1)
y_train = train_for_tree['price']

# # 定义交叉验证的目标函数
# def catboost_cv(iterations, learning_rate, depth, subsample, colsample_bylevel, l2_leaf_reg):
#     val = cross_val_score(
#         CatBoostRegressor(iterations=int(iterations),
#                           learning_rate=learning_rate,
#                           depth=int(depth),
#                           subsample=subsample,
#                           colsample_bylevel=colsample_bylevel,
#                           l2_leaf_reg=l2_leaf_reg,
#                           silent=True,  # 阻止打印信息
#                           random_state=42),
#         X=x_train, y=y_train, cv=5, scoring=make_scorer(mean_absolute_error)
#     ).mean()
#     return -val  # 贝叶斯优化会最大化目标函数，所以这里加上负号
#
#
# # 创建贝叶斯优化对象
# catboost_bo = BayesianOptimization(
#     catboost_cv,
#     {
#         'iterations': (1000, 3000),
#         'learning_rate': (0.01, 0.3),
#         'depth': (3, 10),
#         'subsample': (0.7, 1),
#         'colsample_bylevel': (0.7, 1),
#         'l2_leaf_reg': (1, 10)
#     }
# )
#
# # 进行优化
# catboost_bo.maximize(init_points=10, n_iter=30)
#
# # 输出最佳参数和目标函数值
# best_params = catboost_bo.max['params']
# best_target = catboost_bo.max['target']
# print("最佳参数：", best_params)
# print("最佳目标函数值：", -best_target)

# def xgb_cv(max_depth, learning_rate, n_estimators, subsample, colsample_bytree, gamma):
#     val = cross_val_score(
#         XGBRegressor(max_depth=int(max_depth),
#                      learning_rate=learning_rate,
#                      n_estimators=int(n_estimators),
#                      subsample=subsample,
#                      colsample_bytree=colsample_bytree,
#                      gamma=gamma,
#                      random_state=42),
#         X=x_train, y=y_train, cv=5, scoring=make_scorer(mean_absolute_error)
#     ).mean()
#     return -val  # 贝叶斯优化会最大化目标函数，所以这里加上负号
#
#
# # 创建贝叶斯优化对象
# xgb_bo = BayesianOptimization(
#     xgb_cv,
#     {
#         'max_depth': (3, 10),
#         'learning_rate': (0.01, 0.3),
#         'n_estimators': (50, 300),
#         'subsample': (0.7, 1),
#         'colsample_bytree': (0.7, 1),
#         'gamma': (0, 5)
#     }
# )
#
# # 进行优化
# xgb_bo.maximize(init_points=10, n_iter=30)
#
# # 输出最佳参数和目标函数值
# best_params = xgb_bo.max['params']
# best_target = xgb_bo.max['target']
# print("最佳参数：", best_params)
# print("最佳目标函数值：", -best_target)

# def lgb_cv(num_leaves, max_depth, learning_rate, subsample, colsample_bytree, min_child_samples, reg_alpha, reg_lambda):
#     val = cross_val_score(
#         LGBMRegressor(
#             num_leaves=int(num_leaves),
#             max_depth=int(max_depth),
#             learning_rate=learning_rate,
#             subsample=subsample,
#             colsample_bytree=colsample_bytree,
#             min_child_samples=int(min_child_samples),
#             reg_alpha=reg_alpha,
#             reg_lambda=reg_lambda,
#             n_jobs=-1,
#             random_state=42,
#             verbose=0
#         ),
#         X=x_train, y=y_train, cv=5, scoring=make_scorer(mean_absolute_error)
#     ).mean()
#     return -val  # 贝叶斯优化会最大化目标函数，所以这里加上负号


# # 创建贝叶斯优化对象
# lgbm_bo = BayesianOptimization(
#     lgb_cv,
#     {
#         'num_leaves': (24, 1024),
#         'max_depth': (5, 30),
#         'learning_rate': (0.01, 0.3),
#         'subsample': (0.7, 1),
#         'colsample_bytree': (0.7, 1),
#         'min_child_samples': (2, 100),
#         'reg_alpha': (0, 10),
#         'reg_lambda': (0, 10)
#     }
# )
#
# # 进行优化
# lgbm_bo.maximize(init_points=10, n_iter=30)
#
# # 输出最佳参数和目标函数值
# best_params = lgbm_bo.max['params']
# best_target = lgbm_bo.max['target']
# print("最佳参数：", best_params)
# print("最佳目标函数值：", -best_target)

# # stacking
# def Stacking_method(train_pre1, train_pre2, train_pre3, y_train_t,
#                     model=LinearRegression()):
#     model.fit(pd.concat([pd.Series(train_pre1), pd.Series(train_pre2), pd.Series(train_pre3)], axis=1).values,
#               y_train_t)
#     return model


# xgb
xgb = XGBRegressor(colsample_bytree=0.9778647892857296, gamma=0.5103670062050109,
                   learning_rate=0.06781018807817607, max_depth=9,
                   n_estimators=258, subsample=0.8182314719391712)
xgb = xgb.fit(x_train, y_train)
y_train_xgb = xgb.predict(x_train)
y_test_xgb = xgb.predict(test_for_tree)
mae_xgb = cross_val_score(xgb, x_train, y_train, cv=5, scoring=make_scorer(mean_absolute_error))
print(mae_xgb)
print("MAE of xgb is :", mae_xgb.mean())

# lgb
lgb = LGBMRegressor(colsample_bytree=0.9083273674936061, learning_rate=0.13235329056696224,
                    max_depth=25,
                    min_child_samples=70, num_leaves=779,
                    reg_alpha=6.413209047808175, reg_lambda=8.339089484535307, subsample=0.79037301878376)
lgb = lgb.fit(x_train, y_train)
y_train_lgb = lgb.predict(x_train)
y_test_lgb = lgb.predict(test_for_tree)
mae_lgb = cross_val_score(lgb, x_train, y_train, cv=5, scoring=make_scorer(mean_absolute_error))
print(mae_lgb)
print("MAE of lgb is :", mae_lgb.mean())

# catboost
cat = CatBoostRegressor(colsample_bylevel=0.7668410394301646, depth=8,
                        iterations=2890, l2_leaf_reg=1.63867705141528,
                        learning_rate=0.05049519256369473, subsample=0.9218621753718955)
cat = cat.fit(x_train, y_train)
y_train_cat = cat.predict(x_train)
y_test_cat = cat.predict(test_for_tree)
mae_cat = cross_val_score(cat, x_train, y_train, cv=5, scoring=make_scorer(mean_absolute_error))
print(mae_cat)
print("MAE of cat is :", mae_cat.mean())


# stacking
train_stack = pd.concat([pd.Series(y_train_xgb), pd.Series(y_train_lgb),
                         pd.Series(y_train_cat)], axis=1)
test_stack = pd.concat([pd.Series(y_test_xgb), pd.Series(y_test_lgb),
                        pd.Series(y_test_cat)], axis=1)
#
# model = LinearRegression()
# final_model = Stacking_method(y_train_xgb, y_train_lgb, y_train_cat, y_train, model)
# predictions_test = final_model.predict(test_stack)
# mae_final = cross_val_score(final_model, train_stack, y_train, cv=5, scoring=make_scorer(mean_absolute_error))
# print(mae_final)
# print("MAE of final model(Linear) is :", mae_final.mean())
#
# stacking
Bayes = linear_model.BayesianRidge()
Bayes.fit(train_stack, y_train)
predictions_test = Bayes.predict(test_stack)
predictions_train = Bayes.predict(train_stack)
mae_final = cross_val_score(Bayes, train_stack, y_train, cv=5, scoring=make_scorer(mean_absolute_error))
print(mae_final)
print("MAE of final model(Bayes) is :", mae_final.mean())

# 6. final
sub = pd.DataFrame()
sub['SaleID'] = np.arange(200000, 250000)
sub['price'] = np.expm1(predictions_test)
sub.to_csv('./sub_Weighted.csv', index=False)
