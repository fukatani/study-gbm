'''
simple xgboost benchmark
'''
from rgf import RGFClassifier, FastRGFClassifier
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from util import Gini
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from util import proj_num_on_cat, cat_count, cat_count_train
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse

cv_only = True
save_cv = True
full_train = False

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
# max_bin = x
feature_fraction = 1.0
num_boost_round = 190

train = pd.read_csv("../input/train.csv")
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

cat_list = [x for x in list(train) if 'cat' in x]
print(cat_list)

train_copy = train.copy()
train_copy = train_copy.replace(-1, np.NaN)
train['num_na'] = train_copy.isnull().sum(axis=1)
del train_copy

ohe = OneHotEncoder(sparse=False)
train_cat = train[[x for x in list(train) if 'cat' in x]].as_matrix()
train_num = train[[x for x in list(train) if 'cat' not in x]]
train_cat[train_cat < 0] = 99

train_ohe = ohe.fit_transform(train_cat)

print("cat_list now:", cat_list)
train_cat_count = cat_count_train(train, cat_list)
print("cat count shape:", train_cat_count.shape)

# X = sparse.hstack([train_num, train_ohe, train_cat_count]).tocsr()
# X = np.hstack([train_num, train_ohe, train_cat_count])
X = np.hstack([train_num, train_cat_count])
print(X.shape)

params = {"objective": "binary",
          "boosting_type": "rgf",  # 0.00438164253256, Bug
          # "boosting_type": "gbdt",  # 0.284558375695
          "learning_rate": learning_rate,
          "num_leaves": int(num_leaves),
          "max_bin": 128,
          "min_data_in_leaf": min_data_in_leaf,
          "feature_fraction": feature_fraction,
          "verbosity": 1,
          "seed": 218,
          "drop_rate": 0.1,
          "is_unbalance": False,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 1.0,
          "num_trees": 99,
          "num_threads": 4,
          }

x_score = []
final_cv_train = np.zeros(len(train_label))

iter_num = 1

for s in range(iter_num):
    cv_train = np.zeros(len(train_label))
    params['seed'] = s

    if cv_only:
        kf = kfold.split(X, train_label)
        fold_scores = []
        for i, (train_fold, validate) in enumerate(kf):
            print(str(i) + "th round.")
            if i >= 1:
                continue
            X_train, X_validate, label_train, label_validate = \
                X[train_fold, :], X[validate, :], train_label[train_fold], train_label[validate]
            rgf = RGFClassifier(max_leaf=500,
                                l2=0.01,
                                min_samples_leaf=20,
                                opt_interval=100,
                                )
            # rgf = FastRGFClassifier()
            # rgf = RandomForestClassifier()
            # rgf.fit(X_train, label_train)
            rgf.fit(X_validate, label_validate)
            rgf.fit(X_validate[:40000, :], label_validate[:40000])
            cv_train[validate] += rgf.predict(X_validate)

            score = Gini(label_validate, cv_train[validate])
            print(score)
            fold_scores.append(score)

        final_cv_train += cv_train

        print("cv score:")
        print(Gini(train_label, cv_train))
        print("current score:", Gini(train_label, final_cv_train / (s + 1.)), s+1)
        print(fold_scores)
        x_score.append(Gini(train_label, cv_train))

print(x_score)
pd.DataFrame({'id': train_id, 'target': 1.0 * final_cv_train / iter_num}).to_csv('../model/lgbm1_cv_avg.csv', index=False)
