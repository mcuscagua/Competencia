import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
from collections import Counter
import operator
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import roc_auc_score


num_sim_models = 1000

Categorical = np.array([str(x) for x in X.dtypes.values])


# More Functions

def run_randomForests(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4, n_jobs=8)
    rf.fit(X_train, y_train)
    pred_train = rf.predict_proba(X_train)
    pred_test = rf.predict_proba(X_test)

    return (roc_auc_score(y_train, pred_train[:,1]), roc_auc_score(y_test, pred_test[:,1]))

# Mutual Information Criteria
MutInfoVar = np.array([])
mutual_info = pd.Series(mutual_info_classif(X,Y))
mutual_info.index = X.columns
mutual_info = mutual_info.sort_values(ascending=False)

for i in range(num_sim_models):
    MutInfoVar = np.concatenate([MutInfoVar, X.columns[SelectPercentile(mutual_info_classif, percentile=50).fit(X, Y).get_support()].values])

MI_Freq = sorted(Counter(MutInfoVar).items(), key=operator.itemgetter(1), reverse=True)
Sort_Var = np.array([x[0] for x in MI_Freq])
Sort_Val = np.array([x[1]/num_sim_models for x in MI_Freq])

MutInfoDef = Sort_Var[Sort_Val > 0.5]

# MSE Feature Selection
roc_values = []
for feature in X.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X[feature].to_frame(), Y)
    y_scored = clf.predict_proba(Xt[feature].to_frame())
    roc_values.append(roc_auc_score(Yt, y_scored[:, 1]))

roc_values = pd.Series(roc_values)
roc_values.index = Xt.columns
roc_values = roc_values.sort_values(ascending=False)

MSEDef = roc_values.index[roc_values > 0.5].values

# Fisher score
f_score = chi2(X[X.columns[Categorical == 'int64'].values], Y)
pvalues = pd.Series(f_score[1])
pvalues.index = X.columns[Categorical == 'int64'].values
pvalues = pvalues.sort_values(ascending= True)

FisScoDef = pvalues.index[pvalues <= 0.05].values

# Step Forward Selection
N_feat = [15, 16, 17, 18, 19, 20, 21, 22]
selected_features = []
for i in N_feat:
    sfs1 = SFS(RandomForestClassifier(n_estimators=200),
               n_jobs=-1,
               k_features=i,
               forward=True,
               floating=False,
               verbose=2,
               scoring='roc_auc',
               cv=5)

    sfs1 = sfs1.fit(X,Y)
    selected_feat = X.columns[list(sfs1.k_feature_idx_)]
    selected_features.append(selected_feat.values)

SF_Score = []

for selected_feat in selected_features:
    SF_Score.append(run_randomForests(X[selected_feat], Xt[selected_feat], Y, Yt)[1])

SFSDef = selected_features[np.argmax(SF_Score)]

# Step Backward Selection
N_feat = [15, 16, 17, 18, 19, 20, 21, 22]
selected_features = []
for i in N_feat:
    sfs1 = SFS(RandomForestClassifier(n_estimators=200),
               n_jobs=-1,
               k_features=i,
               forward=False,
               floating=False,
               verbose=2,
               scoring='roc_auc',
               cv=5)

    sfs1 = sfs1.fit(X,Y)
    selected_feat = X.columns[list(sfs1.k_feature_idx_)]
    selected_features.append(selected_feat.values)

SB_Score = []

for selected_feat in selected_features:
    SB_Score.append(run_randomForests(X[selected_feat], Xt[selected_feat], Y, Yt)[1])

SBSDef = selected_features[np.argmax(SB_Score)]

# Lasso
SFM_LR = SelectFromModel(LogisticRegression())
RL = SFM_LR.fit(X, Y)
LRDef = X.columns[(SFM_LR.get_support())].values

## Elasticnet Regularization

#SFM_EN = SelectFromModel(ElasticNet())
#EN = SFM_EN.fit(X, Y)
#ENDef = X.columns[(SFM_EN.get_support())].values


# Exhaustive Feature Selection
efs1 = EFS(RandomForestClassifier(random_state=0), n_jobs=8, min_features=1, max_features=12,
           scoring='roc_auc',print_progress=True, cv=2)
efs1 = efs1.fit(X[X.columns[0:15]], Y)
EFSDef = X.columns[list(efs1.best_idx_)].values

# Random Forest Importance
RFI = SelectFromModel(RandomForestClassifier(n_estimators=200, min_samples_split=5))
RFI.fit(X, Y)
RFIDef = X.columns[(RFI.get_support())].values

# Recursive Feature Selection with Random Forest Importance
RFSRFI = RFECV(estimator=RandomForestClassifier(n_estimators=200, min_samples_split=5))
RFSRFI.fit(X, Y)
RFSRFIDef = X.columns[(RFSRFI.get_support())].values

# Gradient Boosted Trees Importance
GBTIVar = np.array([])
for i in range(num_sim_models):
    GBTI = SelectFromModel(GradientBoostingClassifier())
    GBTI.fit(X, Y)
    GBTIVar = np.concatenate([GBTIVar, X.columns[(GBTI.get_support())].values])

GBTI_Freq = sorted(Counter(GBTIVar).items(), key=operator.itemgetter(1), reverse=True)
Sort_Var = np.array([x[0] for x in GBTI_Freq ])
Sort_Val = np.array([x[1]/num_sim_models for x in GBTI_Freq ])

GBTIDef = Sort_Var[Sort_Val > 0.5]

# Recursive Feature Selection with Gradient Boosted
sel_ = RFE(GradientBoostingClassifier(), n_features_to_select=len(GBTIDef))
sel_.fit(X, Y)
selected_feat_rfe = X.columns[(sel_.get_support())]
selected_feat_rfe