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

num_sim_models = 1000

# Variables per model
MutInfoVar = np.array([])

# Mutual Information Criteria
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
roc_values.sort_values(ascending=False)