#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time 
import warnings 
from pathlib import Path
from sklearn.metrics import accuracy_score

from scipy.stats import pearsonr, chi2_contingency, uniform, loguniform
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from imblearn.ensemble     import BalancedRandomForestClassifier
from sklearn.feature_selection import SelectKBest
from scipy.sparse import issparse

# models 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


# # WiDS Datathon 2025: ADHD and Sex Prediction
# 
# **Goal:** Build a multi-outcome model to predict:
# 1. ADHD diagnosis (ADHD_Outcome: 0=Other/None, 1=ADHD)
# 2. Sex (Sex_F: 0=Male, 1=Female)

# ## Data Loading and Preprocessing

# In[2]:


base_path = "widsdata"
train_cat_path      = "TRAIN_CATEGORICAL_METADATA_new.csv"
train_fcm_path      = "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv"
train_quant_path    = "TRAIN_QUANTITATIVE_METADATA_new.csv"
train_solution_path = "TRAINING_SOLUTIONS.csv"

# Load the training solutions file 
test_cat_path   = "TEST_CATEGORICAL.xlsx"
test_fcm_path   = "TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
test_quant_path = "TEST_QUANTITATIVE_METADATA.xlsx"


# In[3]:


def read_data(path_str):
    """
    Read a CSV or Excel file using pandas, handling Windows/Mac paths.
    """
    path = Path(path_str)
    if path.suffix.lower() in ['.csv']:
        return pd.read_csv(path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


# In[4]:


# Load training data
train_cat = read_data(train_cat_path)
train_fcm = read_data(train_fcm_path)
train_quant = read_data(train_quant_path)
train_solution = read_data(train_solution_path)

# Load test data
test_cat = read_data(test_cat_path)
test_fcm = read_data(test_fcm_path)
test_quant = read_data(test_quant_path)


# In[5]:


print("Training shapes:", train_cat.shape, train_fcm.shape, train_quant.shape)
print("Test shapes:", test_cat.shape, test_fcm.shape, test_quant.shape)


# In[6]:


print("Train_Categorical")
train_cat.head()


# In[7]:


print("Train_FCM")
train_fcm.head()


# In[8]:


print("Train_Quant")
train_quant.head()


# In[9]:


print("Train_solutions")
train_solution.head()


# In[10]:


print("Test_Categorical")
test_cat.head()


# In[11]:


print("Test_FCM")
test_fcm.head()


# In[12]:


print("Test_Quant")
test_quant.head()


# In[13]:


# Merge by participant_id
train_merged = (
    train_cat
    .merge(train_fcm, on="participant_id", how="inner")
    .merge(train_quant, on="participant_id", how="inner")
)

test_merged = (
    test_cat
    .merge(test_fcm, on="participant_id", how="inner")
    .merge(test_quant, on="participant_id", how="inner")
)

print("Merged train shape:", train_merged.shape)
print("Merged test shape:", test_merged.shape)


# # Missing Value Handling

# In[14]:


# Calculate missing fraction in training
missing_frac = train_merged.isnull().mean()
cols_high_missing = missing_frac[missing_frac > 0.10].index.tolist()
print(f"Dropping {len(cols_high_missing)} cols >10% missing:\n", cols_high_missing)

# Drop these columns in both train and test
train_clean = train_merged.drop(columns=cols_high_missing)
test_clean = test_merged.drop(columns=cols_high_missing, errors='ignore')


# In[15]:


# Numeric impute
num_cols = train_clean.select_dtypes(include=[np.number]).columns
imp_num = SimpleImputer(strategy='median')
train_clean[num_cols] = imp_num.fit_transform(train_clean[num_cols])
test_clean[num_cols]  = imp_num.transform(test_clean[num_cols])

# Drop categorical NA
cat_cols = train_clean.select_dtypes(include=['object', 'category']).columns
train_clean = train_clean.dropna(subset=cat_cols)
test_clean  = test_clean.dropna(subset=cat_cols)

print("Post‑clean train shape:", train_clean.shape)
print("Post‑clean test shape:", test_clean.shape)


# # Merge Targets & Split Features/Labels

# In[16]:


train_data = train_clean.merge(
    train_solution[['participant_id','ADHD_Outcome','Sex_F']],
    on='participant_id', how='inner'
)

X_train = train_data.drop(['participant_id','ADHD_Outcome','Sex_F'], axis=1)
y_train = train_data[['ADHD_Outcome','Sex_F']]


# # Mutual Information Analysis (Top 20)

# In[17]:


def analyze_mi(df, target, top_n=20):
    X = df.drop(['participant_id', 'ADHD_Outcome', 'Sex_F'], axis=1).copy()
    y = df[target]
    # Factorize categoricals
    for col in X.select_dtypes(include=['object', 'category']):
        X[col] = pd.factorize(X[col])[0]
    X = X.fillna(X.median())
    mi_scores = mutual_info_classif(X, y)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10,8))
    sns.barplot(x=mi_series.head(top_n).values, y=mi_series.head(top_n).index, palette="viridis")
    plt.title(f"Top {top_n} MI features for {target}")
    plt.xlabel("Mutual Information Score")
    plt.tight_layout()
    plt.show()
    return mi_series

mi_adhd = analyze_mi(train_data, 'ADHD_Outcome', top_n=20)
mi_sex  = analyze_mi(train_data, 'Sex_F',       top_n=20)


# # Modeling Building 
# *** we are not using PCA here anymore since the cPVE graphs shows 50 PC only reached 40% variance. using MI to find the most important variables here instead (just correlation will miss some non-linear correlation)  
# 
# ## Split the data through MI only 
# 
# 

# In[18]:


def mi_rank(X, y, k=50, random_state=0):
    """
    Return list of top-k column names ranked by mutual information.
    Categoricals are ordinal-encoded *only* for the MI calculation.
    """
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    num_cols = X.select_dtypes(include=np.number).columns

    temp_pre = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ])

    X_enc = temp_pre.fit_transform(X)
    discrete_mask = [False] * len(num_cols) + [True] * len(cat_cols)

    mi = mutual_info_classif(
        X_enc, y,
        discrete_features=discrete_mask,
        random_state=random_state
    )
    mi_series = pd.Series(mi, index=temp_pre.get_feature_names_out())
    # tidy column names produced by ColumnTransformer
    mi_series.index = [c.split('__')[-1] for c in mi_series.index]
    return mi_series.sort_values(ascending=False).head(k).index.tolist()

# 1) seperate predictors and targets 
drop_cols = ['participant_id', 'ADHD_Outcome', 'Sex_F']   
X_full = train_data.drop(columns=drop_cols)
y_adhd = train_data['ADHD_Outcome']
y_sex  = train_data['Sex_F']

# one stratification criterion is fine – ADHD is usually the rarer label
X_train, X_test, y_adhd_tr, y_adhd_te, y_sex_tr, y_sex_te = \
    train_test_split(
        X_full, y_adhd, y_sex,
        test_size=0.30,
        stratify=y_adhd,
        random_state=42
    )

# 2) top 50 MI screening 
## when using 50 MI for each predictors (ADHD and gender), there will be approximately 100 predictors 
## at the end
k = 50          # top-k per target
top_adhd = mi_rank(X_train, y_adhd_tr, k=k)
top_sex  = mi_rank(X_train, y_sex_tr,  k=k)
sel_cols = sorted(set(top_adhd) | set(top_sex)) # take union --> delete the overlapped predictor

print(f'Keeping {len(sel_cols)} predictors after MI screening')

# 3) final preprocessing pipeline 
num_cols = [c for c in sel_cols if X_train[c].dtype.kind in 'bifc']
cat_cols = [c for c in sel_cols if c not in num_cols]

preprocess = ColumnTransformer([
    ('num', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler())
    ]), num_cols),

    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ]), cat_cols)
])

# 4) fit on training, transform both splits 
## X_train_proc and X_test_proc are the final dataset used for modeling and testing !!!
X_train_proc = preprocess.fit_transform(X_train[sel_cols])
X_test_proc  = preprocess.transform(X_test[sel_cols])

print('Processed shapes  |  train:', X_train_proc.shape, ' test:', X_test_proc.shape)


# ## Model Fitting 
# Construct models in the first hand 

# In[19]:


# 1) helper function: run and measure one model 
def fit_and_score(clf, X_tr, y_tr, X_te, y_te, dense=False):
    """Fit *clf*; return accuracy & elapsed seconds."""
    start = time.perf_counter()

    if dense and hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
        X_te = X_te.toarray()

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    secs = time.perf_counter() - start
    return acc, secs

# 2) model zoo (prebuild model for reuse) 
models = {
    "LDA"          : (LinearDiscriminantAnalysis(), True),      # needs dense
    "QDA"          : (QuadraticDiscriminantAnalysis(reg_param=0.01), True),
    "Logistic"     : (LogisticRegression(max_iter=500, n_jobs=-1), False),
    "KNN(k=11)"    : (KNeighborsClassifier(n_neighbors=11), True),
    "RandomForest" : (RandomForestClassifier(
                        n_estimators=300, max_depth=None,
                        n_jobs=-1, random_state=0), False),
    "BoostedTree"  : (GradientBoostingClassifier(random_state=0), False),
    "SVM-RBF"      : (SVC(kernel='rbf', C=1.0, gamma='scale'), True)
}

# 3) loop over targets and models 
results = []

targets = {
    "ADHD" : (y_adhd_tr, y_adhd_te),
    "Sex_F": (y_sex_tr , y_sex_te )
}

for tgt_name, (y_tr, y_te) in targets.items():
    for mdl_name, (clf, needs_dense) in models.items():
        acc, secs = fit_and_score(clf, X_train_proc, y_tr,
                                       X_test_proc , y_te,
                                       dense=needs_dense)
        results.append((tgt_name, mdl_name, round(acc, 3), round(secs, 2)))
        print(f"[{tgt_name:5s}] {mdl_name:12s}  acc={acc:.3f}   time={secs:.2f}s")

# 4) give out summary for visualization 
summary = pd.DataFrame(results,
                       columns=["Target","Model","Accuracy","Seconds"]
                       ).pivot(index="Model", columns="Target", values="Accuracy")

print("\nAccuracy table:")
display(summary)


# ## Model Tuning 
# Further tuning the model using cross validation to reach a higher accuracy. 
# From the model fitting, we choose top 3 accuracy in each perdictor: 
#     
#     ADHD - SVM, RandomForest, BoostedTree
#     Sex_F - Logistic, LDA, SVM

# In[20]:


# silence ALL User / Convergence / FitFailed warnings emitted inside GridSearch
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

def densify_if_needed(X, dense):
    return X.toarray() if dense and hasattr(X, "toarray") else X

# ==================================
# 1) model template for tuning 
## LDA grid is now two dicts: with & without shrinkage
## logistic grid only uses valid solver/penalty combos
model_templates = {
    "LDA" : (
        LinearDiscriminantAnalysis(),
        True,
        [
          {"solver": ["svd"]},                                        # → no shrinkage
          {"solver": ["lsqr"], "shrinkage": ["auto", 0.1, 0.3]}
        ]
    ), 
    "Logistic" : (
        LogisticRegression(max_iter=1000, solver="liblinear"),        # liblinear supports l1/l2
        False,
        {
          "penalty": ["l1", "l2"],
          "C"      : np.logspace(-3, 2, 6)    # 0.001 … 100
        }
    ),
    "RandomForest" : (
        RandomForestClassifier(random_state=0, n_jobs=-1),
        False,
        {
          "n_estimators"     : [300, 600, 900],
          "max_depth"        : [None, 10, 20],
          "min_samples_leaf" : [1, 3, 5]
        }
    ),
    "BoostedTree" : (
        GradientBoostingClassifier(random_state=0),
        False,
        {
          "n_estimators" : [300, 600],
          "learning_rate": [0.03, 0.05, 0.1],
          "max_depth"    : [2, 3]
        }
    ),
    "SVM-RBF" : (
        SVC(kernel="rbf"),
        True,
        {
          "C"     : [0.1, 1, 10, 100, 300],
          "gamma" : ["scale", 0.03, 0.01, 0.003]
        }
    )
}

# ==========================================
# 2) pick top 3 accuracy model for each target 
top3 = {
    tgt: summary[tgt].nlargest(3).index.tolist()
    for tgt in ["ADHD", "Sex_F"]
}
print("Model candidates:", top3)

# ==========================================
# 3) grid search the chosen models 
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
targets = {
    "ADHD" : (y_adhd_tr, y_adhd_te),
    "Sex_F": (y_sex_tr , y_sex_te )
}

rows, best_final = [], {}

for tgt, (y_tr, y_te) in targets.items():
    print(f"\n===== tuning for {tgt} =====")
    best_cv, best_name = -np.inf, None

    for mdl in top3[tgt]:
        est, need_dense, grid = model_templates[mdl]
        gs = GridSearchCV(
            estimator = est,
            param_grid = grid,
            cv = cv10,
            scoring = "accuracy",
            n_jobs = -1,
            error_score = "raise"      # stop if truly broken
        )
        Xt = densify_if_needed(X_train_proc, need_dense)
        Xe = densify_if_needed(X_test_proc , need_dense)

        tic = time.perf_counter()
        gs.fit(Xt, y_tr)
        toc = time.perf_counter()

        cv_acc   = gs.best_score_
        test_acc = accuracy_score(y_te, gs.best_estimator_.predict(Xe))
        rows.append([tgt, mdl, round(cv_acc,3), round(test_acc,3)])

        print(f"{mdl:<12s} CV={cv_acc:.3f}  Test={test_acc:.3f}  "
              f"time={toc-tic:.1f}s  {gs.best_params_}")

        if cv_acc > best_cv:
            best_cv, best_name          = cv_acc, mdl
            best_final[tgt]             = gs.best_estimator_
            joblib.dump(gs.best_estimator_, f"best_{tgt}_{mdl}.joblib")

    print(f"→ selected **{best_name}** for {tgt} (CV={best_cv:.3f})")

# ==========================================
# 4） present the tuned accuracy 

result_df = (
    pd.DataFrame(rows, columns=["Target","Model","CV_acc","Test_acc"])
      .set_index(["Target","Model"])
      .sort_index()
)

print("\nTuned accuracy (CV and Test):")
display(result_df)


# In[30]:


import warnings, re
def filter_boring(record):
    txt = str(record.message)
    if re.search(r"All the features will be returned", txt): return True
    if re.search(r"l1_ratio parameter is only used", txt): return True
    return False

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn",
                        append=False)      # blanket off
warnings.showwarning = lambda *args, **kw: None  # turn default back on
warnings.filterwarnings("default")              # then restore defaults
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*All the features will be returned.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*l1_ratio parameter is only used.*")

# 2) helper: densify only when *requested* & really sparse
# ---------------------------------------------------------------
def densify_if_needed(X, need_dense):
    return X.toarray() if need_dense and issparse(X) else X

# 3) pipeline factory
# ---------------------------------------------------------------
def make_pipe(clf, needs_scale, needs_dense):
    steps = []
    if needs_scale:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("kbest",  SelectKBest(mutual_info_classif)))
    steps.append(("clf",    clf))
    pipe = Pipeline(steps)
    pipe.needs_dense = needs_dense      # remember for later
    return pipe

# 4) search space definitions
# ---------------------------------------------------------------
n_feat  = X_train_proc.shape[1]            # e.g. 98 columns after your pre-processing
k_grid  = [min(n_feat, int(.3*n_feat)),
           min(n_feat, int(.6*n_feat)),
           n_feat]                         # roughly 30 %, 60 %, 100 % of cols

search_spaces = {

    # Elastic-Net logistic
    "LogitEN": (
        make_pipe(LogisticRegression(max_iter=4000, solver="saga",
                                     penalty="elasticnet",
                                     class_weight="balanced"),
                  needs_scale=True, needs_dense=False),
        { "kbest__k"   : k_grid,
          "clf__C"     : loguniform(1e-3, 1e2),
          "clf__l1_ratio": uniform(0,1) }
    ),

    # SVM with RBF kernel
    "SVM-RBF": (
        make_pipe(SVC(kernel="rbf", class_weight="balanced", probability=False),
                  needs_scale=True, needs_dense=True),
        { "kbest__k" : k_grid,
          "clf__C"   : loguniform(1e-1, 1e3),
          "clf__gamma": loguniform(1e-4, 1e0) }
    ),

    # Balanced Random Forest
    "BalRF": (
        make_pipe(BalancedRandomForestClassifier(random_state=0, n_jobs=-1),
                  needs_scale=False, needs_dense=False),
        { "kbest__k"          : k_grid,
          "clf__n_estimators" : [400, 600, 800],
          "clf__max_depth"    : [None, 10, 20],
          "clf__min_samples_leaf":[1, 3, 5] }
    ),

    # Gradient Boosting
    "GB": (
        make_pipe(GradientBoostingClassifier(random_state=0),
                  needs_scale=False, needs_dense=False),
        { "kbest__k"        : k_grid,
          "clf__n_estimators":[300, 600],
          "clf__learning_rate":[0.03, 0.05, 0.1],
          "clf__max_depth"  : [2, 3] }
    ),

    # LDA with/without shrinkage
    "LDA": (
        make_pipe(LinearDiscriminantAnalysis(),
                  needs_scale=False, needs_dense=True),
        [ {"kbest__k": k_grid, "clf__solver":["svd"]},
          {"kbest__k": k_grid, "clf__solver":["lsqr"],
           "clf__shrinkage":["auto", 0.1, 0.3]} ]
    ),

    # k-NN
    "KNN": (
        make_pipe(KNeighborsClassifier(),
                  needs_scale=True, needs_dense=True),
        { "kbest__k": k_grid,
          "clf__n_neighbors":[5, 11, 17, 25] }
    ),
}

# 5) main tuning loop
# ---------------------------------------------------------------
cv10      = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
targets   = { "ADHD": (y_adhd_tr, y_adhd_te),
              "Sex_F": (y_sex_tr , y_sex_te ) }

rows, best_final = [], {}

for tgt, (y_tr, y_te) in targets.items():
    print(f"\n================ tuning {tgt} ================\n")
    best_cv = -np.inf

    for name, (pipe, space) in search_spaces.items():

        Xtr = densify_if_needed(X_train_proc, pipe.needs_dense)
        Xte = densify_if_needed(X_test_proc , pipe.needs_dense)

        rs  = RandomizedSearchCV(pipe, space,
                                 n_iter      = 120,   # feel free to lower to 60 for speed
                                 scoring     = "accuracy",
                                 cv          = cv10,
                                 random_state= 42,
                                 n_jobs      = -1,
                                 error_score = "raise")

        tic = time.perf_counter()
        rs.fit(Xtr, y_tr)
        toc = time.perf_counter()

        cv_acc   = rs.best_score_
        test_acc = accuracy_score(y_te, rs.best_estimator_.predict(Xte))
        rows.append([tgt, name, round(cv_acc,3), round(test_acc,3)])

        print(f"{name:8s}  CV={cv_acc:.3f}  Test={test_acc:.3f} "
              f"  time={toc-tic:.1f}s")

        if cv_acc > best_cv:
            best_cv         = cv_acc
            best_final[tgt] = rs.best_estimator_
            joblib.dump(rs.best_estimator_, f"best_{tgt}_{name}.joblib")

    print(f"\n→ chosen model for {tgt}: "
          f"{best_final[tgt].steps[-1][0]}  (CV={best_cv:.3f})\n")

# 6) tidy summary table
# ---------------------------------------------------------------
result_df = (
    pd.DataFrame(rows, columns=["Target","Model","CV_acc","Test_acc"])
      .set_index(["Target","Model"])
      .sort_index()
)

print("\n=== Tuned accuracy (CV & held-out) ===")
display(result_df)


# In[ ]:


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

def dense(X):                        # helper for dense-only models
    return X.toarray() if hasattr(X, "toarray") else X

# ==============================================================
# 1)   Hyper-search configuration
# ==============================================================

inner_cv  = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
outer_cv5 = StratifiedKFold(n_splits=5 , shuffle=True, random_state=2)   # for nested score

def build_search(model_name, target):
    """Return (search_object, needs_dense_flag) per model/target."""
    pos_weight = "balanced" if target=="ADHD" else None   # tweak only ADHD

    if model_name == "SVM":
        search = RandomizedSearchCV(
            SVC(kernel="rbf", class_weight=pos_weight),
            param_distributions={
                "C": np.logspace(-1, 3, 50),              # 0.1 … 1000
                "gamma": np.logspace(-4, -0.5, 50),       # 1e-4 … 0.3
            },
            n_iter=200,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1,
            random_state=3
        )
        return search, True

    if model_name == "RF":
        search = RandomizedSearchCV(
            RandomForestClassifier(class_weight=pos_weight, n_jobs=-1, random_state=0),
            param_distributions={
                "n_estimators": np.arange(300,1201,150),
                "max_depth"   : [None,*np.arange(5,26,5)],
                "min_samples_leaf": [1,2,3,5],
                "max_features": ["sqrt","log2",0.3,0.5,0.8]
            },
            n_iter=120,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1,
            random_state=4
        )
        return search, False

    if model_name == "GB":
        search = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=0),
            param_distributions={
                "n_estimators": np.arange(300,1201,150),
                "learning_rate": np.logspace(-2, -0.3, 20),   # 0.01-0.5
                "max_depth": [2,3,4],
                "subsample": [0.7,0.8,0.9,1.0]
            },
            n_iter=120,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1,
            random_state=5
        )
        return search, False

    if model_name == "LogReg":
        search = GridSearchCV(
            LogisticRegression(max_iter=2000, solver="liblinear",
                               class_weight=pos_weight),
            param_grid={
                "penalty": ["l1","l2"],
                "C"      : np.logspace(-3,2,10)
            },
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1
        )
        return search, False

    if model_name == "LDA":
        search = GridSearchCV(
            LinearDiscriminantAnalysis(),
            param_grid=[
                {"solver":["svd"]},
                {"solver":["lsqr"], "shrinkage":["auto",0.1,0.3]}
            ],
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1
        )
        return search, True

    raise ValueError("unknown model")

# map of top-3 baseline winners per target (from your summary table)
top_models = {
  "ADHD":  ["SVM", "RF", "GB"],
  "Sex_F": ["LDA", "LogReg", "SVM"]
}

targets = {
    "ADHD" : (y_adhd_tr, y_adhd_te),
    "Sex_F": (y_sex_tr , y_sex_te )
}

result_rows = []
best_pipe   = {}

for tgt, (y_tr, y_te) in targets.items():
    print(f"\n===== {tgt} =====")
    Xt, Xe = X_train_proc, X_test_proc   # both already sparse; we densify per model

    best_cv, best_name = -np.inf, None

    for mdl in top_models[tgt]:
        search, needs_dense = build_search(mdl, tgt)

        X_inner = dense(Xt) if needs_dense else Xt
        X_hold  = dense(Xe) if needs_dense else Xe

        tic = time.perf_counter()
        search.fit(X_inner, y_tr)
        toc = time.perf_counter()

        # nested CV score (outer 5-fold) for more honest estimate
        nested = cross_val_score(search.best_estimator_, X_inner, y_tr,
                                 cv=outer_cv5, scoring="accuracy").mean()

        test_acc = accuracy_score(y_te, search.best_estimator_.predict(X_hold))

        result_rows.append([tgt, mdl, round(search.best_score_,3),
                            round(nested,3), round(test_acc,3)])

        print(f"{mdl:<6s}  inner-CV={search.best_score_:0.3f}  "
              f"nested={nested:0.3f}  test={test_acc:0.3f}  "
              f"{search.best_params_}  ({toc-tic:0.1f}s)")

        if nested > best_cv:          # pick by nested CV
            best_cv, best_name       = nested, mdl
            best_pipe[tgt]           = search.best_estimator_
            joblib.dump(search.best_estimator_, f"BEST_{tgt}_{mdl}.joblib")

    print(f"→ final choice for {tgt}: **{best_name}**  (nested={best_cv:0.3f})")

# tidy table
display(
    pd.DataFrame(result_rows,
                 columns=["Target","Model","inner_CV","outer_CV","Test_acc"]
            ).set_index(["Target","Model"]).sort_index()
)

