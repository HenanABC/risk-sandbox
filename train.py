import os, json, yaml, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

os.makedirs("models", exist_ok=True)
cfg = yaml.safe_load(open("config.yaml"))["features"]
rng = np.random.default_rng(42)

# 生成合成数据（与占位公式一致）
N = 1000
def sample(spec):
    if "choices" in spec: return rng.choice(spec["choices"], size=N)
    vals = rng.uniform(spec["min"], spec["max"], size=N)
    if "step" in spec and spec["step"]>=1: vals = (vals/spec["step"]).round()*spec["step"]
    return vals

df = pd.DataFrame({k: sample(v) for k,v in cfg.items()})
df["term_months"] = df["term_months"].astype(int)
for k in ["employment_status","home_ownership"]:
    df[k] = df[k].astype(str)

z = (-0.00004*df["loan_amount"] + 0.09*df["interest_rate"] + 0.03*(df["term_months"]/12)
     + 3.2*df["dti_ratio"] - 0.000006*df["annual_income"] - 0.007*(df["credit_score"]-680)
     + (df["employment_status"]!="employed")*0.5 + (df["home_ownership"]=="rent")*0.4)
p = 1/(1+np.exp(-z))
df["label"] = rng.binomial(1, p)

y = df["label"].values
X = df.drop(columns=["label"])

num = ["loan_amount","interest_rate","term_months","dti_ratio","annual_income","credit_score"]
cat = ["employment_status","home_ownership"]
pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# LR + 校准
lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300, class_weight="balanced", C=1.0))])
lr.fit(Xtr, ytr)
lr_cal = CalibratedClassifierCV(lr, cv=3, method="sigmoid").fit(Xtr, ytr)
proba_lr = lr_cal.predict_proba(Xte)[:,1]
print("LR+Cal  AUC:", round(roc_auc_score(yte, proba_lr),3), "Brier:", round(brier_score_loss(yte, proba_lr),3))

# RF + 校准
rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced"))])
rf.fit(Xtr, ytr)
rf_cal = CalibratedClassifierCV(rf, cv=3, method="sigmoid").fit(Xtr, ytr)
proba_rf = rf_cal.predict_proba(Xte)[:,1]
print("RF+Cal  AUC:", round(roc_auc_score(yte, proba_rf),3), "Brier:", round(brier_score_loss(yte, proba_rf),3))

best_name = "rf" if roc_auc_score(yte, proba_rf) >= roc_auc_score(yte, proba_lr) else "lr"
best = rf_cal if best_name=="rf" else lr_cal
print("Best model:", best_name)

joblib.dump(best, "models/model.pkl")
joblib.dump(pre,  "models/preprocessor.pkl")
feat_names = pre.get_feature_names_out()
pd.Series(feat_names).to_json("models/feature_names.json", orient="values")
Xt = pre.transform(Xtr); Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
bg_idx = rng.choice(Xt.shape[0], size=min(100, Xt.shape[0]), replace=False)
np.save("models/background_X.npy", Xt[bg_idx])
json.dump({"best": best_name}, open("models/metadata.json","w"), indent=2)
print("Saved to models/")
