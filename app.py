import os, json
import numpy as np, pandas as pd, yaml, joblib
import streamlit as st
import plotly.graph_objects as go

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="金融风险沙盒", layout="wide")
st.title("金融风险沙盒（真实预测 + Top-3 解释）")

CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))["features"]
MODEL_PATH, PREP_PATH = "models/model.pkl", "models/preprocessor.pkl"
FN_PATH, BG_PATH, META_PATH = "models/feature_names.json", "models/background_X.npy", "models/metadata.json"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    pre   = joblib.load(PREP_PATH)  if os.path.exists(PREP_PATH)  else None
    feat_names = json.load(open(FN_PATH)) if os.path.exists(FN_PATH) else None
    bg_X = np.load(BG_PATH) if os.path.exists(BG_PATH) else None
    best = json.load(open(META_PATH))["best"] if os.path.exists(META_PATH) else "lr"
    return model, pre, feat_names, bg_X, best

model, pre, feat_names, bg_X, best_name = load_artifacts()

with st.sidebar:
    st.header("参数设置")
    def pick_num(k): f=CFG[k]; return st.slider(k, f["min"],f["max"],f["default"],f["step"])
    def pick_choice(k): f=CFG[k]; return st.selectbox(k, f["choices"], index=f["choices"].index(f["default"]))
    loan_amount   = pick_num("loan_amount")
    interest_rate = pick_num("interest_rate")
    term_months   = st.selectbox("term_months", CFG["term_months"]["choices"], index=CFG["term_months"]["choices"].index(CFG["term_months"]["default"]))
    dti_ratio     = pick_num("dti_ratio")
    annual_income = pick_num("annual_income")
    credit_score  = pick_num("credit_score")
    employment_status = pick_choice("employment_status")
    home_ownership    = pick_choice("home_ownership")

row = pd.DataFrame([{
    "loan_amount": loan_amount, "interest_rate": interest_rate, "term_months": term_months,
    "dti_ratio": dti_ratio, "annual_income": annual_income, "credit_score": credit_score,
    "employment_status": employment_status, "home_ownership": home_ownership
}])

def fallback(df: pd.DataFrame) -> float:
    z = (-0.00004*df["loan_amount"] + 0.09*df["interest_rate"] + 0.03*(df["term_months"]/12)
         + 3.2*df["dti_ratio"] - 0.000006*df["annual_income"] - 0.007*(df["credit_score"]-680)
         + (df["employment_status"]!="employed")*0.5 + (df["home_ownership"]=="rent")*0.4)
    return float(1/(1+np.exp(-z.values))[0])

def predict(df: pd.DataFrame) -> float:
    if model is not None:
        try: return float(model.predict_proba(df)[0,1])
        except Exception: pass
    return fallback(df)

p = predict(row)

def top3(df: pd.DataFrame):
    if model is None or pre is None:
        # 占位解释（与公式一致）
        drivers = [("DTI", 3.2*df["dti_ratio"].iloc[0]),
                   ("利率", 0.09*df["interest_rate"].iloc[0]),
                   ("额度", -0.00004*df["loan_amount"].iloc[0]),
                   ("收入", -0.000006*df["annual_income"].iloc[0]),
                   ("信用分", -0.007*(df["credit_score"].iloc[0]-680))]
        if df["employment_status"].iloc[0]!="employed": drivers.append(("就业形态(非正式)",0.5))
        if df["home_ownership"].iloc[0]=="rent": drivers.append(("住房(租)",0.4))
        drivers.sort(key=lambda x: abs(x[1]), reverse=True)
        return drivers[:3], drivers

    Xt = pre.transform(df); Xt = Xt.toarray() if hasattr(Xt,"toarray") else Xt
    # RF → SHAP
    if best_name=="rf" and SHAP_AVAILABLE:
        try:
            rf = model.base_estimator.named_steps["clf"]
            expl = shap.TreeExplainer(rf, data=bg_X if isinstance(bg_X,np.ndarray) else None, feature_perturbation="tree_path_dependent")
            shap_values = expl.shap_values(Xt)
            sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            names = feat_names if isinstance(feat_names,list) else [f"f{i}" for i in range(len(sv))]
            pairs = sorted(list(zip(names, sv)), key=lambda x: abs(x[1]), reverse=True)
            return pairs[:3], pairs
        except Exception:
            pass
    # LR 或兜底：线性贡献
    try:
        lr = model.base_estimator.named_steps["clf"]
        contrib = Xt[0] * lr.coef_[0]
        names = feat_names if isinstance(feat_names,list) else [f"f{i}" for i in range(len(contrib))]
        pairs = sorted(list(zip(names, contrib)), key=lambda x: abs(x[1]), reverse=True)
        return pairs[:3], pairs
    except Exception:
        return [("无法计算解释", 0.0)], None

top3_pairs, all_pairs = top3(row)

c1, c2 = st.columns([1,2])
with c1:
    st.metric("预测风险", f"{p*100:.1f}%")
with c2:
    txt = "、".join([f"{k}（{'+' if v>=0 else ''}{v:.2f}）" for k,v in top3_pairs])
    st.write(f"**解释**：主要驱动因素：{txt}。")
    if all_pairs is not None:
        show = sorted(all_pairs, key=lambda x: abs(x[1]), reverse=True)[:8]
        fig2 = go.Figure(go.Bar(x=[v for _,v in show], y=[k for k,_ in show], orientation="h"))
        fig2.update_layout(title="局部贡献（前8）", xaxis_title="贡献（正↑负↓）", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("风险热力图（额度 × 利率）")
la = np.linspace(CFG["loan_amount"]["min"], CFG["loan_amount"]["max"], 40)
ir = np.linspace(CFG["interest_rate"]["min"], CFG["interest_rate"]["max"], 40)
Z = np.zeros((len(ir), len(la)))
for i, r in enumerate(ir):
    for j, a in enumerate(la):
        df2 = row.copy()
        df2.loc[:, "loan_amount"] = a
        df2.loc[:, "interest_rate"] = r
        try: Z[i,j] = predict(df2)
        except Exception: Z[i,j] = fallback(df2)
fig = go.Figure(data=go.Heatmap(z=Z, x=la, y=ir, colorbar=dict(title="风险")))
fig.update_layout(title="风险随（额度×利率）变化", xaxis_title="loan_amount", yaxis_title="interest_rate")
st.plotly_chart(fig, use_container_width=True)

if model is None:
    st.info("当前使用占位公式。执行 `python train.py` 训练后将自动切换为真实模型。")
else:
    st.success(f"已使用真实模型（best={best_name.upper()}）。")
