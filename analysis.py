import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
import shap, warnings, json, os; warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 120
RND = 42

# Built by build_dataset.py. Every accepted loan is kept; the outcome is default within a fixed
# 24-month window, used only for loans whose whole window was observable by the snapshot.
# (Earlier versions kept only finished loans, which over-represents defaults in recent vintages.)
WINDOW = 24
df_all = pd.read_parquet(os.environ.get("AFFORDABILITY_PARQUET", "data/lending_affordability.parquet"))
df = df_all[df_all[f"obs_{WINDOW}"]].copy()
df["default"] = df[f"bad_{WINDOW}"].astype(int)          # default within 24 months of issue
base = df["default"].mean()
print(f"Loans with an observable {WINDOW}-month outcome: {len(df):,} of {len(df_all):,} | "
      f"{WINDOW}-month default rate: {base:.1%} | issued {int(df['issue_year'].min())}-{int(df['issue_year'].max())}")

AFFORD = ["annual_inc_eff","dti_eff","payment_to_income","loan_to_income","installment","loan_amnt","term_months","emp_length_num"]
CREDIT = ["fico","delinq_2yrs","inq_last_6mths","revol_util","open_acc","pub_rec","total_acc","credit_history_yrs"]
BOTH = AFFORD + CREDIT

# ---------------- FIG 1: the affordability gradient ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
def gradient(col, bins, labels, axis, xlabel):
    g = df.groupby(pd.cut(df[col], bins=bins, labels=labels))["default"].mean()
    g.plot(kind="bar", ax=axis, color="#c44e52"); axis.axhline(base, ls="--", color="black", label=f"overall {base:.0%}")
    axis.set_xlabel(xlabel); axis.set_ylabel("default rate"); axis.legend(); axis.tick_params(axis="x", rotation=0)
gradient("payment_to_income", [0,.05,.10,.15,.20,2], ["<5%","5-10%","10-15%","15-20%",">20%"], ax[0], "annual loan payment / income")
gradient("dti_eff", [0,10,20,30,40,100], ["<10","10-20","20-30","30-40",">40"], ax[1], "debt-to-income (DTI)")
gradient("loan_to_income", [0,.1,.2,.3,.5,5], ["<10%","10-20%","20-30%","30-50%",">50%"], ax[2], "loan size / income")
fig.suptitle("The affordability gradient: the more of someone's income a loan consumes, the more they default", fontsize=12)
plt.tight_layout(); plt.savefig("figures/01_affordability_gradient.png"); plt.close()
print("\nDefault rate by payment-to-income:")
print(df.groupby(pd.cut(df["payment_to_income"], [0,.05,.10,.15,.20,2], labels=["<5%","5-10%","10-15%","15-20%",">20%"]))["default"].mean().round(3).to_string())

# ---------------- FIG 2: affordability adds signal BEYOND the credit score ----------------
fico_bins = [660,680,700,720,740,760,850]; fico_lab = ["660-680","680-700","700-720","720-740","740-760","760+"]
dti_bins = [0,10,20,30,40,100];           dti_lab = ["<10","10-20","20-30","30-40",">40"]
piv = df.assign(fb=pd.cut(df["fico"],fico_bins,labels=fico_lab), db=pd.cut(df["dti_eff"],dti_bins,labels=dti_lab)) \
        .groupby(["fb","db"])["default"].mean().unstack()
plt.figure(figsize=(9,6))
sns.heatmap(piv*100, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label":"default rate (%)"})
plt.xlabel("debt-to-income (affordability)"); plt.ylabel("FICO score (creditworthiness)")
plt.title("Even at the same credit score, worse affordability means more default")
plt.tight_layout(); plt.savefig("figures/02_fico_vs_affordability.png"); plt.close()
print("\nWithin FICO 700-720, default rate by DTI:"); print((piv.loc["700-720"]*100).round(0).to_string())

# ---------------- MODELS: credit-only vs affordability-only vs both ----------------
# Out-of-time split: train on loans issued up to 2014, test on later loans the model never saw.
dev, oot = df["issue_year"] <= 2014, df["issue_year"] >= 2015
Xtr, Xte, ytr, yte = df.loc[dev, BOTH], df.loc[oot, BOTH], df.loc[dev, "default"], df.loc[oot, "default"]
print(f"\nOut-of-time split: train {dev.sum():,} loans issued <=2014 ({ytr.mean():.1%} default) | "
      f"test {oot.sum():,} loans issued 2015-{int(df.loc[oot, 'issue_year'].max())} ({yte.mean():.1%} default)")
spw = (ytr==0).sum()/(ytr==1).sum()
def run(feats, name):
    m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                      scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=RND, n_jobs=4)
    m.fit(Xtr[feats], ytr); p = m.predict_proba(Xte[feats])[:,1]
    auc, ap = roc_auc_score(yte,p), average_precision_score(yte,p)
    print(f"  {name:24s} ROC-AUC {auc:.3f} | PR-AUC {ap:.3f}")
    return m, p, auc, ap
print(f"\nPredicting {WINDOW}-month default, out of time (no-skill PR-AUC = {yte.mean():.3f}):")
m_cr, p_cr, auc_cr, ap_cr = run(CREDIT, "Credit score + bureau")
m_af, p_af, auc_af, ap_af = run(AFFORD, "Affordability only")
m_bo, p_bo, auc_bo, ap_bo = run(BOTH,   "Both combined")

# ---------------- FIG 3: model comparison ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for name, p, c in [("Credit score + bureau", p_cr, "#4c72b0"), ("Affordability only", p_af, "#dd8452"), ("Both combined", p_bo, "#55a868")]:
    fpr, tpr, _ = roc_curve(yte, p); ax[0].plot(fpr, tpr, c=c, label=f"{name} (AUC {roc_auc_score(yte,p):.3f})")
ax[0].plot([0,1],[0,1],"k--",lw=1); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("Affordability alone rivals the credit score; together they win"); ax[0].legend()
comp = pd.DataFrame({"ROC-AUC":[auc_cr,auc_af,auc_bo],"PR-AUC":[ap_cr,ap_af,ap_bo]},
                    index=["Credit\n+bureau","Afford-\nability","Both"])
comp.plot(kind="bar", ax=ax[1], color=["#4c72b0","#55a868"]); ax[1].set_ylabel("score"); ax[1].set_ylim(0,max(auc_bo,0.75)*1.1)
ax[1].axhline(yte.mean(), ls=":", color="grey", label=f"no-skill PR-AUC ({yte.mean():.2f})")
ax[1].set_title("Affordability adds real predictive power"); ax[1].tick_params(axis="x", rotation=0); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/03_model_comparison.png"); plt.close()
print(f"\nAffordability-only AUC ({auc_af:.3f}) vs credit-only ({auc_cr:.3f}); combining lifts to {auc_bo:.3f}")

# ---------------- FIG 4: what drives default (SHAP, combined model) ----------------
samp = Xte.sample(3000, random_state=RND); sv = shap.TreeExplainer(m_bo).shap_values(samp)
plt.figure(); shap.summary_plot(sv, samp, show=False, max_display=12)
plt.title("What predicts default: affordability and credit signals together (SHAP)", fontsize=11)
plt.tight_layout(); plt.savefig("figures/04_shap_summary.png", bbox_inches="tight"); plt.close()
imp = pd.Series(np.abs(sv).mean(0), index=samp.columns).sort_values(ascending=False)
print("\nTop default drivers (mean |SHAP|):"); print(imp.head(8).round(3).to_string())

# ---------------- FIG 5: lower income, higher default ----------------
dec = df.groupby(pd.qcut(df["annual_inc_eff"], 10, labels=False))["default"].mean()
plt.figure(figsize=(8,5))
plt.plot(range(1,11), dec.values*100, "o-", color="#8172b3")
plt.axhline(base*100, ls="--", color="black", label=f"overall {base:.0%}")
plt.xlabel("income decile (1 = lowest, 10 = highest)"); plt.ylabel("default rate (%)")
plt.title("Default falls steadily as income rises"); plt.legend(); plt.xticks(range(1,11))
plt.tight_layout(); plt.savefig("figures/05_income_gradient.png"); plt.close()

# ---------------- FIG 6: score separation (combined model) ----------------
ks = ks_2samp(p_bo[yte==1], p_bo[yte==0]).statistic
plt.figure(figsize=(8,5))
sns.kdeplot(p_bo[yte==0], fill=True, color="#4c72b0", label="Repaid", clip=(0,1))
sns.kdeplot(p_bo[yte==1], fill=True, color="#c44e52", label="Defaulted", clip=(0,1))
plt.xlabel("predicted default risk"); plt.ylabel("density")
plt.title(f"The combined model separates outcomes (KS = {ks:.2f})"); plt.legend()
plt.tight_layout(); plt.savefig("figures/06_score_separation.png"); plt.close()
print(f"\nCombined-model KS statistic = {ks:.3f}")
# ---------------- random split, for comparison with earlier versions only ----------------
rXtr, rXte, rytr, ryte = train_test_split(df[BOTH], df["default"], test_size=0.25, stratify=df["default"], random_state=RND)
random_split = {}
for feats, name in [(CREDIT, "credit"), (AFFORD, "affordability"), (BOTH, "both")]:
    m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                      scale_pos_weight=(rytr == 0).sum() / (rytr == 1).sum(), eval_metric="aucpr",
                      tree_method="hist", random_state=RND, n_jobs=4).fit(rXtr[feats], rytr)
    random_split[name] = round(float(roc_auc_score(ryte, m.predict_proba(rXte[feats])[:, 1])), 3)
print(f"\nFor comparison, a random split gives ROC-AUC {random_split}")

g = df.groupby(pd.cut(df["payment_to_income"], [0, .05, .10, .15, .20, 2],
               labels=["<5%", "5-10%", "10-15%", "15-20%", ">20%"]), observed=False)["default"].mean()
results = dict(window_months=WINDOW, loans_all=int(len(df_all)), loans_observable=int(len(df)),
               default_rate=round(float(base), 4),
               issued=f"{int(df['issue_year'].min())}-{int(df['issue_year'].max())}",
               train=dict(n=int(dev.sum()), issued="<=2014", default_rate=round(float(ytr.mean()), 4)),
               test=dict(n=int(oot.sum()), issued=f"2015-{int(df.loc[oot, 'issue_year'].max())}",
                         default_rate=round(float(yte.mean()), 4)),
               out_of_time=dict(credit=dict(roc_auc=round(auc_cr, 3), pr_auc=round(ap_cr, 3)),
                                affordability=dict(roc_auc=round(auc_af, 3), pr_auc=round(ap_af, 3)),
                                both=dict(roc_auc=round(auc_bo, 3), pr_auc=round(ap_bo, 3)), ks_both=round(float(ks), 3)),
               random_split_roc_auc=random_split,
               default_rate_by_payment_to_income={k: round(float(v), 4) for k, v in g.items()})
json.dump(results, open("results.json", "w"), indent=2)
print("\nAll figures and results.json saved. DONE.")
