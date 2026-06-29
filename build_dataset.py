import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
U = "/mnt/user-data/uploads/accepted_2007_to_2018Q4_csv.gz"

USECOLS = ["loan_status","annual_inc","dti","installment","loan_amnt","term","purpose",
           "home_ownership","emp_length","verification_status","application_type",
           "annual_inc_joint","dti_joint","fico_range_low","fico_range_high","delinq_2yrs",
           "inq_last_6mths","revol_util","open_acc","pub_rec","total_acc","earliest_cr_line","issue_d"]

BAD  = {"Charged Off","Default","Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid","Does not meet the credit policy. Status:Fully Paid"}
KEEP = BAD | GOOD
EMP  = {"< 1 year":0,"1 year":1,"2 years":2,"3 years":3,"4 years":4,"5 years":5,
        "6 years":6,"7 years":7,"8 years":8,"9 years":9,"10+ years":10}

def clean(ch):
    ch = ch[ch["loan_status"].isin(KEEP)].copy()
    if len(ch) == 0: return None
    ch["default"] = ch["loan_status"].isin(BAD).astype("int8")
    ch["term_months"]    = ch["term"].str.extract(r"(\d+)").astype("float32")
    ch["emp_length_num"] = ch["emp_length"].map(EMP).astype("float32")
    joint = (ch["application_type"] == "Joint App")
    ch["annual_inc_eff"] = np.where(joint & ch["annual_inc_joint"].notna(), ch["annual_inc_joint"], ch["annual_inc"]).astype("float32")
    ch["dti_eff"]        = np.where(joint & ch["dti_joint"].notna(), ch["dti_joint"], ch["dti"]).astype("float32")
    ch["fico"]           = ((ch["fico_range_low"] + ch["fico_range_high"]) / 2).astype("float32")
    ch["payment_to_income"] = (ch["installment"] * 12 / ch["annual_inc_eff"]).astype("float32")
    ch["loan_to_income"]    = (ch["loan_amnt"] / ch["annual_inc_eff"]).astype("float32")
    ed  = pd.to_datetime(ch["earliest_cr_line"], format="%b-%Y", errors="coerce")
    isd = pd.to_datetime(ch["issue_d"], format="%b-%Y", errors="coerce")
    ch["credit_history_yrs"] = ((isd - ed).dt.days / 365.25).astype("float32")
    ch["issue_year"] = isd.dt.year.astype("float32")
    out = ["default","annual_inc_eff","dti_eff","payment_to_income","loan_to_income","installment",
           "loan_amnt","term_months","emp_length_num","home_ownership","purpose","verification_status",
           "fico","delinq_2yrs","inq_last_6mths","revol_util","open_acc","pub_rec","total_acc",
           "credit_history_yrs","issue_year"]
    return ch[out]

parts, n = [], 0
for ch in pd.read_csv(U, usecols=USECOLS, compression="gzip", chunksize=300000, low_memory=False):
    c = clean(ch)
    if c is not None: parts.append(c); n += len(c)
    print(f"  processed chunk -> kept {n:,} completed loans", end="\r")
df = pd.concat(parts, ignore_index=True); del parts
print(f"\nCompleted loans: {len(df):,}")

# sanity filters: drop erroneous incomes, clip absurd ratios
before = len(df)
df = df[(df["annual_inc_eff"] >= 2000) & df["annual_inc_eff"].notna()]
df["payment_to_income"] = df["payment_to_income"].clip(0, 2)
df["loan_to_income"]    = df["loan_to_income"].clip(0, 5)
df["dti_eff"]           = df["dti_eff"].clip(0, 100)
df["revol_util"]        = df["revol_util"].clip(0, 200)
print(f"Dropped {before-len(df):,} rows with invalid income; final {len(df):,} | default rate {df['default'].mean():.1%}")

try:
    df.to_parquet("data/lending_affordability.parquet", index=False); print("saved parquet")
except Exception as e:
    df.to_csv("data/lending_affordability.csv.gz", index=False, compression="gzip"); print("saved csv.gz (", e, ")")
print("memory (MB):", round(df.memory_usage(deep=True).sum()/1e6, 1))
