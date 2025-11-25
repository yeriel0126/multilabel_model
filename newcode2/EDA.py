import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]  # repo root: ../ from newcode2
OUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

CANDIDATE_PATTERNS = [
    "**/complete_data.csv",
    "**/complete_data.parquet",
    "**/complete_data*.csv",
    "**/complete_data*.parquet",
    "**/complete_data/*.csv",
    "**/complete_data/*.parquet",
    "**/complete_data.json",
    "**/complete_data/*.json",
]

def find_complete_data(root=ROOT):
    for pat in CANDIDATE_PATTERNS:
        matches = list(root.glob(pat))
        if matches:
            return matches[0]
    # fallback: any file or folder named 'complete_data'
    for p in root.rglob("*"):
        if p.name.lower().startswith("complete_data"):
            return p
    return None

def load_data(path: Path):
    if path.is_dir():
        # try typical files inside
        for ext in ("csv", "parquet", "json"):
            candidate = next(path.glob(f"**/*.{ext}"), None)
            if candidate:
                path = candidate
                break
    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path)
    if path.suffix.lower() in [".parquet"]:
        return pd.read_parquet(path)
    if path.suffix.lower() in [".json"]:
        return pd.read_json(path, lines=False)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def report_missing(df: pd.DataFrame):
    miss = df.isna().sum()
    pct = (miss / len(df)) * 100
    r = pd.DataFrame({"missing_count": miss, "missing_pct": pct, "dtype": df.dtypes})
    r.to_csv(OUT_DIR / "missing_report.csv")
    return r

def numeric_outliers(df: pd.DataFrame, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = {}
    for c in cols:
        col = df[c].dropna()
        if col.empty:
            outlier_summary[c] = {"iqr_outliers": 0, "zscore_outliers": 0, "n": 0}
            continue
        q1, q3 = np.percentile(col, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_mask = (col < lower) | (col > upper)
        z_mask = np.abs(stats.zscore(col)) > 3
        outlier_summary[c] = {
            "n": len(col),
            "iqr_outliers": int(iqr_mask.sum()),
            "iqr_outlier_pct": float(iqr_mask.sum() / len(col) * 100),
            "zscore_outliers": int(z_mask.sum()),
            "zscore_outlier_pct": float(z_mask.sum() / len(col) * 100),
            "min": float(col.min()),
            "q1": float(q1),
            "median": float(col.median()),
            "q3": float(q3),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
        }
        # plot histogram + boxplot
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        sns.histplot(col, bins=50, kde=True)
        plt.title(f"{c} histogram")
        plt.subplot(1,2,2)
        sns.boxplot(x=col)
        plt.title(f"{c} boxplot")
        plt.tight_layout()
        safe_name = c.replace("/", "_").replace(" ", "_")
        plt.savefig(PLOTS_DIR / f"{safe_name}_hist_box.png", dpi=150)
        plt.close()
    pd.DataFrame.from_dict(outlier_summary, orient="index").to_csv(OUT_DIR / "numeric_outlier_summary.csv")
    return pd.DataFrame.from_dict(outlier_summary, orient="index")

def categorical_report(df: pd.DataFrame):
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    rows = []
    for c in cats:
        vc = df[c].value_counts(dropna=False).head(10)
        rows.append({"column": c, "n_unique": df[c].nunique(dropna=True), "top_values": vc.to_dict()})
    rpt = pd.DataFrame(rows)
    rpt.to_json(OUT_DIR / "categorical_summary.json", orient="records", force_ascii=False)
    return rpt

def generate_basic_plots(df: pd.DataFrame, max_cols=6):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for c in numeric:
        try:
            plt.figure()
            sns.histplot(df[c].dropna(), bins=50, kde=True)
            plt.title(c)
            plt.savefig(PLOTS_DIR / f"{c}_hist.png", dpi=150)
            plt.close()
        except Exception:
            continue
    # correlation heatmap for numeric
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Numeric correlation (sample)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "numeric_correlation.png", dpi=150)
        plt.close()

def main():
    print("검색: repository에서 complete_data 파일/폴더 탐색...")
    p = find_complete_data()
    if p is None:
        print("complete_data를 찾을 수 없습니다. 루트에서 파일명/폴더명을 확인하세요.")
        return
    print("발견:", p)
    print("로드 중...")
    df = load_data(p)
    print("로딩 완료. 행,열:", df.shape)
    # 기본 정보 저장
    with open(OUT_DIR / "df_info.txt", "w") as f:
        df.info(buf=f)
    df.head(20).to_csv(OUT_DIR / "head_sample.csv", index=False)
    df.describe(include='all').to_csv(OUT_DIR / "describe_all.csv")

    print("결측치 리포트 생성...")
    miss = report_missing(df)
    print(miss.sort_values("missing_pct", ascending=False).head(10))

    print("카테고리 요약...")
    cat = categorical_report(df)

    print("수치형 이상치 검사 및 플롯 생성...")
    out = numeric_outliers(df)

    print("기본 플롯 생성...")
    generate_basic_plots(df)

    print("EDA 완료. 결과는 newcode2/ 하위 파일 및 plots/에 저장되었습니다.")

if __name__ == "__main__":
    main()