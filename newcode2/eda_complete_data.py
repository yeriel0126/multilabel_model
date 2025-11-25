import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETE_DATA_DIR = PROJECT_ROOT / "complete_data"
SUMMARY_PATH = COMPLETE_DATA_DIR / "complete_data_summary.json"


def load_complete_data() -> List[Dict]:
    """
    complete_data_summary.json 이 존재하면 그 안의 data 리스트를 사용하고,
    없으면 complete_data 폴더 내의 개별 JSON 파일을 모두 읽어서 리스트로 반환한다.
    """
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("data", [])
        if data:
            return data

    records: List[Dict] = []
    for entry in COMPLETE_DATA_DIR.glob("*.json"):
        if entry.name == "complete_data_summary.json":
            continue
        with entry.open("r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                continue
        records.append(obj)
    return records


def to_dataframe(records: List[Dict]) -> pd.DataFrame:
    """레코드를 DataFrame으로 변환하고 파생 변수 추가"""
    df = pd.DataFrame(records)
    
    # 리스트 길이 기반 파생 변수
    df["note_count"] = df["notes"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["blender_count"] = df["blenders"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["smiles_length"] = df["smiles"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df["has_cas"] = df["cas"].apply(lambda x: bool(x and str(x).strip()))
    
    # 노트 관련 파생 변수
    df["unique_note_count"] = df["notes"].apply(
        lambda x: len(set(x)) if isinstance(x, list) else 0
    )
    df["note_duplicates"] = df["note_count"] - df["unique_note_count"]
    
    # 블렌더 그룹 추출 및 분석
    def extract_blender_groups(blenders):
        if not isinstance(blenders, list):
            return []
        groups = []
        for item in blenders:
            if isinstance(item, list) and len(item) >= 2:
                group = str(item[1]).strip() if item[1] is not None else ""
                if group:
                    groups.append(group)
        return groups
    
    df["blender_groups"] = df["blenders"].apply(extract_blender_groups)
    df["unique_group_count"] = df["blender_groups"].apply(lambda x: len(set(x)) if isinstance(x, list) else 0)
    
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 및 빈 값 리포트"""
    columns = ["name", "cas", "smiles", "notes", "blenders"]
    report = []
    
    for col in columns:
        missing = df[col].isna().sum()
        
        # 빈 값 체크 (타입별로 다르게 처리)
        if col in ["notes", "blenders"]:
            # 리스트 타입: 빈 리스트 체크
            def is_empty_list(x):
                try:
                    if x is None:
                        return False
                    # 리스트인 경우만 체크
                    if isinstance(x, list):
                        return len(x) == 0
                    return False
                except (TypeError, ValueError):
                    return False
            empty = df[col].apply(is_empty_list).sum()
        else:
            # 문자열 타입: 빈 문자열 체크
            empty = (
                df[col].astype(str).str.strip().eq("").sum()
                if df[col].dtype == object
                else 0
            )
        
        report.append(
            {
                "column": col,
                "missing_count": int(missing),
                "empty_count": int(empty),
                "missing_ratio": f"{(missing / len(df) * 100):.2f}%",
                "empty_ratio": f"{(empty / len(df) * 100):.2f}%",
            }
        )
    
    return pd.DataFrame(report)


def detect_outliers(series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """IQR 기반 이상치 탐지"""
    clean_series = series.dropna()
    if clean_series.empty:
        return pd.Series([False] * len(series), index=series.index), {}
    
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    
    # IQR=0 인 경우 대비
    if iqr == 0:
        std = clean_series.std()
        iqr = std if std > 0 else 1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (series < lower_bound) | (series > upper_bound)
    
    stats = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_ratio": mask.mean(),
        "mean": clean_series.mean(),
        "median": clean_series.median(),
        "std": clean_series.std(),
        "min": clean_series.min(),
        "max": clean_series.max(),
    }
    return mask, stats


def print_outlier_summary(df: pd.DataFrame, column: str, head: int = 10) -> None:
    """이상치 요약 출력"""
    mask, stats = detect_outliers(df[column])
    print(f"\n[{column}] IQR 기반 이상치 탐지")
    print(f"  - 평균: {stats['mean']:.2f}")
    print(f"  - 중앙값: {stats['median']:.2f}")
    print(f"  - 표준편차: {stats['std']:.2f}")
    print(f"  - 최소값: {stats['min']:.0f}")
    print(f"  - 최대값: {stats['max']:.0f}")
    print(f"  - Q1: {stats['q1']:.2f}, Q3: {stats['q3']:.2f}, IQR: {stats['iqr']:.2f}")
    print(f"  - 이상치 범위: < {stats['lower_bound']:.2f} 또는 > {stats['upper_bound']:.2f}")
    print(f"  - 이상치 비율: {stats['outlier_ratio']:.2%}")
    print(f"  - 이상치 개수: {len(df[mask])}/{len(df)}")
    
    outliers = df.loc[mask, ["name", "cas", column]].sort_values(column, ascending=False)
    if not outliers.empty:
        print(f"\n  상위 {min(head, len(outliers))}개 이상치:")
        print(outliers.head(head).to_string(index=False))


def duplicate_report(df: pd.DataFrame) -> None:
    """중복 데이터 리포트"""
    print("\n[중복 확인]")
    
    # CAS 기준 중복
    duplicated_cas = df[df["cas"].duplicated(keep=False) & df["cas"].notna()]
    if duplicated_cas.empty:
        print("  - CAS 기준 중복 없음")
    else:
        print(f"  - CAS 기준 중복 {len(duplicated_cas)}건")
        print(duplicated_cas[["name", "cas"]].sort_values("cas").head(10).to_string(index=False))
    
    # 이름 기준 중복
    duplicated_name = df[df["name"].duplicated(keep=False)]
    if duplicated_name.empty:
        print("  - 이름 기준 중복 없음")
    else:
        print(f"  - 이름 기준 중복 {len(duplicated_name)}건")
        print(duplicated_name[["name", "cas"]].sort_values("name").head(10).to_string(index=False))
    
    # SMILES 기준 중복
    duplicated_smiles = df[df["smiles"].duplicated(keep=False) & df["smiles"].notna()]
    if duplicated_smiles.empty:
        print("  - SMILES 기준 중복 없음")
    else:
        print(f"  - SMILES 기준 중복 {len(duplicated_smiles)}건")
        print(duplicated_smiles[["name", "cas", "smiles"]].sort_values("smiles").head(10).to_string(index=False))


def analyze_blender_groups(df: pd.DataFrame) -> None:
    """블렌더 그룹 분석"""
    print("\n[블렌더 그룹 분석]")
    
    # 모든 그룹 수집
    all_groups = []
    for groups in df["blender_groups"]:
        if isinstance(groups, list):
            all_groups.extend(groups)
    
    if not all_groups:
        print("  - 블렌더 그룹 정보 없음")
        return
    
    group_counter = Counter(all_groups)
    print(f"  - 총 고유 그룹 수: {len(group_counter)}")
    print(f"  - 총 그룹 등장 횟수: {len(all_groups):,}")
    
    # 상위 그룹
    print(f"\n  상위 20개 그룹:")
    for group, count in group_counter.most_common(20):
        print(f"    - {group}: {count:,}회")
    
    # 그룹별 분자 수
    group_to_molecules = {}
    for idx, groups in df["blender_groups"].items():
        if isinstance(groups, list):
            unique_groups = set(groups)
            for group in unique_groups:
                if group not in group_to_molecules:
                    group_to_molecules[group] = set()
                group_to_molecules[group].add(idx)
    
    print(f"\n  그룹별 분자 수 (상위 20개):")
    group_molecule_counts = {g: len(mols) for g, mols in group_to_molecules.items()}
    for group, count in sorted(group_molecule_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"    - {group}: {count}개 분자")


def analyze_notes(df: pd.DataFrame) -> None:
    """노트 분석"""
    print("\n[노트 분석]")
    
    # 모든 노트 수집
    all_notes = []
    for notes in df["notes"]:
        if isinstance(notes, list):
            all_notes.extend(notes)
    
    if not all_notes:
        print("  - 노트 정보 없음")
        return
    
    note_counter = Counter(all_notes)
    print(f"  - 총 고유 노트 수: {len(note_counter)}")
    print(f"  - 총 노트 등장 횟수: {len(all_notes):,}")
    print(f"  - 평균 노트 중복도: {df['note_duplicates'].mean():.2f}")
    
    # 상위 노트
    print(f"\n  상위 20개 노트:")
    for note, count in note_counter.most_common(20):
        print(f"    - {note}: {count}회")
    
    # 노트 중복이 있는 분자
    duplicated_notes = df[df["note_duplicates"] > 0]
    if not duplicated_notes.empty:
        print(f"\n  노트 중복이 있는 분자: {len(duplicated_notes)}개")
        print(f"  - 평균 중복 수: {duplicated_notes['note_duplicates'].mean():.2f}")


def summarize(df: pd.DataFrame) -> None:
    """전체 요약 리포트"""
    print("=" * 60)
    print("complete_data EDA 요약")
    print("=" * 60)
    print(f"총 레코드 수: {len(df):,}")
    print(f"CAS 정보 보유 비율: {df['has_cas'].mean():.2%}")
    print(f"평균 노트 수: {df['note_count'].mean():.2f} (고유: {df['unique_note_count'].mean():.2f})")
    print(f"평균 블렌더 수: {df['blender_count'].mean():.2f}")
    print(f"평균 블렌더 그룹 수: {df['unique_group_count'].mean():.2f}")
    print(f"SMILES 문자열 평균 길이: {df['smiles_length'].mean():.2f}")

    # 결측치 리포트
    missing_report = missing_value_report(df)
    print("\n[결측치/빈 값 요약]")
    print(missing_report.to_string(index=False))

    # 중복 리포트
    duplicate_report(df)
    
    # 노트 분석
    analyze_notes(df)
    
    # 블렌더 그룹 분석
    analyze_blender_groups(df)
    
    # 이상치 분석
    print_outlier_summary(df, "note_count")
    print_outlier_summary(df, "blender_count")
    print_outlier_summary(df, "smiles_length")
    print_outlier_summary(df, "unique_group_count")


def main():
    """메인 함수"""
    if not COMPLETE_DATA_DIR.exists():
        raise FileNotFoundError(f"complete_data 디렉토리를 찾을 수 없습니다: {COMPLETE_DATA_DIR}")

    records = load_complete_data()
    if not records:
        raise RuntimeError("complete_data에서 로드할 수 있는 레코드가 없습니다.")

    df = to_dataframe(records)
    summarize(df)


if __name__ == "__main__":
    main()
