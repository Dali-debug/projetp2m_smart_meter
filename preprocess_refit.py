import re
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "Processed_Data_CSV"
OUTPUT_DIR = "output_preprocessed"

# fréquence cible imposée pour la suite HMM
TARGET_FREQ = "8s"

# interpolation autorisée seulement sur petits trous
MAX_INTERP_GAP_STEPS = 3

# Hampel filter
HAMPEL_WINDOW = 7
HAMPEL_N_SIGMAS = 3.0

# valeurs physiques
MIN_POWER = 0.0
MAX_POWER = 30000.0

# minimum de longueur d'une séquence continue exploitable pour HMM
# 30 points à 8s = 240s = 4 minutes
MIN_HMM_RUN_LENGTH = 30

# compression gzip pour économiser l'espace disque
SAVE_COMPRESSED = True


# ============================================================
# OUTILS
# ============================================================

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def numeric_parse_ratio(series: pd.Series, sample_size: int = 2000) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    s = s.head(sample_size)
    parsed = pd.to_numeric(s, errors="coerce")
    return float(parsed.notna().mean())


def is_likely_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    return numeric_parse_ratio(series) >= threshold


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    candidates = []
    for col in df.columns:
        c = col.lower().strip()
        if any(k in c for k in ["time", "timestamp", "date", "datetime"]):
            candidates.append(col)

    if candidates:
        return candidates[0]

    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(20)
            success = 0
            for v in sample:
                try:
                    pd.to_datetime(v)
                    success += 1
                except Exception:
                    pass
            if len(sample) > 0 and success >= max(3, len(sample) // 2):
                return col

    return None


def detect_aggregate_power_column(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None
) -> Optional[str]:
    exclude_columns = exclude_columns or []

    priority_patterns = [
        r"^aggregate$",
        r"^aggregate_power$",
        r"^power$",
        r"^mains$",
        r"^mains_power$",
        r"^total_power$",
        r"^whole_house$",
        r"^apparent_power$",
        r"^active_power$",
    ]

    cols = [c for c in df.columns if c not in exclude_columns]

    # priorité aux noms explicites et colonnes numériques ou quasi-numériques
    for pat in priority_patterns:
        for col in cols:
            if re.search(pat, col.strip().lower()) and is_likely_numeric(df[col]):
                return col

    weak_keywords = ["aggregate", "total", "mains", "house", "active", "power"]
    scored = []

    for col in cols:
        c = col.strip().lower()
        keyword_score = sum(1 for k in weak_keywords if k in c)
        parse_ratio = numeric_parse_ratio(df[col])

        if keyword_score > 0 and parse_ratio >= 0.8:
            # score mixte : mots-clés + capacité de conversion numérique
            score = keyword_score + parse_ratio
            scored.append((score, col))

    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    # fallback: première colonne réellement numérique / convertible
    numeric_like = []
    for col in cols:
        parse_ratio = numeric_parse_ratio(df[col])
        if parse_ratio >= 0.8:
            numeric_like.append((parse_ratio, col))

    if numeric_like:
        numeric_like.sort(reverse=True)
        return numeric_like[0][1]

    return None


def infer_house_id(file_path: Path) -> str:
    match = re.search(r"house[_\s-]?(\d+)", file_path.stem.lower())
    if match:
        return f"House_{match.group(1)}"
    return file_path.stem


def hampel_filter(
    series: pd.Series,
    window_size: int = 7,
    n_sigmas: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    x = series.copy()
    corrected_mask = pd.Series(False, index=x.index)

    rolling_median = x.rolling(window=2 * window_size + 1, center=True).median()
    diff = (x - rolling_median).abs()
    mad = diff.rolling(window=2 * window_size + 1, center=True).median()

    threshold = n_sigmas * 1.4826 * mad
    outliers = (diff > threshold).fillna(False)

    x[outliers] = rolling_median[outliers]
    corrected_mask.loc[outliers.index] = outliers

    return x, corrected_mask


def build_quality_report(report_rows: List[Dict], output_dir: str) -> None:
    report_df = pd.DataFrame(report_rows)
    report_csv = Path(output_dir) / "preprocessing_report.csv"
    report_json = Path(output_dir) / "preprocessing_report.json"

    report_df.to_csv(report_csv, index=False, encoding="utf-8")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report_rows, f, indent=2, ensure_ascii=False)


def summarize_time_step(index: pd.DatetimeIndex) -> Optional[float]:
    if len(index) < 2:
        return None
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        return None
    return float(diffs.mode().iloc[0])


def build_run_metadata(valid_mask: pd.Series) -> Tuple[pd.Series, pd.Series]:
    valid_mask = valid_mask.fillna(False).astype(bool)

    if valid_mask.empty:
        empty_i = pd.Series(dtype="int32")
        return empty_i, empty_i

    grp = (valid_mask != valid_mask.shift(fill_value=False)).cumsum()
    run_length = pd.Series(
        np.where(valid_mask, valid_mask.groupby(grp).transform("sum"), 0),
        index=valid_mask.index,
        dtype="int32"
    )

    raw_run_id = pd.Series(np.where(valid_mask, grp, 0), index=valid_mask.index)
    unique_valid_ids = [v for v in pd.unique(raw_run_id) if v != 0]
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_valid_ids, start=1)}

    run_id = raw_run_id.map(id_map).fillna(0).astype("int32")

    return run_id, run_length.astype("int32")


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_one_file(file_path: Path, output_dir: str) -> Dict:
    house_id = infer_house_id(file_path)

    raw = pd.read_csv(file_path, low_memory=False)

    original_rows = len(raw)
    original_cols = list(raw.columns)

    ts_col = detect_timestamp_column(raw)
    if ts_col is None:
        raise ValueError(f"{house_id}: impossible de détecter la colonne timestamp.")

    power_col = detect_aggregate_power_column(raw, exclude_columns=[ts_col])
    if power_col is None:
        raise ValueError(f"{house_id}: impossible de détecter la colonne puissance agrégée.")

    df = raw[[ts_col, power_col]].copy()
    df.columns = ["timestamp", "aggregate_power_raw"]

    # conversion timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_timestamps = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        raise ValueError(f"{house_id}: plus aucune ligne après conversion timestamp.")

    # conversion puissance
    df["aggregate_power_raw"] = pd.to_numeric(df["aggregate_power_raw"], errors="coerce")
    invalid_power = int(df["aggregate_power_raw"].isna().sum())

    # tri initial
    df = df.sort_values("timestamp")

    # alignement explicite sur la grille 8s
    df["timestamp"] = df["timestamp"].dt.round(TARGET_FREQ)

    # doublons après alignement
    duplicated_timestamps = int(df["timestamp"].duplicated().sum())

    # fusion des doublons créés par l'arrondi
    df = (
        df.groupby("timestamp", as_index=False)["aggregate_power_raw"]
        .mean()
        .sort_values("timestamp")
    )

    if df.empty:
        raise ValueError(f"{house_id}: aucune donnée exploitable après groupby timestamp.")

    # index temps
    df = df.set_index("timestamp")

    detected_step_seconds = summarize_time_step(df.index)

    # bornes physiques
    invalid_low = int((df["aggregate_power_raw"] < MIN_POWER).sum())
    invalid_high = int((df["aggregate_power_raw"] > MAX_POWER).sum())

    df.loc[df["aggregate_power_raw"] < MIN_POWER, "aggregate_power_raw"] = np.nan
    df.loc[df["aggregate_power_raw"] > MAX_POWER, "aggregate_power_raw"] = np.nan

    if pd.isna(df.index.min()) or pd.isna(df.index.max()):
        raise ValueError(f"{house_id}: index temporel invalide après nettoyage.")

    # rééchantillonnage sur grille fixe
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=TARGET_FREQ)
    if len(full_index) == 0:
        raise ValueError(f"{house_id}: impossible de construire la grille cible.")

    df = df.reindex(full_index)
    df.index.name = "timestamp"

    # marquage des manquants
    df["was_missing_before_interp"] = df["aggregate_power_raw"].isna()

    # taille des gaps
    is_na = df["aggregate_power_raw"].isna()
    grp = (is_na != is_na.shift(fill_value=False)).cumsum()
    gap_size = is_na.groupby(grp).transform("sum")
    df["gap_size_steps"] = np.where(is_na, gap_size, 0).astype(np.int32)

    # petits / grands gaps
    small_gap_mask = is_na & (df["gap_size_steps"] <= MAX_INTERP_GAP_STEPS)
    large_gap_mask = is_na & (df["gap_size_steps"] > MAX_INTERP_GAP_STEPS)

    # interpolation limitée à l'intérieur de la série
    interp_series = df["aggregate_power_raw"].copy()
    interp_series = interp_series.interpolate(
        method="time",
        limit=MAX_INTERP_GAP_STEPS,
        limit_direction="both",
        limit_area="inside"
    )

    # sécurité: les grands gaps restent NaN
    interp_series.loc[large_gap_mask] = np.nan

    df["aggregate_power_interp"] = interp_series
    df["is_interpolated"] = small_gap_mask & df["aggregate_power_interp"].notna()
    df["is_large_gap"] = large_gap_mask

    # Hampel
    filtered, outlier_mask = hampel_filter(
        df["aggregate_power_interp"],
        window_size=HAMPEL_WINDOW,
        n_sigmas=HAMPEL_N_SIGMAS
    )

    # sécurité supplémentaire
    filtered.loc[df["is_large_gap"]] = np.nan

    df["aggregate_power"] = filtered
    df["is_outlier_corrected"] = outlier_mask.fillna(False) & df["aggregate_power"].notna()
    df["house_id"] = house_id

    # préparation HMM sans segmentation de fichiers
    candidate_valid_mask = df["aggregate_power"].notna()

    candidate_run_id, candidate_run_length = build_run_metadata(candidate_valid_mask)

    # on ne garde comme HMM-ready que les runs suffisamment longs
    df["is_valid_for_hmm"] = candidate_valid_mask & (candidate_run_length >= MIN_HMM_RUN_LENGTH)

    hmm_run_id, hmm_run_length = build_run_metadata(df["is_valid_for_hmm"])

    df["hmm_run_id"] = hmm_run_id
    df["hmm_run_length"] = hmm_run_length
    df["aggregate_power_for_hmm"] = df["aggregate_power"].where(df["is_valid_for_hmm"], np.nan)

    # optimisation légère taille disque
    for col in [
        "aggregate_power_raw",
        "aggregate_power_interp",
        "aggregate_power",
        "aggregate_power_for_hmm",
    ]:
        df[col] = df[col].astype("float32")

    for col in [
        "was_missing_before_interp",
        "is_interpolated",
        "is_large_gap",
        "is_outlier_corrected",
        "is_valid_for_hmm",
    ]:
        df[col] = df[col].astype(bool)

    df["gap_size_steps"] = df["gap_size_steps"].astype("int32")
    df["hmm_run_id"] = df["hmm_run_id"].astype("int32")
    df["hmm_run_length"] = df["hmm_run_length"].astype("int32")

    # export
    suffix = ".csv.gz" if SAVE_COMPRESSED else ".csv"
    output_file = Path(output_dir) / f"{house_id}_clean{suffix}"

    export_cols = [
        "house_id",
        "aggregate_power_raw",
        "aggregate_power_interp",
        "aggregate_power",
        "aggregate_power_for_hmm",
        "was_missing_before_interp",
        "is_interpolated",
        "is_large_gap",
        "is_outlier_corrected",
        "gap_size_steps",
        "is_valid_for_hmm",
        "hmm_run_id",
        "hmm_run_length",
    ]

    df.reset_index().to_csv(
        output_file,
        index=False,
        columns=["timestamp"] + export_cols,
        encoding="utf-8",
        compression="gzip" if SAVE_COMPRESSED else None
    )

    valid_after = int(df["aggregate_power"].notna().sum())
    total_after = len(df)

    hmm_runs_df = (
        df.loc[df["hmm_run_id"] > 0, ["hmm_run_id", "hmm_run_length"]]
        .drop_duplicates(subset=["hmm_run_id"])
        .sort_values("hmm_run_id")
    )

    num_hmm_runs = int(len(hmm_runs_df))
    median_hmm_run_length = float(hmm_runs_df["hmm_run_length"].median()) if not hmm_runs_df.empty else 0.0
    max_hmm_run_length = int(hmm_runs_df["hmm_run_length"].max()) if not hmm_runs_df.empty else 0

    report = {
        "house_id": house_id,
        "file_name": file_path.name,
        "detected_timestamp_column": ts_col,
        "detected_aggregate_power_column": power_col,
        "original_rows": int(original_rows),
        "original_columns": original_cols,
        "invalid_timestamps": int(invalid_timestamps),
        "invalid_power_values_before_cleaning": int(invalid_power),
        "duplicated_timestamps_removed": int(duplicated_timestamps),
        "detected_step_seconds": detected_step_seconds,
        "target_frequency": TARGET_FREQ,
        "physically_invalid_low_values": int(invalid_low),
        "physically_invalid_high_values": int(invalid_high),
        "rows_after_resampling": int(total_after),
        "missing_before_interp": int(df["was_missing_before_interp"].sum()),
        "interpolated_points": int(df["is_interpolated"].sum()),
        "large_gap_points": int(df["is_large_gap"].sum()),
        "outlier_corrected_points": int(df["is_outlier_corrected"].sum()),
        "valid_points_final": int(valid_after),
        "missing_points_final": int(df["aggregate_power"].isna().sum()),
        "min_hmm_run_length_steps": int(MIN_HMM_RUN_LENGTH),
        "points_valid_for_hmm": int(df["is_valid_for_hmm"].sum()),
        "points_excluded_for_hmm_short_runs": int(candidate_valid_mask.sum() - df["is_valid_for_hmm"].sum()),
        "num_hmm_runs": num_hmm_runs,
        "median_hmm_run_length_steps": median_hmm_run_length,
        "max_hmm_run_length_steps": max_hmm_run_length,
        "output_file": str(output_file),
    }

    return report


def list_house_files(input_dir: str) -> List[Path]:
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Dossier introuvable: {input_dir}")

    files = sorted(p.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {input_dir}")

    return files


def main():
    ensure_dir(OUTPUT_DIR)

    files = list_house_files(INPUT_DIR)
    reports = []
    failures = []

    print(f"\nNombre de fichiers trouvés: {len(files)}")
    print(f"Dossier entrée: {INPUT_DIR}")
    print(f"Dossier sortie: {OUTPUT_DIR}")
    print(f"Fréquence cible: {TARGET_FREQ}")
    print(f"Min HMM run length: {MIN_HMM_RUN_LENGTH} pas")
    print(f"Compression gzip: {SAVE_COMPRESSED}\n")

    for file_path in tqdm(files, desc="Preprocessing REFIT"):
        try:
            report = preprocess_one_file(file_path, OUTPUT_DIR)
            reports.append(report)
        except Exception as e:
            failures.append({
                "file_name": file_path.name,
                "error": str(e)
            })

    build_quality_report(reports, OUTPUT_DIR)

    if failures:
        fail_path = Path(OUTPUT_DIR) / "failures.json"
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)

    print("\nTerminé.")
    print(f"Rapport principal: {Path(OUTPUT_DIR) / 'preprocessing_report.csv'}")
    if failures:
        print(f"Erreurs: {Path(OUTPUT_DIR) / 'failures.json'}")


if __name__ == "__main__":
    main()