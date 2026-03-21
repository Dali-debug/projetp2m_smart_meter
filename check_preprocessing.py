from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# mets ici seulement le nom logique de la maison, sans extension
FILE_BASENAME = "House_3_clean"

START_ROW = 0
NROWS = 1500

OUTPUT_DIR = Path("output_preprocessed")
MAX_GAP_INTERVALS_TO_SHADE = 120


# ============================================================
# OUTILS
# ============================================================

def to_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
        .fillna(False)
        .astype(bool)
    )


def resolve_input_file(file_basename: str, output_dir: Path) -> Path:
    candidates = [
        output_dir / f"{file_basename}.csv.gz",
        output_dir / f"{file_basename}.csv",
        Path(file_basename),
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Aucun fichier trouvé pour {file_basename}. "
        f"Vérifie dans {output_dir} avec extension .csv.gz ou .csv"
    )


def build_run_metadata(valid_mask: pd.Series):
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


def load_window(file_path: Path, start_row: int, nrows: int) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    available_columns = pd.read_csv(file_path, nrows=0).columns.tolist()

    base_required_columns = [
        "timestamp",
        "aggregate_power_raw",
        "aggregate_power_interp",
        "aggregate_power",
        "is_interpolated",
        "is_large_gap",
        "is_outlier_corrected",
    ]

    optional_columns = [
        "aggregate_power_for_hmm",
        "is_valid_for_hmm",
        "hmm_run_id",
        "hmm_run_length",
    ]

    missing_required = [c for c in base_required_columns if c not in available_columns]
    if missing_required:
        raise ValueError(f"Colonnes obligatoires manquantes: {missing_required}")

    usecols = base_required_columns + [c for c in optional_columns if c in available_columns]

    skiprows = range(1, start_row + 1) if start_row > 0 else None

    df = pd.read_csv(
        file_path,
        usecols=usecols,
        skiprows=skiprows,
        nrows=nrows,
        low_memory=False,
    )

    if df.empty:
        raise ValueError("La fenêtre lue est vide.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ["aggregate_power_raw", "aggregate_power_interp", "aggregate_power"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "aggregate_power_for_hmm" in df.columns:
        df["aggregate_power_for_hmm"] = pd.to_numeric(df["aggregate_power_for_hmm"], errors="coerce")
    else:
        df["aggregate_power_for_hmm"] = df["aggregate_power"]

    for col in ["is_interpolated", "is_large_gap", "is_outlier_corrected"]:
        df[col] = to_bool_series(df[col])

    if "is_valid_for_hmm" in df.columns:
        df["is_valid_for_hmm"] = to_bool_series(df["is_valid_for_hmm"])
    else:
        df["is_valid_for_hmm"] = df["aggregate_power_for_hmm"].notna()

    if "hmm_run_id" in df.columns:
        df["hmm_run_id"] = pd.to_numeric(df["hmm_run_id"], errors="coerce").fillna(0).astype("int32")
    else:
        df["hmm_run_id"], _ = build_run_metadata(df["is_valid_for_hmm"])

    if "hmm_run_length" in df.columns:
        df["hmm_run_length"] = pd.to_numeric(df["hmm_run_length"], errors="coerce").fillna(0).astype("int32")
    else:
        _, df["hmm_run_length"] = build_run_metadata(df["is_valid_for_hmm"])

    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Aucune ligne exploitable après conversion.")

    return df


def find_gap_intervals(df: pd.DataFrame, gap_col: str = "is_large_gap"):
    intervals = []

    if df.empty:
        return intervals

    mask = df[gap_col].fillna(False).astype(bool).tolist()
    in_gap = False
    start_ts = None

    for i, is_gap in enumerate(mask):
        ts = df.iloc[i]["timestamp"]

        if pd.isna(ts):
            continue

        if is_gap and not in_gap:
            in_gap = True
            start_ts = ts
        elif not is_gap and in_gap:
            end_ts = df.iloc[i - 1]["timestamp"]
            if pd.notna(start_ts) and pd.notna(end_ts) and end_ts >= start_ts:
                intervals.append((start_ts, end_ts))
            in_gap = False
            start_ts = None

    if in_gap:
        end_ts = df.iloc[-1]["timestamp"]
        if pd.notna(start_ts) and pd.notna(end_ts) and end_ts >= start_ts:
            intervals.append((start_ts, end_ts))

    return intervals


def shade_gaps(ax, intervals, max_intervals: int = 120):
    if len(intervals) > max_intervals:
        return False

    for i, (start_ts, end_ts) in enumerate(intervals):
        ax.axvspan(
            start_ts,
            end_ts,
            alpha=0.12,
            color="lightblue",
            label="Grand gap" if i == 0 else None,
        )
    return True


def compute_metrics(df: pd.DataFrame) -> dict:
    n = len(df)

    raw_missing = int(df["aggregate_power_raw"].isna().sum())
    interp_count = int(df["is_interpolated"].sum())
    gap_count = int(df["is_large_gap"].sum())
    outlier_count = int(df["is_outlier_corrected"].sum())
    valid_final = int(df["aggregate_power"].notna().sum())
    valid_hmm = int(df["is_valid_for_hmm"].sum())

    delta = df["aggregate_power"] - df["aggregate_power_raw"]
    delta_valid = delta.dropna()

    mean_abs_delta = float(delta_valid.abs().mean()) if not delta_valid.empty else np.nan
    max_abs_delta = float(delta_valid.abs().max()) if not delta_valid.empty else np.nan

    runs_df = (
        df.loc[df["hmm_run_id"] > 0, ["hmm_run_id", "hmm_run_length"]]
        .drop_duplicates(subset=["hmm_run_id"])
        .sort_values("hmm_run_id")
    )

    num_hmm_runs = int(len(runs_df))
    median_hmm_run_length = float(runs_df["hmm_run_length"].median()) if not runs_df.empty else 0.0
    max_hmm_run_length = int(runs_df["hmm_run_length"].max()) if not runs_df.empty else 0

    return {
        "rows_window": n,
        "raw_missing_pct": 100 * raw_missing / n if n else 0.0,
        "interpolated_pct": 100 * interp_count / n if n else 0.0,
        "large_gap_pct": 100 * gap_count / n if n else 0.0,
        "outlier_pct": 100 * outlier_count / n if n else 0.0,
        "valid_final_pct": 100 * valid_final / n if n else 0.0,
        "valid_for_hmm_pct": 100 * valid_hmm / n if n else 0.0,
        "num_hmm_runs": num_hmm_runs,
        "median_hmm_run_length": median_hmm_run_length,
        "max_hmm_run_length": max_hmm_run_length,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
    }


def qualitative_assessment(metrics: dict) -> str:
    gap_pct = metrics["large_gap_pct"]
    outlier_pct = metrics["outlier_pct"]
    mean_abs_delta = metrics["mean_abs_delta"]
    valid_hmm_pct = metrics["valid_for_hmm_pct"]

    if not np.isfinite(mean_abs_delta):
        return "À surveiller"

    if valid_hmm_pct >= 85 and gap_pct < 10 and outlier_pct < 2 and mean_abs_delta < 20:
        return "Bon"
    if valid_hmm_pct >= 60 and gap_pct < 25 and outlier_pct < 5 and mean_abs_delta < 50:
        return "Acceptable"
    return "À surveiller"


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_path = resolve_input_file(FILE_BASENAME, OUTPUT_DIR)
    df = load_window(file_path, START_ROW, NROWS)
    df["delta_cleaning"] = df["aggregate_power"] - df["aggregate_power_raw"]

    metrics = compute_metrics(df)
    status = qualitative_assessment(metrics)
    gap_intervals = find_gap_intervals(df, "is_large_gap")

    print("\n===== CHECK PREPROCESSING =====")
    print(f"Fichier                  : {file_path.name}")
    print(f"Lignes analysées         : {metrics['rows_window']}")
    print(f"Raw missing (%)          : {metrics['raw_missing_pct']:.2f}")
    print(f"Interpolated (%)         : {metrics['interpolated_pct']:.2f}")
    print(f"Large gap (%)            : {metrics['large_gap_pct']:.2f}")
    print(f"Outlier corrected (%)    : {metrics['outlier_pct']:.2f}")
    print(f"Valid final (%)          : {metrics['valid_final_pct']:.2f}")
    print(f"Valid for HMM (%)        : {metrics['valid_for_hmm_pct']:.2f}")
    print(f"Nombre de runs HMM       : {metrics['num_hmm_runs']}")
    print(f"Median run length (pas)  : {metrics['median_hmm_run_length']:.1f}")
    print(f"Max run length (pas)     : {metrics['max_hmm_run_length']}")
    print(f"Mean abs delta           : {metrics['mean_abs_delta']:.4f}")
    print(f"Max abs delta            : {metrics['max_abs_delta']:.4f}")
    print(f"Évaluation               : {status}")
    print("================================\n")

    fig, axes = plt.subplots(
        3, 1,
        figsize=(16, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.5, 1.6]}
    )

    ax1, ax2, ax3 = axes

    shade_ok_1 = shade_gaps(ax1, gap_intervals, MAX_GAP_INTERVALS_TO_SHADE)

    ax1.plot(df["timestamp"], df["aggregate_power_raw"], label="Brut", alpha=0.40, linewidth=1.0)
    ax1.plot(df["timestamp"], df["aggregate_power_interp"], label="Après interpolation", alpha=0.85, linewidth=1.0)
    ax1.plot(df["timestamp"], df["aggregate_power"], label="Signal final nettoyé", linewidth=1.4)
    ax1.plot(df["timestamp"], df["aggregate_power_for_hmm"], label="Signal prêt HMM", linewidth=1.8)

    interp_points = df[df["is_interpolated"]]
    if not interp_points.empty:
        ax1.scatter(
            interp_points["timestamp"],
            interp_points["aggregate_power_interp"],
            label="Points interpolés",
            s=14,
            marker="o",
        )

    outlier_points = df[df["is_outlier_corrected"]]
    if not outlier_points.empty:
        ax1.scatter(
            outlier_points["timestamp"],
            outlier_points["aggregate_power"],
            label="Points corrigés (Hampel)",
            s=18,
            marker="x",
        )

    summary_text = (
        f"Évaluation : {status}\n"
        f"Raw missing : {metrics['raw_missing_pct']:.2f}%\n"
        f"Interpolés : {metrics['interpolated_pct']:.2f}%\n"
        f"Large gaps : {metrics['large_gap_pct']:.2f}%\n"
        f"Valide HMM : {metrics['valid_for_hmm_pct']:.2f}%\n"
        f"Runs HMM : {metrics['num_hmm_runs']}\n"
        f"Median run : {metrics['median_hmm_run_length']:.1f}\n"
        f"Mean |Δ| : {metrics['mean_abs_delta']:.3f}"
    )

    ax1.text(
        0.99,
        0.98,
        summary_text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
    )

    if not shade_ok_1:
        ax1.text(
            0.01,
            0.02,
            "Trop de gaps pour les ombrer tous proprement sur cette fenêtre.",
            transform=ax1.transAxes,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax1.set_title(f"Préprocessing - contrôle HMM-ready - {file_path.stem}")
    ax1.set_ylabel("Puissance")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="upper left")

    shade_gaps(ax2, gap_intervals, MAX_GAP_INTERVALS_TO_SHADE)
    ax2.plot(df["timestamp"], df["delta_cleaning"], label="Delta (final - brut)", linewidth=1.2)
    ax2.axhline(0, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Delta")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left")

    gap_points = df[df["is_large_gap"]]
    interp_points = df[df["is_interpolated"]]
    outlier_points = df[df["is_outlier_corrected"]]
    hmm_points = df[df["is_valid_for_hmm"]]

    if not gap_points.empty:
        ax3.scatter(gap_points["timestamp"], np.full(len(gap_points), 4), s=10, marker="|", label="Large gap")
    if not interp_points.empty:
        ax3.scatter(interp_points["timestamp"], np.full(len(interp_points), 3), s=10, marker="|", label="Interpolé")
    if not outlier_points.empty:
        ax3.scatter(outlier_points["timestamp"], np.full(len(outlier_points), 2), s=10, marker="|", label="Outlier corrigé")
    if not hmm_points.empty:
        ax3.scatter(hmm_points["timestamp"], np.full(len(hmm_points), 1), s=10, marker="|", label="Valide HMM")

    ax3.set_yticks([1, 2, 3, 4])
    ax3.set_yticklabels(["HMM", "Outlier", "Interpolé", "Gap"])
    ax3.set_ylim(0.5, 4.5)
    ax3.set_ylabel("Flags")
    ax3.set_xlabel("Timestamp")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="upper left")

    plt.tight_layout()

    out = OUTPUT_DIR / f"{file_path.stem}_objective_check.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()