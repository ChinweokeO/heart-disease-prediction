import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def generate_production_data(reference: pd.DataFrame, test_size: float = 0.3) -> pd.DataFrame:
    _, production = train_test_split(reference, test_size=test_size, random_state=42)
    return production


def run_evidently_drift_detection(reference: pd.DataFrame, production: pd.DataFrame) -> Report:
    """Run Evidently data drift detection and return the report."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=production)
    return report


def build_report(reference: pd.DataFrame, production: pd.DataFrame) -> Report:
    return run_evidently_drift_detection(reference, production)


def summarize_report(report: Report):
    report_dict = report.as_dict()
    metrics = report_dict.get("metrics", [])
    if not metrics:
        return [], 0.0, False

    result = metrics[0].get("result", {})
    dataset_drift = bool(result.get("dataset_drift", False))
    drift_columns = []
    data = result.get("data", {})
    drift_by_columns = data.get("drift_by_columns", [])

    for column in drift_by_columns:
        name = column.get("column_name") or column.get("feature_name") or "<unknown>"
        drifted = bool(column.get("drifted", False))
        score = column.get("drift_score", column.get("drift_stat", None))
        drift_columns.append((name, drifted, score))

    drift_share = 0.0
    if drift_columns:
        drift_share = sum(1 for _, drifted, _ in drift_columns if drifted) / len(drift_columns)

    return drift_columns, drift_share, dataset_drift


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor data drift with Evidently.")
    parser.add_argument("--reference-data", type=Path, default=None, help="Path to reference CSV file.")
    parser.add_argument("--production-data", type=Path, default=None, help="Path to production CSV file.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Feature drift share threshold for failure.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory to save the HTML report.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    reference_path = args.reference_data or project_root / "data" / "train.csv"
    if not reference_path.exists():
        print(f"Reference data not found at {reference_path}", file=sys.stderr)
        sys.exit(1)

    reference_df = load_csv(reference_path)

    if args.production_data:
        production_path = args.production_data
        if not production_path.exists():
            print(f"Production data not found at {production_path}", file=sys.stderr)
            sys.exit(1)
        production_df = load_csv(production_path)
    else:
        production_df = generate_production_data(reference_df)

    print("Running Evidently data drift detection...")
    report = run_evidently_drift_detection(reference_df, production_df)
    report_dir = args.report_dir or project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "drift_report.html"
    report.save_html(str(report_path))

    drift_columns, drift_share, dataset_drift = summarize_report(report)

    print("Drift detection summary")
    print(f"Overall drift share: {drift_share:.2%}")
    print(f"Dataset drift detected: {dataset_drift}")
    if drift_columns:
        print("Feature drift status:")
        for name, drifted, score in drift_columns:
            status = "drifted" if drifted else "ok"
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else score
            print(f" - {name}: {status} (score={score_text})")
    else:
        print("No feature drift details available.")

    print(f"HTML report saved to {report_path}")

    if drift_share > args.threshold:
        print(f"Drift share {drift_share:.2%} exceeds threshold {args.threshold:.2%}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
