import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

# Configuration
DRIFT_THRESHOLD = 0.3  # Exit with code 1 if drift share exceeds 30%
REPORTS_DIR = "reports"
DATA_DIR = "data"

def load_reference_data(filepath):
    """Load training data as reference distribution"""
    print(f"Loading reference data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Reference data shape: {df.shape}")
    return df

def load_or_generate_production_data(reference_df, production_filepath=None, drift_ratio=0.5):
    """
    Load or generate production dataset.
    If production_filepath is provided, load from it.
    Otherwise, split reference data and apply drift to simulation.
    """
    if production_filepath and os.path.exists(production_filepath):
        print(f"Loading production data from {production_filepath}...")
        production_df = pd.read_csv(production_filepath)
    else:
        print("Generating production dataset by splitting reference data with drift simulation...")
        # Split the data
        split_idx = int(len(reference_df) * drift_ratio)
        production_df = reference_df.iloc[split_idx:].copy()
        
        # Apply synthetic drift to numeric columns
        numeric_cols = production_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Add slight shift and noise to simulate drift
            production_df[col] = production_df[col] + np.random.normal(
                loc=production_df[col].std() * 0.2,
                scale=production_df[col].std() * 0.1,
                size=len(production_df)
            )
    
    print(f"Production data shape: {production_df.shape}")
    return production_df

def run_drift_detection(reference_df, production_df):
    """Run Evidently drift detection on all features"""
    print("\nRunning drift detection...")
    
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df, current_data=production_df)
    
    return snapshot

def print_drift_summary(snapshot):
    """Print summary of drifted features and overall drift share"""
    print("\n" + "="*60)
    print("DRIFT DETECTION SUMMARY")
    print("="*60)
    
    # Extract drift results from the Evidently snapshot
    results = snapshot.dict()
    metrics = results.get("metrics", [])
    
    drifted_features = []
    total_features = 0
    drift_share = 0.0
    
    for metric in metrics:
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {})
        value = metric.get("value")
        
        if metric_name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            drift_share = value.get("share", drift_share)
            print(f"Overall Drift Share: {drift_share:.2%}")
            continue
        
        if config.get("type", "").endswith("ValueDrift"):
            column = config.get("column")
            threshold = config.get("threshold")
            if column is None or not isinstance(value, (int, float)):
                continue
            total_features += 1
            is_drifted = threshold is not None and value < threshold
            if is_drifted:
                drifted_features.append(column)
    
    if drifted_features:
        print(f"\nDrifted Features ({len(drifted_features)}):")
        for feature in drifted_features:
            print(f"  - {feature}")
    else:
        print("\nNo significant drift detected in features.")
    
    print(f"\nTotal Features Analyzed: {total_features}")
    print("="*60)
    
    return drift_share

def save_report(report, output_dir=REPORTS_DIR):
    
    """Save HTML report to reports directory"""
    Path(output_dir).mkdir(exist_ok=True)
    report_path = os.path.join(output_dir, "drift_report.html")
    report.save_html(report_path)
    print(f"\nHTML report saved to: {report_path}")
    return report_path

def main():
    """Main execution function"""
    # Load reference data
    ref_data_path = os.path.join(DATA_DIR, "heart_disease.csv")
    if not os.path.exists(ref_data_path):
        print(f"Error: Reference data not found at {ref_data_path}")
        sys.exit(1)
    
    reference_df = load_reference_data(ref_data_path)
    
    # Load or generate production data
    production_data_path = os.path.join(DATA_DIR, "production.csv")
    production_df = load_or_generate_production_data(reference_df, production_data_path)
    
    # Run drift detection
    report = run_drift_detection(reference_df, production_df)
    
    # Print summary
    drift_share = print_drift_summary(report)
    
    # Save report
    save_report(report)
    
    # Check threshold and exit with appropriate code
    if drift_share > DRIFT_THRESHOLD:
        print(f"\n⚠️  ALERT: Drift share ({drift_share:.2%}) exceeds threshold ({DRIFT_THRESHOLD:.2%})")
        sys.exit(1)
    else:
        print(f"\n✓ Drift share ({drift_share:.2%}) is within acceptable threshold ({DRIFT_THRESHOLD:.2%})")
        sys.exit(0)

if __name__ == "__main__":
    main()
