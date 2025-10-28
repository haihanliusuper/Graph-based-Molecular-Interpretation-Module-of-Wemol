#!/usr/bin/env python
# fullpipeline.py

import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def run(cmd):
    print(f"🚀 Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete pipeline for GNN training and prediction")
    parser.add_argument('--task_type', choices=['classification', 'regression'], required=True, help='Type of task')
    parser.add_argument('--do_train', action='store_true', help='Run training')
    parser.add_argument('--do_predict', action='store_true', help='Run prediction')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV data for training')
    parser.add_argument('--smiles_col', default='Smiles', help='Column name for SMILES')
    parser.add_argument('--label_col', default='Label', help='Column name for label')
    parser.add_argument('--train_pdf', default='training_report.pdf', help='Output PDF report for training')
    parser.add_argument('--predict_input', type=str, required=True, help='Input file with SMILES for prediction')
    parser.add_argument('--predict_col', default='Smiles', help='Column name for SMILES in prediction file')
    parser.add_argument('--predict_output', default='predict_result.csv', help='Output file for prediction results')

    args = parser.parse_args()

    pt_file = SCRIPT_DIR / "dataset.pt"

    if args.do_train:
        # Save training data
        cmd_save_data = (
            f"python {SCRIPT_DIR}/save_data.py "
            f"--input_csv {args.csv} "
            f"--output_pt {pt_file} "
            f"--smiles_col {args.smiles_col} "
            f"--label_col {args.label_col}"
        )
        run(cmd_save_data)

        # Train model
        train_script = "Train_R.py" if args.task_type == 'regression' else "Train_C.py"
        cmd_train = (
            f"python {SCRIPT_DIR}/{train_script} "
            f"--ptinput {pt_file} "
            f"--output_pdf {args.train_pdf} "
        )
        run(cmd_train)

    if args.do_predict:
        # Run prediction
        cmd_predict = (
            f"python {SCRIPT_DIR}/predict.py "
            f"--predict_input {args.predict_input} "
            f"--predict_col {args.predict_col} "
            f"--output {args.predict_output} "
            f"--task_type {args.task_type}"
        )
        run(cmd_predict)

    print("✅ Full pipeline execution completed.")
