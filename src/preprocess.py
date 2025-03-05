# src/preprocess.py
import pandas as pd

def load_bpcc_data(file_path):
    """Load and clean the BPCC dataset."""
    df = pd.read_csv(file_path, delimiter="\t", names=["en", "target"])
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    data = load_bpcc_data("../data/bpcc_en_ml.tsv")  # Example: English-Malayalam
    print(data.head())
