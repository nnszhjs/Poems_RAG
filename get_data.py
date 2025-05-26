import pandas as pd
from typing import List, Dict, Any, Optional
def get_poems_data(save_path: Optional[str] = None) -> pd.DataFrame:
    splits = {'train': 'data/train-00000-of-00001-6914ee5fabc145c0.parquet', 'test': 'data/test-00000-of-00001-a794cd4c018c9326.parquet'}
    df = pd.read_parquet("hf://datasets/xmj2002/tang_poems/" + splits["train"])
    print(df.head())
    # 保存数据
    if save_path is None:
        save_path = "./data/tang_poems.csv"
    df.to_csv(save_path, index=False)
    return df