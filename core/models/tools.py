import re

import numpy as np
import pandas as pd


def change_type_to_list(x):
    return np.array(
        [
            float(val)
            for val in re.split(
                "\s+", x.replace("[", "").replace("]", "").replace("\n", "")
            )
            if val
        ]
    )


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["embedding"] = df.embedding.apply(lambda x: change_type_to_list(x))
    return df
