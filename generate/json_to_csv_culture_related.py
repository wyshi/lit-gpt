"""
convert json to csv
"""

import json
import pandas as pd

from glob import glob
from tqdm import tqdm
from json.decoder import JSONDecodeError


with open("/sailhome/lansong/CultureBank/tiktok/results_70b_chat_500_culture_related.json") as fh:
    results = json.load(fh)



df_results = []
for result in results:
    vid = result[0][1]
    comment = result[0][0]
    try:
        res = result[1]
        res = res.replace("<EOD>", "").strip()
        
        pred = "Yes" if res.split()[0] == "Yes" else "No"
        record = [vid, comment, pred]
        df_results.append(record)
    except:
        # cannot convert to json
        continue

print(len(results), len(df_results))

pd.DataFrame(df_results, columns=["vid", "comment", "is_culture_related"]).to_csv(
    "/sailhome/lansong/CultureBank/tiktok/results_70b_chat_500_culture_related.csv", index=None
)
