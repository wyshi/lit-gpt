"""
convert json to csv
"""

import json
import pandas as pd

with open("/sailhome/weiyans/CultureBank/tiktok/results_70b_2k.json") as fh:
    results = json.load(fh)

keys = [
    "cultural group",
    "context",
    "goal",
    "relation",
    "actor",
    "recipient",
    "actor's behavior",
    "recipient's behavior",
    "other descriptions",
    "topic",
]
df_results = []
for result in results:
    comment = result[0][0]
    vid = result[0][1]
    try:
        output = json.loads(result[1].split("\n")[1].strip("Output: "))

        for o in output:
            record = [
                vid,
                comment,
            ] + [o[k] for k in keys]
            df_results.append(record)
        if len(output) == 0:
            df_results.append([vid, comment] + [None] * (len(keys)))
    except:
        # cannot convert to json
        continue
pd.DataFrame(df_results, columns=["vid", "comment"] + keys).to_csv(
    "/sailhome/weiyans/CultureBank/tiktok/results_70b_2k.csv", index=None
)
