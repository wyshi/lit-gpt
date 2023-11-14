"""
convert json to csv
"""

import json
import pandas as pd

from glob import glob
from tqdm import tqdm
from json.decoder import JSONDecodeError


with open("/sailhome/lansong/CultureBank/tiktok/results_70b_chat_500_comments.json") as fh:
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
    vid = result[0][1]
    comment = result[0][0]
    try:
        res = result[1]
        res = res.replace("<EOD>", "").strip()
        # print(res)
        # look for '[' & ']'
        start_index = res.find('[')
        end_index = res.rfind(']') if start_index != -1 else -1
        
        # if no array is found, try to find single json objects
        if start_index == -1 or end_index == -1:
            start_index = res.find('{')
            end_index = res.rfind('}')
            if start_index != -1 and end_index != -1:
                json_string = '[' + res[start_index:end_index+1] + ']'  # Wrap the object in a list
            else:
                json_string = '[]'  # No JSON object or array found
        else:
            # get the json array
            json_string = res[start_index:end_index+1]
        
        # output = json.loads(result[1].split("\n")[1].strip("Output: "))
        output = json.loads(json_string)
        print(output)
        print()

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

print(len(results), len(df_results))

pd.DataFrame(df_results, columns=["vid", "comment"] + keys).to_csv(
    "/sailhome/lansong/CultureBank/tiktok/results_70b_chat_500_comments.csv", index=None
)
