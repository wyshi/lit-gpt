"""
convert json to csv
"""

import json
import pandas as pd

from glob import glob
from tqdm import tqdm
from json.decoder import JSONDecodeError

def get_videos_json(video_dir):
    video_files = glob(video_dir + "/*.json")
    video_files.extend(glob(video_dir + "/culture/*.json"))
    videos_json = {}
    for video_file in tqdm(video_files):
        # print(video_file)
        with open(video_file) as fh:
            # file_json = json.load(fh)
            try:
                file_json = json.load(fh)
                if "data" not in file_json or "videos" not in file_json["data"]:
                    continue
                videos = file_json["data"]["videos"]
                videos_json.update({video["id"]: video for video in videos})
            except JSONDecodeError:
                pass
    # print(videos_json)
    return videos_json

def get_video_desc(videos_json, vid):
    video = videos_json[vid]
    video_desc = []
    if "stickersOnItem" in video:
        for itm in video["stickersOnItem"]:
            if "stickerText" in itm:
                video_desc.extend(itm["stickerText"])
    video_desc.append(video["video_description"])
    return "\n".join(video_desc)   

with open("/sailhome/lansong/CultureBank/tiktok/results_70b_chat_200.json") as fh:
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
    "norm",
    "legality",
]

# videos_json = get_videos_json("/juice/scr/weiyans/CultureBank/tiktok_data/videos")

df_results = []
for result in results:
    # comment = result[0][1]
    # vid = result[0][0]
    # video_desc = get_video_desc(videos_json, int(vid.split("-")[0]))
    
    vid = result[0][0]
    video_desc = result[0][1]
    comment = result[0][2]
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
        # print(output)
        # print()

        for o in output:
            record = [
                vid,
                video_desc,
                comment,
            ] + [o[k] for k in keys]
            df_results.append(record)
        if len(output) == 0:
            df_results.append([vid, video_desc, comment] + [None] * (len(keys)))
    except:
        # cannot convert to json
        continue

print(len(results), len(df_results))

pd.DataFrame(df_results, columns=["vid", "video_desc", "comment"] + keys).to_csv(
    "/sailhome/lansong/CultureBank/tiktok/results_70b_chat_200.csv", index=None
)
