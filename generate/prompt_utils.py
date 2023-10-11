import json

example_format = [
    {
        "cultural group": "cultural group",
        "context": "context like in public, wedding, at 15 degrees celsius, etc",
        "goal": "the goal of the social interaction",
        "relation": "the social relation between recipient and actor",
        "actor": "the actor of the action",
        "recipient": "the recipient of the action",
        "actor's behavior": "actor's behavior",
        "recipient's behavior": "recipient's behavior",
        "other descriptions": "other descriptions that don't fit",
        "topic": "cultural topic",
    },
    {
        "cultural group": "cultural group",
        "context": "context like in public, wedding, etc",
        "goal": "the goal of the social interaction",
        "relation": "the social relation between recipient and actor",
        "actor": "the actor of the action",
        "recipient": "the recipient of the action",
        "actor's behavior": "actor's behavior",
        "recipient's behavior": "recipient's behavior",
        "other descriptions": "other descriptions that don't fit",
        "topic": "cultural topic",
    },
    ...,
]
ex1_answer = [
    {
        "cultural group": "American",
        "context": "in public",
        "goal": None,
        "relation": None,
        "actor": "people",
        "recipient": None,
        "actor's behavior": "wear at-home clothes",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "clothing",
    },
    {
        "cultural group": "non-American",
        "context": "in public",
        "goal": None,
        "relation": None,
        "actor": "people",
        "recipient": None,
        "actor's behavior": "dress up",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "clothing",
    },
]
ex2_answer = []
ex3_answer = [
    {
        "cultural group": "tourist",
        "context": "",
        "goal": "to prevent sunburn",
        "relation": None,
        "actor": "tourist",
        "recipient": None,
        "actor's behavior": "wear jacket",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "clothing",
    },
]

ANNOTATE_CULTURE_PROMPT = """You are a highly intelligent and accurate information extraction system. You take a comment as input and your task is to extract information in the comment.

You need to output a list of JSON encoded values. {}

-------------------------------
Here are some examples:
Video description: Reply to @thebearsalad the limit does not exist #cultureshock #americansinitaly
Comment: i feel like americans are the only ones who goes out in their at-home clothes, while the rest of the world dresses up just to go to the grocery store
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: Reply to @thebearsalad the limit does not exist #cultureshock #americansinitaly
Comment: cool video
Contain cultural knowledge: No
Output: {}
<EOD>

Video description: POV: It's 20 degrees in Rome
Tourists: 🥵🔥
Italians: 🥶🤧
Italians vs tourists. #romeitalytravel #culturaldifference
Comment: I (tourist) was with jacket because first day in Rome got sunburn
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: {}
Comment: {}
Contain Cultural knowledge: """


def annotate_prompt(video_desc, comment):
    return ANNOTATE_CULTURE_PROMPT.format(
        example_format, json.dumps(ex1_answer), json.dumps(ex2_answer), json.dumps(ex3_answer), video_desc, comment
    )
