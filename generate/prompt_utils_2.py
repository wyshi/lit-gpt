# prompt_utils.py has too many different versions of prompts, so I'm starting a new (clean) file.

import json



chat_example_format = [
    {
        "cultural group": "group of people with the same cultural background",
        "context": "location, or other settings this behavior is performed",
        "goal": "goal of the behavior",
        "relation": "relation between the actor and recipient",
        "actor": "the actor of the action",
        "recipient": "the recipient of the action",
        "actor's behavior": "the behavior of the actor",
        "recipient's behavior": "the behavior of the recipient",
        "other descriptions": "any other description that doesn't fit into previous categories",
        "topic": "cultural topic",
        "norm": "whether the described event is considered norm according to the given comment. 1 = norm; 0 = taboo",
        "legality": "whether the described event is legal in the given cultural context. 1 = legal; 0 = illegal",
    },
]

chat_ex1_answer = [
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
        "norm": 1,
        "legality": 1,
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
        "norm": 1,
        "legality": 1,
    },
]

chat_ex3_answer = [
    {
        "cultural group": "Japanese",
        "context": "in restaurants",
        "goal": "express gratitude",
        "relation": "customer to service staff",
        "actor": "customer",
        "recipient": "waiter",
        "actor's behavior": "try to tip",
        "recipient's behavior": "return the tip",
        "other descriptions": "tipping is considered rude",
        "topic": "dining etiquette",
        "norm": 0,
        "legality": 1,
    },
]

chat_ex4_answer = [
    {
        "cultural group": "midwestern American",
        "context": None,
        "goal": None,
        "relation": None,
        "actor": "people",
        "recipient": "everyone they meet",
        "actor's behavior": "say hello",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "greeting",
        "norm": 1,
        "legality": 1,
    },
]

chat_ex6_answer = [
    {
        "cultural group": "American",
        "context": "in the United States",
        "goal": None,
        "relation": None,
        "actor": "individuals under 21",
        "recipient": None,
        "actor's behavior": "drink alcohol",
        "recipient's behavior": None,
        "other descriptions": "legal age for drinking is 21",
        "topic": "drinking",
        "norm": 0, 
        "legality": 0,
    },
]

chat_ex7_answer = [
    {
        "cultural group": "Norwegian",
        "context": "summer in Norway",
        "goal": "sleep",
        "relation": None,
        "actor": "residents",
        "recipient": None,
        "actor's behavior": "struggle to sleep",
        "recipient's behavior": None,
        "other descriptions": "sun doesn't set",
        "topic": "natural phenomena",
        "norm": 1,
        "legality": 1,
    },
]

CHAT_PROMPT = """[INST] <<SYS>>
You are a helpful, respectful and intelligent assistant trained to identify and extract cultural information. Your role is to follow the given instructions precisely and format your responses as required. Keep your responses succinct and limited to the requested information. If you don't know the answer to a question, please don't share false information.

Cultural information encompasses content that showcases the distinctive characteristics, artifacts, or manifestations of a specific group, community, or region. This includes, but is not limited to, practices, behaviors, norms, values, beliefs, habits, customs, architectural styles, environmental engagements, and any other elements that are emblematic of a particular cultural setting. It does not include generic information or widespread practices that are not distinctly tied to a specific cultural identity.

For this task, consider information as "cultural" if:

1. It is associated with or characteristic of a specific identified group (e.g., Americans, Italians, midwestern Americans, etc.).
2. It reveals a unique aspect of that group's way of life, including social conventions, physical creations, or interactions with their surroundings that are not typically seen in other cultures.
3. It provides insight into the cultural uniqueness, whether through social practices, material culture, or other culturally significant elements.

Please exclude generic or ubiquitous statements or observations that do not clearly relate to the unique cultural context of a specific group.
<</SYS>>
For each video-comment pair, you need to do two things:
1. Determine whether the provided example contains cultural information.
2. If the example does include cultural information, extract the cultural knowledge into a list of JSON objects with the following fields:
```
{}
```
If an example contains multiple cultural knowledge, please encode each piece of knowledge into a seperate JSON object.
Output the extracted cultural knowledge as a list of JSON objects, or an empty list if the provided example does not contain any cultural information.

----------------------------------------------------
Here are some examples:

Video description: Reply to @thebearsalad the limit does not exist
Comment: "i feel like americans are the only ones who goes out in their at-home clothes, while the rest of the world dresses up just to go to the grocery store"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: Reply to @thebearsalad the limit does not exist
Comment: "cool video"
Contain cultural knowledge: No
Output: []
<EOD>

Video description: Exploring Tokyo!
Comment: "Tried to tip at a restaurant in Tokyo, the waiter chased us down to return it. Tipping is actually rude here, who knew?"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description:
Comment: "As a midwestern American I literally say hello to everyone I meet. \ud83d\udc4b"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: Baseball edition! \u26be\ufe0f (IB: @kacierose4)
Comment: "this is why ion really like baseball games and i watch basketball"
Contain cultural knowledge: No
Output: []
<EOD>

Video description: Celebrating 21st birthdays in the USA! 🎉
Comment: "It still blows my mind that in the US you can join the army before you can legally buy a beer."
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: Midnight Sun in Norway ☀️
Comment: "Can't sleep at all during the summer, the sun never sets here!"
Contain cultural knowledge: Yes
Output: {}

----------------------------------------------------
Now determine if the following example contains cultural information and extract any cultrual knowledge into a list of JSON objects. Please only include information that you directly extract from the provided text and do not hallucinate.

[Reminder]: Consider information as "cultural" if:
1. It pertains to a specific identified group (e.g., Americans, Italians).
2. It shows unique cultural traits or practices of that group differing from others.
3. It provides insight into the cultural uniqueness, whether through social practices, material culture, or other culturally significant elements.
Please avoid considering generic statements or behaviors that are common across multiple cultures or lack specificity as "cultural information."

Please base your answers strictly on the provided text. If important cultural context, such as the cultural group, is not explicitly mentioned or directly inferable from the text, output an empty list. Avoid adding or assuming any information that is not directly supported by the text.
Once you've outputed a list of JSON objects, please immediately output "<EOD>".

Video description: {}
Comment: {}
Contain cultural knowledge: [/INST]
"""

def annotate_chat_prompt(video_desc, comment):
    # print(video_desc, comment)
    return CHAT_PROMPT.format(
        json.dumps(chat_example_format, indent=4), json.dumps(chat_ex1_answer, indent=4), json.dumps(chat_ex3_answer, indent=4), json.dumps(chat_ex4_answer, indent=4),\
        json.dumps(chat_ex6_answer, indent=4), json.dumps(chat_ex7_answer, indent=4), video_desc, comment
    )