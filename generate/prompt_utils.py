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
    },
    ...,
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

chat_ex3_answer = [
    {
        "cultural group": "non-American",
        "context": "in the USA",
        "goal": None,
        "relation": None,
        "actor": "people",
        "recipient": "an American",
        "actor's behavior": "asked for a spoggie",
        "recipient's behavior": "looked confused",
        "other descriptions": None,
        "topic": "language",
    },
    {
        "cultural group": "American",
        "context": "in the USA",
        "goal": None,
        "relation": None,
        "actor": "people",
        "recipient": None,
        "actor's behavior": "do not understand spoggie",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "language",
    },
]

chat_ex4_answer = [
    {
        "cultural group": "midwestern American",
        "context": None,
        "goal": None,
        "relation": None,
        "actor": "people",
        "recipient": "everyone",
        "actor's behavior": "say hello",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "greeting",
    },
]

CHAT_PROMPT = """[INST] <<SYS>>
You are a helpful, respectful and intelligent assistant. You should always follow exactly the instructions provided to you and format your answers exactly as required. Please keep your responses concise and provide no additional output other than the information required.
If you don't know the answer to a question, please don't share false information.
<</SYS>>
Cultural information refers to content that highlights practices, behaviors, norms, values, beliefs, habits, or any customs specific to a particular group, community, or region. This can encompass everyday habits, social etiquettes, traditional rituals, clothing preferences, greeting methods, and more. It doesn't include generic information or practices that are widespread and don't pertain to any specific cultural group.

For this task, consider information as "cultural" if:
1. It pertains to a specific identified group (e.g., Americans, Italians, midwestern Americans, etc.).
2. It highlights a distinct behavior, norm, or custom of that group which might be different from other groups.
3. It is related to social interactions, events, or specific situations where cultural norms are evident.

For each video-comment pair, you need to do two things:
1. Determine whether the provided example contains cultural information.
2. If the example does include cultural information, extract the cultural knowledge into JSON objects with the following fields:
```
{}
```
If an example contains multiple cultural knowledge, please encode each piece of knowledge into a seperate JSON object.
Output the extracted cultural knowledge as a list of JSON objects, or an empty list if the provided example does not contain any cultural information.

----------------------------------------------------
Here are some examples:

Video description: Reply to @thebearsalad the limit does not exist #cultureshock #americansinitaly
Comment: "i feel like americans are the only ones who goes out in their at-home clothes, while the rest of the world dresses up just to go to the grocery store"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: Reply to @thebearsalad the limit does not exist #cultureshock #americansinitaly
Comment: "cool video"
Contain cultural knowledge: No
Output: []
<EOD>

Video description: American Diner Edition (IB: @kacierose4)  #cultureshock #culturedifferences #britsintheusa #ukvsusa #culturaldifference
Comment: "I remember asking an American for a spoggie once, they looked so confused \ud83d\ude02
(spoggie = chewing gum)"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: #sayhitoastranger #friendly #strangers #countryside #germanlife #irishlife #irishcountryside #germancountryside #german #irish #countrysidelife #wave
Comment: "As a midwestern American I literally say hello to everyone I meet. \ud83d\udc4b"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: Baseball edition! \u26be\ufe0f (IB: @kacierose4) #cultureshock #cultureshocks #culturaldifference #ukvsusa #baseball
Comment: "this is why ion really like baseball games and i watch basketball"
Contain cultural knowledge: No
Output: []
<EOD>

----------------------------------------------------
Now determine if the following example contains cultural information and extract any cultrual knowledge into JSON objects. Please only include information that you directly extract from the provided text and do not hallucinate.

[Reminder]: Consider information as "cultural" if and only if:
1. It pertains to a specific identified group (e.g., Americans, Italians).
2. It highlights a distinct behavior or norm of that group differing from others.
3. It's about social interactions, events, or situations showcasing cultural norms.
Please avoid considering generic statements or behaviors that are common across multiple cultures or lack specificity as "cultural information."

Once you've outputed the JSON objects, please immediately output "<EOD>".

Video description: {}
Comment: {}
Contain cultural knowledge: [/INST]
"""

def annotate_chat_prompt(video_desc, comment):
    # print(video_desc, comment)
    return CHAT_PROMPT.format(
        chat_example_format, json.dumps(chat_ex1_answer, indent=4), json.dumps(chat_ex3_answer, indent=4), json.dumps(chat_ex4_answer, indent=4), video_desc, comment
    )
    

    
    
    
    
chat2_ex3_answer = [
    {
        "cultural group": "non-Italian",
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
    {
        "cultural group": "Italian",
        "context": "",
        "goal": None,
        "relation": None,
        "actor": "people",
        "recipient": None,
        "actor's behavior": "feels cold",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "clothing",
    },
]
    
    
    
    

CHAT_PROMPT_2 = """[INST] <<SYS>>
You are a helpful, respectful and intelligent assistant. You should always follow exactly the instructions provided to you and format your answers exactly as required. Please keep your responses concise and provide no additional output other than the information required.
If you don't know the answer to a question, please don't share false information.
<</SYS>>
Cultural information refers to content that highlights practices, behaviors, norms, values, beliefs, habits, or any customs specific to a particular group, community, or region. This can encompass everyday habits, social etiquettes, traditional rituals, clothing preferences, greeting methods, and more. It doesn't include generic information or practices that are widespread and don't pertain to any specific cultural group.

For this task, consider information as "cultural" if:
1. It pertains to a specific identified group (e.g., Americans, Italians, midwestern Americans, etc.).
2. It highlights a distinct behavior, norm, or custom of that group which might be different from other groups.
3. It is related to social interactions, events, or specific situations where cultural norms are evident.

For each video-comment pair, you need to do two things:
1. Determine whether the provided example contains cultural information.
2. If the example does include cultural information, extract the cultural knowledge into JSON objects with the following fields:
```
{}
```
If an example contains multiple cultural knowledge, please encode each piece of knowledge into a seperate JSON object.
Output the extracted cultural knowledge as a list of JSON objects, or an empty list if the provided example does not contain any cultural information.

----------------------------------------------------
Here are some examples:

Video description: Reply to @thebearsalad the limit does not exist #cultureshock #americansinitaly
Comment: "i feel like americans are the only ones who goes out in their at-home clothes, while the rest of the world dresses up just to go to the grocery store"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: Reply to @thebearsalad the limit does not exist #cultureshock #americansinitaly
Comment: "cool video"
Contain cultural knowledge: No
Output: []
<EOD>

Video description: POV: It's 20 degrees in Rome
Tourists: 🥵🔥
Italians: 🥶🤧
Italians vs tourists. #romeitalytravel #culturaldifference
Comment: "I (tourist) was with jacket because first day in Rome got sunburn"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: #sayhitoastranger #friendly #strangers #countryside #germanlife #irishlife #irishcountryside #germancountryside #german #irish #countrysidelife #wave
Comment: "As a midwestern American I literally say hello to everyone I meet. \ud83d\udc4b"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Video description: Baseball edition! \u26be\ufe0f (IB: @kacierose4) #cultureshock #cultureshocks #culturaldifference #ukvsusa #baseball
Comment: "this is why ion really like baseball games and i watch basketball"
Contain cultural knowledge: No
Output: []
<EOD>

----------------------------------------------------
Now determine if the following example contains cultural information and extract any cultrual knowledge into JSON objects. Please only include information that you directly extract from the provided text and do not hallucinate.

[Reminder]: Consider information as "cultural" if and only if:
1. It pertains to a specific identified group (e.g., Americans, Italians).
2. It highlights a distinct behavior or norm of that group differing from others.
3. It's about social interactions, events, or situations showcasing cultural norms.
Please avoid considering generic statements or behaviors that are common across multiple cultures or lack specificity as "cultural information."

Once you've outputed the JSON objects, please immediately output "<EOD>".

Video description: {}
Comment: {}
Contain cultural knowledge: [/INST]
"""

def annotate_chat_prompt2(video_desc, comment):
    # print(video_desc, comment)
    return CHAT_PROMPT_2.format(
        chat_example_format, json.dumps(chat_ex1_answer, indent=4), json.dumps(chat2_ex3_answer, indent=4), json.dumps(chat_ex4_answer, indent=4), video_desc, comment
    )






chat2_comment_ex3_answer = [
    {
        "cultural group": "non-Italian",
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



CHAT_PROMPT_COMMENTS_ONLY = """[INST] <<SYS>>
You are a helpful, respectful and intelligent assistant. You should always follow exactly the instructions provided to you and format your answers exactly as required. Please keep your responses concise and provide no additional output other than the information required.
If you don't know the answer to a question, please don't share false information.
<</SYS>>
Cultural information refers to content that highlights practices, behaviors, norms, values, beliefs, habits, or any customs specific to a particular group, community, or region. This can encompass everyday habits, social etiquettes, traditional rituals, clothing preferences, greeting methods, and more. It doesn't include generic information or practices that are widespread and don't pertain to any specific cultural group.

For this task, consider information as "cultural" if:
1. It pertains to a specific identified group (e.g., Americans, Italians, midwestern Americans, etc.).
2. It highlights a distinct behavior, norm, or custom of that group which might be different from other groups.
3. It is related to social interactions, events, or specific situations where cultural norms are evident.

For each comment below, you need to do two things:
1. Determine whether the provided example contains cultural information.
2. If the example does include cultural information, extract the cultural knowledge into JSON objects with the following fields:
```
{}
```
If an example contains multiple cultural knowledge, please encode each piece of knowledge into a seperate JSON object.
Output the extracted cultural knowledge as a list of JSON objects, or an empty list if the provided example does not contain any cultural information.

----------------------------------------------------
Here are some examples:

Comment: "i feel like americans are the only ones who goes out in their at-home clothes, while the rest of the world dresses up just to go to the grocery store"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Comment: "cool video"
Contain cultural knowledge: No
Output: []
<EOD>

Comment: "I (tourist) was with jacket because first day in Rome got sunburn"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Comment: "As a midwestern American I literally say hello to everyone I meet. \ud83d\udc4b"
Contain cultural knowledge: Yes
Output: {}
<EOD>

Comment: "this is why ion really like baseball games and i watch basketball"
Contain cultural knowledge: No
Output: []
<EOD>

----------------------------------------------------
Now determine if the following example contains cultural information and extract any cultrual knowledge into a list of JSON objects. Please only include information that you directly extract from the provided text and do not hallucinate.

[Reminder]: Consider information as "cultural" if and only if:
1. It pertains to a specific identified group (e.g., Americans, Italians).
2. It highlights a distinct behavior or norm of that group differing from others.
3. It's about social interactions, events, or situations showcasing cultural norms.
Please avoid considering generic statements or behaviors that are common across multiple cultures or lack specificity as "cultural information."

If the given comment lacks important context or is not obvious whether it contains cultural information, answer "No" and leave the output an empty list.
Please do not make assumptions about the cultural group if it is not explicitly mentioned in the comment. If a specific cultural group is not identified, instead of assuming that the cultural group is "American", you should answer 'No' and provide an empty list as output.

Once you've outputed the JSON objects, please immediately output "<EOD>".

Comment: {}
Contain cultural knowledge: [/INST]
"""

def annotate_chat_prompt_comment_only(comment):
    # print(video_desc, comment)
    return CHAT_PROMPT_COMMENTS_ONLY.format(
        chat_example_format, json.dumps(chat_ex1_answer, indent=4), json.dumps(chat2_comment_ex3_answer, indent=4), json.dumps(chat_ex4_answer, indent=4), comment
    )
    
    






CHAT_PROMPT_CULTURE_RELATED = """[INST] <<SYS>>
You are a helpful, respectful and intelligent assistant. You should always follow exactly the instructions provided to you and format your answers exactly as required. Please keep your responses concise and provide no additional output other than the information required.
If you don't know the answer to a question, please don't share false information.
<</SYS>>
Cultural information refers to content that highlights practices, behaviors, norms, values, beliefs, habits, or any customs specific to a particular group, community, or region. This can encompass everyday habits, social etiquettes, traditional rituals, clothing preferences, greeting methods, and more. It doesn't include generic information or practices that are widespread and don't pertain to any specific cultural group.

For this task, consider information as "cultural" if:
1. It pertains to a specific identified group (e.g., Americans, Italians, midwestern Americans, etc.).
2. It highlights a distinct behavior, norm, or custom of that group which might be different from other groups.
3. It is related to social interactions, events, or specific situations where cultural norms are evident.

For each comment below, determine if it contains cultural information. Output a single token Yes/No answer.

----------------------------------------------------
Here are some examples:

Comment: "i feel like americans are the only ones who goes out in their at-home clothes, while the rest of the world dresses up just to go to the grocery store"
Contain cultural knowledge (Yes/No): Yes
<EOD>

Comment: "cool video"
Contain cultural knowledge (Yes/No): No
<EOD>

Comment: "I (tourist) was with jacket because first day in Rome got sunburn"
Contain cultural knowledge (Yes/No): Yes
<EOD>

Comment: "I remember asking an American for a spoggie once, they looked so confused \ud83d\ude02"
(spoggie = chewing gum)"
Contain cultural knowledge (Yes/No): Yes

Comment: "As a midwestern American I literally say hello to everyone I meet. \ud83d\udc4b"
Contain cultural knowledge (Yes/No): Yes
<EOD>

Comment: "this is why ion really like baseball games and i watch basketball"
Contain cultural knowledge (Yes/No): No
<EOD>

----------------------------------------------------
Now determine if the following example contains cultural information.

[Reminder]: Consider information as "cultural" if and only if:
1. It pertains to a specific identified group (e.g., Americans, Italians).
2. It highlights a distinct behavior or norm of that group differing from others.
3. It's about social interactions, events, or situations showcasing cultural norms.
Please avoid considering generic statements or behaviors that are common across multiple cultures or lack specificity as "cultural information."

If the given comment lacks important context or you cannot make a decision, answer "No". If the given comment doesn't specify a specific cultural group, please answer "No" and do not hallucinate or share false information.

Please output a single-token Yes/No answer followed immediately by <EOD>.

Comment: {}
Contain cultural knowledge (Yes/No): [/INST]
"""

def annotate_chat_prompt_culture_related(comment):
    # print(video_desc, comment)
    return CHAT_PROMPT_CULTURE_RELATED.format(comment)