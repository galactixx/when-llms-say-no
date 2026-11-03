"""Prompt templates and helpers for all 9 zero-shot AV methods."""

from __future__ import annotations

COT_TEMPLATE = """Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring. Let's think step by step.

Text 1: {text1}
Text 2: {text2}
"""

PROMPTAV_TEMPLATE = """Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring. Here are some relevant variables to this problem.
1. punctuation style(e.g. hyphen, brackets, colon, comma, parenthesis, quotation mark)
2. special characters style, capitalization style(e.g. Continuous capitalization, capitalizing certain words)
3. acronyms and abbreviations(e.g. Usage of acronyms such as OMG, Abbreviations without punctuation marks such as Mr Rochester vs. Mr. Rochester,Unusual abbreviations such as def vs. definitely)
4. writing style
5. expressions and Idioms
6. tone and mood
7. sentence structure
8. any other relevant aspect
First step: Understand the problem, extracting relevant variables and devise a plan to solve the problem. Then, carry out the plan and solve the problem step by step. Finally, show the confidence score.

Text 1: {text1}
Text 2: {text2}
"""

PROMPTAV_FRAMING_ONLY_TEMPLATE = """Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring.
Do not focus on what the texts are about. Focus on recurring surface-form habits that could remain stable across topics. Here are some relevant variables to this problem.
1. punctuation style(e.g. hyphen, brackets, colon, comma, parenthesis, quotation mark)
2. special characters style, capitalization style(e.g. Continuous capitalization, capitalizing certain words)
3. acronyms and abbreviations(e.g. Usage of acronyms such as OMG, Abbreviations without punctuation marks such as Mr Rochester vs. Mr. Rochester,Unusual abbreviations such as def vs. definitely)
4. writing style
5. expressions and Idioms
6. tone and mood
7. sentence structure
8. any other relevant aspect
First step: Understand the problem, extracting relevant variables and devise a plan to solve the problem. Then, carry out the plan and solve the problem step by step. Finally, show the confidence score.

Text 1: {text1}
Text 2: {text2}
"""

PROMPTAV_CONTRASTIVE_ONLY_TEMPLATE = """Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring. Here are some relevant variables to this problem.
1. punctuation style(e.g. hyphen, brackets, colon, comma, parenthesis, quotation mark)
2. special characters style, capitalization style(e.g. Continuous capitalization, capitalizing certain words)
3. acronyms and abbreviations(e.g. Usage of acronyms such as OMG, Abbreviations without punctuation marks such as Mr Rochester vs. Mr. Rochester,Unusual abbreviations such as def vs. definitely)
4. writing style
5. expressions and Idioms
6. tone and mood
7. sentence structure
8. any other relevant aspect
When comparing the texts, explicitly separate low-level stylometric evidence from content-sensitive evidence, and give more weight to low-level stylometric evidence.
First step: Understand the problem, extracting relevant variables and devise a plan to solve the problem. Then, carry out the plan and solve the problem step by step. Finally, show the confidence score.

Text 1: {text1}
Text 2: {text2}
"""

PROMPTAV_FRAMED_CONTRASTIVE_TEMPLATE = """Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring.
Do not focus on what the texts are about. Focus on recurring surface-form habits that could remain stable across topics. Here are some relevant variables to this problem.
1. punctuation style(e.g. hyphen, brackets, colon, comma, parenthesis, quotation mark)
2. special characters style, capitalization style(e.g. Continuous capitalization, capitalizing certain words)
3. acronyms and abbreviations(e.g. Usage of acronyms such as OMG, Abbreviations without punctuation marks such as Mr Rochester vs. Mr. Rochester,Unusual abbreviations such as def vs. definitely)
4. writing style
5. expressions and Idioms
6. tone and mood
7. sentence structure
8. any other relevant aspect
When comparing the texts, explicitly separate low-level stylometric evidence from content-sensitive evidence, and give more weight to low-level stylometric evidence.
First step: Understand the problem, extracting relevant variables and devise a plan to solve the problem. Then, carry out the plan and solve the problem step by step. Finally, show the confidence score.

Text 1: {text1}
Text 2: {text2}
"""

PROMPT_CAVE_TEMPLATE = """Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring. Here are some relevant variables to this problem.
1. punctuation style (e.g. hyphen, brackets, colon, comma, parenthesis, quotation mark)
2. special characters style, capitalization style (e.g. Continuous capitalization, capitalizing certain words)
3. acronyms and abbreviations(e.g. Usage of acronyms such as OMG, Abbreviations without punctuation marks such as Mr Rochester vs. Mr. Rochester,Unusual abbreviations such as def vs. definitely)
4. writing style
5. expressions and Idioms
6. tone and mood
7. sentence structure
8. any other relevant aspect
Provide the answer in a Python JSON format. Use the following keys for your dictionary: 'punctuation style', 'special characters style, capitalization style', 'acronyms and abbreviations', 'writing style', 'expressions and Idioms', 'tone and mood', 'sentence structure', 'any other relevant aspect', 'final score'. Apart from the 'final score', everything else must have a text value; also, the text should include a concluding YES/NO/MAYBE about whether the two texts are similar or not with respect to the key at hand. Finally, provide an 'output' key in your dictionary, which says YES if the two texts are written by the same author, and NO otherwise.

Text 1: {text1}
Text 2: {text2}
"""

PROMPT_CAVE_FRAMING_ONLY_TEMPLATE = """Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring.
Do not focus on what the texts are about. Focus on recurring surface-form habits that could remain stable across topics. Here are some relevant variables to this problem.
1. punctuation style (e.g. hyphen, brackets, colon, comma, parenthesis, quotation mark)
2. special characters style, capitalization style (e.g. Continuous capitalization, capitalizing certain words)
3. acronyms and abbreviations(e.g. Usage of acronyms such as OMG, Abbreviations without punctuation marks such as Mr Rochester vs. Mr. Rochester,Unusual abbreviations such as def vs. definitely)
4. writing style
5. expressions and Idioms
6. tone and mood
7. sentence structure
8. any other relevant aspect
Provide the answer in a Python JSON format. Use the following keys for your dictionary: 'punctuation style', 'special characters style, capitalization style', 'acronyms and abbreviations', 'writing style', 'expressions and Idioms', 'tone and mood', 'sentence structure', 'any other relevant aspect', 'final score'. Apart from the 'final score', everything else must have a text value; also, the text should include a concluding YES/NO/MAYBE about whether the two texts are similar or not with respect to the key at hand. Finally, provide an 'output' key in your dictionary, which says YES if the two texts are written by the same author, and NO otherwise.

Text 1: {text1}
Text 2: {text2}
"""

PROMPT_CAVE_CONTRASTIVE_ONLY_TEMPLATE = """Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring. Here are some relevant variables to this problem.
1. punctuation style (e.g. hyphen, brackets, colon, comma, parenthesis, quotation mark)
2. special characters style, capitalization style (e.g. Continuous capitalization, capitalizing certain words)
3. acronyms and abbreviations(e.g. Usage of acronyms such as OMG, Abbreviations without punctuation marks such as Mr Rochester vs. Mr. Rochester,Unusual abbreviations such as def vs. definitely)
4. writing style
5. expressions and Idioms
6. tone and mood
7. sentence structure
8. any other relevant aspect
When comparing the texts, explicitly separate low-level stylometric evidence from content-sensitive evidence, and give more weight to low-level stylometric evidence.
Provide the answer in a Python JSON format. Use the following keys for your dictionary: 'punctuation style', 'special characters style, capitalization style', 'acronyms and abbreviations', 'writing style', 'expressions and Idioms', 'tone and mood', 'sentence structure', 'any other relevant aspect', 'final score'. Apart from the 'final score', everything else must have a text value; also, the text should include a concluding YES/NO/MAYBE about whether the two texts are similar or not with respect to the key at hand. Finally, provide an 'output' key in your dictionary, which says YES if the two texts are written by the same author, and NO otherwise.

Text 1: {text1}
Text 2: {text2}
"""

PROMPT_CAVE_FRAMED_CONTRASTIVE_TEMPLATE = """Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring.
Do not focus on what the texts are about. Focus on recurring surface-form habits that could remain stable across topics. Here are some relevant variables to this problem.
1. punctuation style (e.g. hyphen, brackets, colon, comma, parenthesis, quotation mark)
2. special characters style, capitalization style (e.g. Continuous capitalization, capitalizing certain words)
3. acronyms and abbreviations(e.g. Usage of acronyms such as OMG, Abbreviations without punctuation marks such as Mr Rochester vs. Mr. Rochester,Unusual abbreviations such as def vs. definitely)
4. writing style
5. expressions and Idioms
6. tone and mood
7. sentence structure
8. any other relevant aspect
When comparing the texts, explicitly separate low-level stylometric evidence from content-sensitive evidence, and give more weight to low-level stylometric evidence.
Provide the answer in a Python JSON format. Use the following keys for your dictionary: 'punctuation style', 'special characters style, capitalization style', 'acronyms and abbreviations', 'writing style', 'expressions and Idioms', 'tone and mood', 'sentence structure', 'any other relevant aspect', 'final score'. Apart from the 'final score', everything else must have a text value; also, the text should include a concluding YES/NO/MAYBE about whether the two texts are similar or not with respect to the key at hand. Finally, provide an 'output' key in your dictionary, which says YES if the two texts are written by the same author, and NO otherwise.

Text 1: {text1}
Text 2: {text2}
"""

_TEMPLATE_MAP = {
    "cot": COT_TEMPLATE,
    "promptav": PROMPTAV_TEMPLATE,
    "promptav_framing_only": PROMPTAV_FRAMING_ONLY_TEMPLATE,
    "promptav_contrastive_only": PROMPTAV_CONTRASTIVE_ONLY_TEMPLATE,
    "promptav_framed_contrastive": PROMPTAV_FRAMED_CONTRASTIVE_TEMPLATE,
    "prompt_cave": PROMPT_CAVE_TEMPLATE,
    "prompt_cave_framing_only": PROMPT_CAVE_FRAMING_ONLY_TEMPLATE,
    "prompt_cave_contrastive_only": PROMPT_CAVE_CONTRASTIVE_ONLY_TEMPLATE,
    "prompt_cave_framed_contrastive": PROMPT_CAVE_FRAMED_CONTRASTIVE_TEMPLATE,
}


def build_prompt(method: str, text1: str, text2: str) -> str:
    template = _TEMPLATE_MAP.get(method)
    if template is None:
        raise ValueError(f"Unknown method: {method}")
    return template.format(text1=text1, text2=text2)


def make_chat_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
    ]
