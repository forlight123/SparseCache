# SPDX-License-Identifier: Apache-2.0
"""Public LongBench prompt templates and generation limits."""

LONG_BENCH_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question asconcisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\nNow, answer the question based on the story "
        "asconcisely as you can, using a single phrase if possible. Do not provide "
        "any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write "
        '"unanswerable". If the question is a yes/no question, answer "yes", "no", or '
        '"unanswerable". Do not provide any explanation.\n\nArticle: '
        "{context}\n\n Answer the question based on the above article as concisely "
        "as you can, using a single phrase or sentence if possible. If the "
        "question cannot be answered based on the information in the article, "
        'write "unanswerable". If the question is a yes/no question, answer '
        '"yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, "
        "answer the following question based on the above text, only give me "
        "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page "
        "summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question "
        "or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\nNow, answer the query based on the above "
        "meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all "
        "news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all "
        "the news.\n\nSummary:"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them "
        "may be duplicates. Please carefully read these paragraphs and determine "
        "how many unique paragraphs there are after removing duplicates. In other "
        "words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\nPlease enter the final count of unique paragraphs after "
        "removing duplicates. The output format should only contain the number, "
        "such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine "
        "which paragraph the abstract is from.\n\n{context}\n\nThe following "
        "is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph "
        "that the abstract is from. The answer format must be like "
        '"Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line of "
        "code:\n"
    ),
}


LONG_BENCH_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


LONG_BENCH_V2_PROMPT = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)"."""

LONG_BENCH_V2_MAX_NEW_TOKENS = 128
