# Parallel NE Pair Extraction and Generation of Parallel Sentence Pairs

Problem Statement: The challenge is to enhance Neural Machine Translation (NMT) systems' capability to accurately translate Named Entities (NEs). NEs are often mistranslated or omitted in NMT outputs, necessitating the creation of a new parallel corpus that prioritizes NE translation accuracy.

Description: To address the aforementioned challenges, the task involves two main objectives:

    Task 1: NE Alignment
    Identify and align NEs between sentence pairs in the provided parallel corpus.
    Example:
    English: "John Doe went to Paris last year."
    Hindi: "जॉन डो पिछले साल पेरिस गया था।"
    Task 2: Parallel Corpus Creation
    Generate a new parallel corpus with a focus on NE translation accuracy, containing sentence pairs where NEs are appropriately translated.
    Example:
    English: "The teacher called Vishal and asked him to solve the problem on the board."
    Hindi: "शिक्षक ने विशाल को बुलाया और उसे बोर्ड पर प्रश्न हल करने को कहा।" (... and so forth, up to a maximum of 5L (500,000) sentence pairs.)

Language Pairs: English-Malayalam

Dataset: BPCC (Link: https://huggingface.co/datasets/ai4bharat/BPCC) parallel corpus.

Overall Input and Output:

    Input: An existing parallel corpus for each language pair.
    Task 1 Output: Aligned NEs for each sentence pair in the evaluation set.
    Task 2 Output: New parallel corpus containing up to 500,000 sentence pairs.

Evaluation Criteria:

    Task 1: Accuracy on the evaluation set.
    Task 2:
        Human Evaluation of a subset of the submitted parallel corpus.
        Quality assessment using LaBSE- or SONAR-based metrics.
        Performance enhancement on a downstream NMT task using the created parallel corpus.

Suggested Techniques: NER tools and classic or neural aligners can be used for Task 1. Rule-based approaches, or LLMs, can be used for sentence pair generation.
