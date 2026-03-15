# llm_authentication
LLM-assisted Authentication Demo

This is the demo code to show how an LLM (Llama-3.3 on Huggingface) can be used in user authentication not requiring exact match. The "what you know" factor like passwords and securit questions requires exact, word for word match of answers for authentication. But this does not align with human communication, which is approxiate and semnatic in nature. This demo explores how the semantic reasoning ability of LLMs can be used to implement a novel "what you know" authentication factor without requiring exact, word-for-word matches.
The demo asks the LLM to generate security questions and model answers. Then the LLM is asked to grade the user answer against the model answer with the given security question. The LLM is asked to authenticate users based on meaning, rather than exact word-for-word matches.
The judgement of the LLM is then combined with a cosine similarity check to decide whether the user authentication is accepted. The user is accepted only when both the LLM accept his/her answer and the consine similarity is larger than a predefined threshold.
Details of the design of the LLM-based authentciation factor can be found at: https://arxiv.org/abs/2601.19684.

To use this code, the Python packes of flask, huggingface_hub, sentence_transformers are required.
