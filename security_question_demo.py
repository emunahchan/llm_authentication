# Security Question Demo
#
# This is a security question demo which ask LLM to generate a security question and its model answer
# When a user enter his answer for authentication, the code will ask LLM to grade the acceptance
# by meaning to authenticate the user.
# Cosine similarity between the model answer and the user answer is computed as a second check.
# When when tests are passed, the user is authenticated.
# The demo will return accepted or rejected based on the user's input answer.
#

import os
import random
from flask import Flask, render_template_string, request, redirect, url_for
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util

# Load a lightweight, fast embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Hardcoded threshold
THRESHOLD = 0.7

# Huggingface access token required to use the LLM service
HF_TOKEN = os.getenv("HF_TOKEN")

# Inference client to query LLM on Huggingface
client = InferenceClient(
#    model="meta-llama/Llama-3.1-8B-Instruct",
    model="meta-llama/Llama-3.3-70B-Instruct",
#    model="Qwen/Qwen3-32B", 
    token=HF_TOKEN,
)

# Start web server locally
app = Flask(__name__)

# -------------------------------------------------------------------
# Helper: Generate a new security question + model answer
# -------------------------------------------------------------------
def generate_security_question():
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {
            "role": "user",
            "content": (
                "Generate a security question and its model answer used for user authentication. "
                "Please consider personal questions in the Hong Kong context, like:\n"
                "What is the name of your first school?\n"
                "What is the name of your first pet?\n"
                "Return them in the format:\n"
                "Question: <question>\n"
                "Answer: <answer>"
            ),
        },
    ]

    resp = client.chat_completion(
        messages=messages,
        max_tokens=200,
        temperature=1.0,
    )

    text = resp.choices[0].message.content.strip()

    # Very simple parsing
    q = ""
    a = ""
    for line in text.split("\n"):
        if line.lower().startswith("question:"):
            q = line.split(":", 1)[1].strip()
        if line.lower().startswith("answer:"):
            a = line.split(":", 1)[1].strip()

    return q, a

# -----------------------------------------------------------------------------------
# Helper: Generate a new security question + model answer locally (for testing)
# -----------------------------------------------------------------------------------
# Hardcoded security questions and answers
# Use this if LLM does not have much variation in question generation
# LLM tends to generate more question variety with input information
#
SECURITY_QA_BANK = [
    {
        "question": "What is the name of your first pet?",
        "answer": "Pokemon Pikachu"
    },
    {
        "question": "What is the name of your first school?",
        "answer": "St. Catherine Kindergarten"
    },
    {
        "question": "What is your mother's hometown?",
        "answer": "Changzhou, Jiangsu"
    },
    {
        "question": "What MTR station do you use most often?",
        "answer": "Tsim Sha Tsui MTR Station"
    },
    {
        "question": "What is the name of the hospital where you were born?",
        "answer": "Queen Mary Hospital"
    },
]


def pick_local_security_question():
    item = random.choice(SECURITY_QA_BANK)
    return item["question"], item["answer"]


# -------------------------------------------------------------------
# Helper: Validate user answer using Llama
# -------------------------------------------------------------------
def validate_answer(question, model_answer, user_answer):
    print(question, model_answer, user_answer)
    prompt = (
        f"You are given a security question: {question}\n"
        f"The model answer is: {model_answer}.\n"
        "Please note that you should authenticate based on meaning, not exact answers.\n"
        f"If a user answers: {user_answer}.\n"
        "Should this be accepted? Please answer yes or no only."
    )

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]

    resp = client.chat_completion(
        messages=messages,
        max_tokens=10,
        temperature=0.5,
    )

    answer = resp.choices[0].message.content.strip().lower()
    print(answer)
    return "yes" in answer

# -------------------------------------------------------------------
# Helper: Validate user answer using cosine similarity
# -------------------------------------------------------------------
def validate_similarity(model_answer: str, user_answer: str) -> bool:
    """
    Compute cosine similarity between model_answer and user_answer.
    Return True only if similarity > threshold.
    """

    # Encode both inputs into embeddings
    emb_model = model.encode(model_answer, convert_to_tensor=True)
    emb_user = model.encode(user_answer, convert_to_tensor=True)

    # Compute cosine similarity
    similarity = util.cos_sim(emb_model, emb_user).item()

    # Return acceptance decision
    return similarity > THRESHOLD


# -------------------------------------------------------------------
# HTML Template (inline for demo simplicity)
# -------------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Security Question Demo</title>
</head>
<body>
    <h1>Security Question Demo<h1>
    <h1>for the 28th Hong Kong Youth Science & Technology Innovation Competition</h1>
    <h2>by Emunah Chan <h2>

    <form method="post" action="{{ url_for('generate') }}">
        <button type="submit">Generate New Question</button>
    </form>

    {% if question %}
        <h2>Security Question</h2>
        <p>{{ question }}</p>

        <h3>Model Answer (for demo purposes)</h3>
        <p>{{ model_answer }}</p>


        <form method="post" action="{{ url_for('validate') }}">
            <label>Your Answer:</label><br>
            <input type="text" name="user_answer" required>

            <!-- FIX: Keep original question + model answer -->
            <input type="hidden" name="question" value="{{ question }}">
            <input type="hidden" name="model_answer" value="{{ model_answer }}">
            <button type="submit">Submit</button>
        </form>
    {% endif %}

    {% if result %}
        <h2>Result: {{ result }}</h2>
    {% endif %}
</body>
</html>
"""

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template_string(TEMPLATE, question=None, model_answer=None, result=None)


@app.route("/generate", methods=["POST"])
def generate():
    q, a = generate_security_question()
    # Generate question locally
    # q, a = pick_local_security_question()
    return render_template_string(TEMPLATE, question=q, model_answer=a, result=None)


@app.route("/validate", methods=["POST"])
def validate():
    # In a real app, you'd store these in session or DB.
    # For demo simplicity, they are passed via hidden fields.
    question = request.form.get("question") or request.args.get("question")
    model_answer = request.form.get("model_answer") or request.args.get("model_answer")

    # If not provided, regenerate (demo fallback)
#    if not question or not model_answer:
#        question, model_answer = generate_security_question()
#        question, model_answer = pick_local_security_question()


    print("1:", question, model_answer)
    
    user_answer = request.form["user_answer"]

    print("2:", user_answer)


    accepted_llm = validate_answer(question, model_answer, user_answer)
    accepted_similarity = validate_similarity(model_answer, user_answer)
    result = "Accepted" if (accepted_llm and accepted_similarity) else "Rejected"

    return render_template_string(
        TEMPLATE,
        question=question,
        model_answer=model_answer,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
