from transformers import pipeline

# Load the question-answering pipeline
question_answerer = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# Input context
context = """
Generative Artificial Intelligence is a type of Artificial Intelligence
that can create new content such as text, images, audio, video, and
computer programs. Large Language Models are commonly used for text
generation, summarization, translation, and question answering.
"""

# Input question
question = "What type of content can Generative AI create?"

# Predict the answer
result = question_answerer(
    question=question,
    context=context
)

# Display the result
print("Answer:", result["answer"])
print("Confidence Score:", round(result["score"], 3))
