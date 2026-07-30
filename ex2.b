from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Input document
document = """
Artificial Intelligence and Machine Learning are transforming
industries through automation and intelligent decision-making.
"""

# Candidate labels
labels = ["Technology", "Sports", "Politics", "Entertainment"]

# Classify document
result = classifier(document, labels)

print(result)
