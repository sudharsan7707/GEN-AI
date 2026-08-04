from transformers import pipeline
# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
# Input text
text = "The Generative AI workshop was extremely informative and useful."
# Predict sentiment
result = sentiment_analyzer(text)
print(result)
B. Document Classification
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

7

# Classify document
result = classifier(document, labels)
print(result)
