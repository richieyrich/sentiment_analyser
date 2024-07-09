import gradio as gr
from transformers import pipeline

# Load the sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    sentiment = result[0]["label"]
    return sentiment

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analyzer",
    description="Enter a text message to analyze its sentiment.",
)

demo.launch()
