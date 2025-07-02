import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. LOAD MODELS (Done once at startup)
# =============================================================================

# Load VADER
sia = SentimentIntensityAnalyzer()

# Load RoBERTa
MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

print("Models loaded successfully!")

# =============================================================================
# 2. SENTIMENT ANALYSIS FUNCTION
# =============================================================================

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the input text using both VADER and RoBERTa models.
    Returns the analysis results and a comparison plot.
    """
    # --------- VADER Analysis ---------
    vader_scores = sia.polarity_scores(text)
    vader_compound = vader_scores['compound']
    if vader_compound >= 0.05:
        vader_sentiment = "Positive"
    elif vader_compound <= -0.05:
        vader_sentiment = "Negative"
    else:
        vader_sentiment = "Neutral"

    vader_output = (
        f"VADER Sentiment: {vader_sentiment}\n"
        f"---------------------------\n"
        f"Positive: {vader_scores['pos']:.3f}\n"
        f"Neutral:  {vader_scores['neu']:.3f}\n"
        f"Negative: {vader_scores['neg']:.3f}\n"
        f"Compound: {vader_compound:.3f}"
    )

    # --------- RoBERTa Analysis ---------
    encoded_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Map scores to labels
    labels = ['Negative', 'Neutral', 'Positive']
    roberta_scores = {label: float(score) for label, score in zip(labels, scores)}
    
    # Get the sentiment with the highest score
    roberta_sentiment = max(roberta_scores, key=roberta_scores.get)

    roberta_output = (
        f"RoBERTa Sentiment: {roberta_sentiment}\n"
        f"----------------------------\n"
        f"Positive: {roberta_scores.get('Positive', 0):.3f}\n"
        f"Neutral:  {roberta_scores.get('Neutral', 0):.3f}\n"
        f"Negative: {roberta_scores.get('Negative', 0):.3f}"
    )

    # --------- Create Comparison Plot ---------
    df_vader = pd.DataFrame([{'model': 'VADER', 'sentiment': 'Positive', 'score': vader_scores['pos']},
                             {'model': 'VADER', 'sentiment': 'Neutral', 'score': vader_scores['neu']},
                             {'model': 'VADER', 'sentiment': 'Negative', 'score': vader_scores['neg']}])
    
    df_roberta = pd.DataFrame([{'model': 'RoBERTa', 'sentiment': 'Positive', 'score': roberta_scores.get('Positive', 0)},
                               {'model': 'RoBERTa', 'sentiment': 'Neutral', 'score': roberta_scores.get('Neutral', 0)},
                               {'model': 'RoBERTa', 'sentiment': 'Negative', 'score': roberta_scores.get('Negative', 0)}])

    df_combined = pd.concat([df_vader, df_roberta])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_combined, x='sentiment', y='score', hue='model', ax=ax, palette={'VADER': '#1f77b4', 'RoBERTa': '#ff7f0e'})
    ax.set_title('Sentiment Score Comparison')
    ax.set_ylabel('Score')
    ax.set_xlabel('Sentiment')
    plt.tight_layout()

    return vader_output, roberta_output, fig

# =============================================================================
# 3. GRADIO UI INTERFACE
# =============================================================================

# Define the input and output components
input_text = gr.Textbox(lines=5, label="Enter Review Text Here", placeholder="e.g., 'This product is amazing! I would definitely buy it again.'")
output_vader = gr.Textbox(label="VADER Analysis")
output_roberta = gr.Textbox(label="RoBERTa Analysis")
output_plot = gr.Plot(label="Score Comparison")

# Create the Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=input_text,
    outputs=[output_vader, output_roberta, output_plot],
    title="Sentiment Analysis of Amazon Reviews",
    description="Enter a review to analyze its sentiment using two different models: VADER (rule-based) and RoBERTa (a powerful transformer-based model).",
    examples=[
        ["This is the best purchase I have ever made. Highly recommended!"],
        ["It was okay, not great but not terrible either."],
        ["I am very disappointed with this product. It broke after just one use."]
    ],
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    iface.launch()