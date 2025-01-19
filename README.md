# Stock Sentiment Analysis Project

This Jupyter Notebook (`Stock_Sentiment_Analysis_final.ipynb`) is designed to analyze stock sentiment from textual data and assess its impact on stock price movements. The project leverages natural language processing (NLP) techniques and machine learning algorithms to classify sentiment and predict trends in stock markets.

## Project Objectives

1. **Sentiment Analysis**: Analyze news headlines, tweets, or other financial text data to determine sentiment (positive, negative, or neutral).
2. **Feature Engineering**: Extract relevant features from text using NLP techniques like tokenization, TF-IDF, or word embeddings.
3. **Model Training and Evaluation**: Train machine learning models to classify sentiment and evaluate performance using metrics like accuracy, precision, recall, and F1-score.
4. **Correlation Analysis**: Investigate the relationship between sentiment scores and stock price movements.
5. **Visualization**: Create insightful visualizations to display sentiment trends and their impact on stocks.

## File Overview

### `Stock_Sentiment_Analysis_final.ipynb`

This notebook contains the following sections:

1. **Introduction**:
   - Overview of the dataset and project goals.
   - Description of data sources (e.g., news headlines, stock price data).

2. **Data Loading**:
   - Imports and loads the dataset(s) required for analysis.
   - Handles missing data and other preprocessing steps.

3. **Data Preprocessing**:
   - Cleans the textual data by removing stopwords, punctuation, and special characters.
   - Tokenizes and stems/lemmatizes text to prepare it for analysis.
   - Encodes labels for sentiment classification (e.g., positive, negative, neutral).

4. **Feature Engineering**:
   - Converts text data into numerical representations using techniques such as:
     - Bag of Words (BOW)
     - TF-IDF (Term Frequency-Inverse Document Frequency)
     - Word Embeddings (e.g., Word2Vec, GloVe)

5. **Exploratory Data Analysis (EDA)**:
   - Visualizes the distribution of sentiment categories.
   - Analyzes word frequency and key terms contributing to sentiment classification.
   - Displays trends in stock prices corresponding to sentiment changes.

6. **Model Training**:
   - Trains machine learning models (e.g., Logistic Regression, Naive Bayes, Random Forest, or Neural Networks) to classify sentiment.
   - Optimizes hyperparameters using techniques like Grid Search or Random Search.

7. **Model Evaluation**:
   - Evaluates model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
   - Plots confusion matrices and ROC curves for model comparison.

8. **Stock Trend Analysis**:
   - Analyzes the correlation between sentiment scores and stock price changes.
   - Evaluates whether sentiment can predict market movements effectively.

9. **Visualizations**:
   - Sentiment trend over time.
   - Stock price movement compared to sentiment trends.

10. **Conclusion and Future Work**:
    - Summarizes findings and model performance.
    - Discusses limitations and potential improvements, such as using advanced models like transformers or integrating additional datasets.

## Metrics Table

| **Model**                  | **Accuracy** | **Precision** | **Recall** | **Confusion Matrix**           |
|----------------------------|--------------|---------------|------------|---------------------------------|
| **Logistic Regression**    | 85.98%       | 0.87          | 0.85       | `[[162,  24], [ 29, 163]]`     |
| **Random Forest Classifier** | 84.92%      | 0.84          | 0.88       | `[[153,  33], [ 24, 168]]`     |
| **Multinomial Naive Bayes** | 83.86%      | 0.85          | 0.83       | `[[158,  28], [ 33, 159]]`     |


## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Libraries used in the notebook:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `nltk`
  - `wordcloud`
  - Any other libraries specified in the notebook.

### Installation

1. Clone this repository or download the notebook file.
   ```bash
   git clone <repository-url>
   cd <repository-folder>

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Stock_Sentiment_Analysis_final.ipynb` and execute the cells sequentially.

## Results

- Provides sentiment classification metrics for financial text.
- Analyzes how sentiment scores align with stock price movements.
- Visualizes sentiment trends and their correlation with market fluctuations.

## Future Enhancements

- Integrate additional text sources (e.g., social media, earnings reports).
- Use advanced deep learning models like BERT or GPT for sentiment analysis.
- Incorporate real-time stock price data and live sentiment analysis.
- Implement a dashboard to visualize results dynamically.