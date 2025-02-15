# Sentiment Analysis using Machine Learning

## Overview
This project focuses on sentiment analysis using a Machine Learning model. The dataset used for training and evaluation consists of 1.6 million tweets, originally collected from Twitter using an API. The primary objective of this project is to classify tweets into sentiment categories such as Positive, Neutral, and Negative.

## Dataset
- **Source**: Kaggle
- **Size**: 1.6 million tweets
- **Collection Method**: Extracted from Twitter using an API
- **Labels**: Tweets are labeled based on their sentiment

## Project Structure
1. **Data Preprocessing**
   - Data cleaning (removing URLs, mentions, special characters, etc.)
   - Tokenization
   - Stopword removal
   - Lemmatization
   
2. **Feature Engineering**
   - TF-IDF Vectorization
   - Word Embeddings (Word2Vec, GloVe, or BERT-based representations)

3. **Model Training**
   - Used machine learning models such as Logistic Regression, Random Forest, and SVM
   - Implemented deep learning models like LSTMs and BERT for better performance
   
4. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-Score
   - Sentiment Score (-10 to 10 scale)

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / PyTorch (for deep learning models)
- NLTK / SpaCy (for text preprocessing)
- Matplotlib / Seaborn (for visualization)

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd sentiment-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Results
- The trained model achieves a high accuracy in classifying tweets based on sentiment.
- The deep learning-based models (BERT, LSTMs) outperform traditional ML models.
- Visualizations provide insights into sentiment distributions and model performance.

## Future Improvements
- Fine-tuning BERT with more data
- Implementing real-time sentiment analysis for Twitter streaming data
- Deploying the model as an API for easy access

## Author
Developed by Harsh Srivastava



