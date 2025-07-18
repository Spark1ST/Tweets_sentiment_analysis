# Tweets Sentiment Analysis

This project analyzes the sentiment of tweets using both rule-based (VADER) and deep learning models (RNN, LSTM, GRU). It demonstrates data preprocessing, sentiment classification, model training,hyperparameter tuning and evaluation.

## Features

- **Data Cleaning:** Removes URLs, special characters, stopwords, and applies lemmatization.
- **VADER Sentiment Analysis:** Uses NLTK's VADER for rule-based sentiment scoring.
- **Deep Learning Models:** Implements RNN, LSTM, and GRU using TensorFlow/Keras for sentiment classification.
- **Hyperparameter Tuning:** Tests different model configurations for best accuracy.
- **Visualization:** Plots confusion matrices and training history.
- **Failure Analysis:** Highlights common cases where VADER misclassifies sentiment.

## Requirements

- Python 3.7+
- pandas
- numpy
- nltk
- scikit-learn
- seaborn
- matplotlib
- tensorflow

Install dependencies:
```sh
pip install pandas numpy nltk scikit-learn seaborn matplotlib tensorflow
```

## Usage

1. Place your `Tweets.csv` file in the project directory.
2. Run the notebook:  
   - Download required NLTK resources.
   - Preprocess the data.
   - Perform sentiment analysis with VADER.
   - Train and evaluate RNN, LSTM, and GRU models.
   - Visualize results.

## File Structure

- `Tweets sentiment analysis.ipynb` – Main Jupyter notebook for analysis and modeling.
- `Tweets.csv` – Dataset containing tweets and sentiment labels.

## Results

- Accuracy and classification reports for VADER and each deep learning model.
- Confusion matrices and training history plots.
- Analysis of common failure cases.

## License

This project is for educational purposes.
