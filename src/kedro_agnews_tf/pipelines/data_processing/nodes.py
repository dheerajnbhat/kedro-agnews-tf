"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Dict
from sklearn.model_selection import train_test_split

VOCAB_SIZE = 10000  # arbitrarily chosen
EMBED_SIZE = 32  # arbitrarily chosen
MAX_LEN = 512


def preprocess_ag_news_train(ag_news_train: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for ag news.
    Args:
        ag_news_train: Raw data.
    Returns:
        Preprocessed data, with Title and Description columns
        combined.
    """
    # Combine title and description (better accuracy than using them as separate features)
    x_train = ag_news_train['Title'] + " " + ag_news_train['Description']
    # Class labels need to begin from 0
    y_train = ag_news_train['ClassIndex'].apply(lambda x: x - 1).values
    return x_train, y_train


def preprocess_ag_news_test(ag_news_test: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for ag news.
    Args:
        ag_news_test: Raw data.
    Returns:
        Preprocessed data, with Title and Description columns
        combined.
    """
    # Combine title and description (better accuracy than using them as separate features)
    x_test = ag_news_test['Title'] + " " + ag_news_test['Description']
    # Class labels need to begin from 0
    y_test = ag_news_test['ClassIndex'].apply(lambda x: x - 1).values
    return x_test, y_test


def _combine_columns(df, col1, col2):
    return df[col1] + " " + df[col2]


def _get_class_labels(df, col):
    # Class labels need to begin from 0
    return df['ClassIndex'].apply(lambda x: x - 1).values


def _convert_text_to_sequences(tokenizer, x):
    return tokenizer.texts_to_sequences(x)


def _pad_sequences(x, maxlen):
    return pad_sequences(x, maxlen=maxlen)


def preprocess_data(ag_news_train: pd.DataFrame, ag_news_test: pd.DataFrame):
    # Combine title and description (better accuracy than using them as separate features)
    x = _combine_columns(ag_news_train, 'Title', 'Description')
    x_test = _combine_columns(ag_news_test, 'Title', 'Description')

    y = _get_class_labels(ag_news_train, 'ClassIndex')
    y_test = _get_class_labels(ag_news_test, 'ClassIndex')

    # Create and Fit tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(x.values)

    # Tokenize data
    x = _convert_text_to_sequences(tokenizer, x)
    x_test = _convert_text_to_sequences(tokenizer, x_test)

    # Pad data
    x = _pad_sequences(x, MAX_LEN)
    x_test = _pad_sequences(x_test, MAX_LEN)

    return x, x_test, y, y_test


def train_val_split(x, y, parameters: Dict):
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=parameters["val_size"],
        random_state=parameters["random_state"]
    )
    return x_train, x_val, y_train, y_val
