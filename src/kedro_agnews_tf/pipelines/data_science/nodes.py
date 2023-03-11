"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""
import logging
from typing import Dict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from kedro_agnews_tf.pipelines.data_processing.nodes import VOCAB_SIZE, EMBED_SIZE, MAX_LEN


def _model():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAX_LEN))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    # Pooling Layer decreases sensitivity to features, thereby creating more
    # generalised data for better test results.
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1024))
    # Dropout layer nullifies certain random input values to generate a more
    # general dataset and prevent the problem of overfitting.
    model.add(Dropout(0.25))
    model.add(Dense(512))
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.25))
    # softmax is used as the activation function for multi-class classification
    # problems where class membership is required on more than two class labels.
    model.add(Dense(4, activation='softmax'))
    # model.summary()
    return model


def _callbacks():
    callbacks = [
        # EarlyStopping is used to stop at the epoch where val_accuracy
        # does not improve significantly
        EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-4,
            patience=4,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='model.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            # save_weights_only=True,
            verbose=1
        )
    ]


def train_model(x_train, x_val, y_train, y_val, parameters: Dict):
    model = _model()
    # Sparse Categorical Cross-entropy Loss because data is not one-hot encoded
    model.compile(
        loss=parameters['model_loss'],
        optimizer=parameters['model_optimizer'],
        metrics=['accuracy']
    )
    model.fit(
        x_train, y_train, batch_size=parameters['batch_size'], validation_data=(x_val, y_val),
        epochs=parameters['epochs'], callbacks=_callbacks()
    )
    return model


def evaluate_model(model, x_test, y_test):
    y_preds = [np.argmax(i) for i in model.predict(x_test)]
    logger = logging.getLogger(__name__)
    logger.info(f"Recall of the model is {recall_score(y_test, y_preds, average='micro')}")
    logger.info(f"Precision of the model is {precision_score(y_test, y_preds, average='micro')}")
    logger.info(f"Accuracy of the model is {accuracy_score(y_test, y_preds)}")
    
    cm = confusion_matrix(y_test, y_preds)
    labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']
    plt.figure()
    plot_confusion_matrix(cm, figsize=(16, 12), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(4), labels, fontsize=12)
    plt.yticks(range(4), labels, fontsize=12)

    return plt
