# ====================================================
# ECHOSIS CLASSIFICATION MODELs
# ====================================================
#
# to train classification models
# for agreement annotation
#
# STILL IN DEVELOPMENT
#

from transformers import TFCamembertModel, CamembertTokenizer, TFAutoModelForSequenceClassification
from typing import Optional
import keras
import tensorflow as tf
import numpy as np
import polars as pl
from echosis.utils import load_file

# initialization of camemBERT
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = TFCamembertModel.from_pretrained('camembert-base')


id2label = {0: "accord", 1: "desaccord", 2: "ambigue", 3: "hs"}
label2id = {"accord": 0, "desaccord": 1, "ambigue": 2, "hs": 3}

model = TFAutoModelForSequenceClassification.from_pretrained(
    "camembert-base",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

def get_embeddings(texts) -> tf. Tensor:
    """transform text in camemBERT embeddings.

    Args:
        texts (str): text or context.

    Returns:
        tf.Tensor: text embeddings.
    """
    inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=128)
    outputs = model(inputs)
    # embeddings = outputs.last_hidden_state.mean(axis=1)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embeddings


def get_dataset(path: str) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """to get embeddings and labels in the right format for training.

    Args:
        path (str): path to train, test or dev file.

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: dataset for training.
    """
    df = load_file(
        path, valid_extensions=".txt"
    ).with_columns(
        pl.col("context").map_elements(lambda x: get_embeddings(x)),
        pl.col("text").map_elements(lambda x: get_embeddings(x))
    )

    context_embeddings = np.stack(df["context"].to_list())
    context_embeddings = tf.convert_to_tensor(context_embeddings, dtype=tf.float32)

    text_embeddings = np.stack(df["text"].to_list())
    text_embeddings = tf.convert_to_tensor(text_embeddings, dtype=tf.float32)

    labels = tf.convert_to_tensor([label2id[tok] for tok in df["label"].to_list()], dtype=tf.int32)

    return context_embeddings, text_embeddings, labels


def get_model(num_classes: Optional[int] = 3):
    """to create tensorflow model.

    Args:
        num_classes (int, optional): number of labels.
    """
    # inputs
    context_embedding_input = keras.layers.Input(shape=(768,), name='context_embedding')
    text_embedding_input = keras.layers.Input(shape=(768,), name='text_embedding')
    combined_embeddings = keras.layers.Concatenate()([context_embedding_input, text_embedding_input])
    # dense layers
    x = keras.layers.Dense(256, activation='relu')(combined_embeddings)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    # model
    model = keras.Model(inputs=[context_embedding_input, text_embedding_input], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=5e-4)
    return model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def train_model(train_path: str, test_path: str, dev_path: str):
    """to train tensorflow model for agreement classification.

    Args:
        train_path (str): path to train dataset.
        test_path (str): path to test dataset.
        dev_path (str): path to dev dataset.
    """
    train, test, dev = get_dataset(train_path), get_dataset(test_path), get_dataset(dev_path)

    model = get_model()
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = keras.callbacks.ModelCheckpoint('model_epoch_{epoch:02d}.h5', save_freq='epoch', period=5)
    model.fit(train, epochs=2, validation_data=dev, callbacks=[early_stopping, model_checkpoint])

    test_loss, test_accuracy = model.evaluate(test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')