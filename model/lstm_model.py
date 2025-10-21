import random
import tqdm

from collections import Counter
from utils import vectors_util

from run import compute_results

import os
import gc

import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import datetime

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

# Agent classification
NUM_CLASSES = 10
NON_VIOLENT_CLASS = 9

label2idAll = {
        "EX_SPOUSE_PARTNER": 0,
        "KNOWNPERSON_FRIEND": 1,
        "OTHER_RELATIONSHIP": 2,
        "RELATIVE": 3,
        "SELF": 4,
        "SPOUSE_PARTNER": 5,
        "THIEF": 6,
        "UNKNOWN_PERSON": 7,
        "UNSPECIFIED_RELATIONSHIP": 8,
        "NON-VIOLENT": 9}

id2labelAll = {
        0: "EX_SPOUSE_PARTNER",
        1: "KNOWNPERSON_FRIEND",
        2: "OTHER_RELATIONSHIP",
        3: "RELATIVE",
        4: "SELF",
        5: "SPOUSE_PARTNER",
        6: "THIEF",
        7: "UNKNOWN_PERSON",
        8: "UNSPECIFIED_RELATIONSHIP",
        9: "NON-VIOLENT"}

label2idViolent = {
        "EX_SPOUSE_PARTNER": 0,
        "KNOWNPERSON_FRIEND": 1,
        "OTHER_RELATIONSHIP": 2,
        "RELATIVE": 3,
        "SPOUSE_PARTNER": 4,
        "THIEF": 5,
        "UNKNOWN_PERSON": 6,
        "UNSPECIFIED_RELATIONSHIP": 7}

id2labelViolent = {
        0: "EX_SPOUSE_PARTNER",
        1: "KNOWNPERSON_FRIEND",
        2: "OTHER_RELATIONSHIP",
        3: "RELATIVE",
        4: "SPOUSE_PARTNER",
        5: "THIEF",
        6: "UNKNOWN_PERSON",
        7: "UNSPECIFIED_RELATIONSHIP"}

def get_tensors(vectors_fp, train_data, dev_data, test_data, binary_classes):
    vectors = vectors_util.load_vectors(vectors_fp)
    vocab_size = len(vectors)
    vocabulary = list(vectors.keys())

    # Build Tokenizer on embeddings vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, filters='', lower=False)
    tokenizer.fit_on_texts(vocabulary)

    # Build Dataset
    if binary_classes:
        # 0 non violent 1 violent
        y_train = [0 if int(e["label"]) == NON_VIOLENT_CLASS else 1 for e in train_data.values()]
        y_dev = [0 if int(e["label"]) == NON_VIOLENT_CLASS else 1 for e in dev_data.values()]
        y_test = [0 if int(e["label"]) == NON_VIOLENT_CLASS else 1 for e in test_data.values()]
    else:
        y_train = [int(e["label"]) for e in train_data.values()]
        y_dev = [int(e["label"]) for e in dev_data.values()]
        y_test = [int(e["label"]) for e in test_data.values()]

    x_train = [e["sentence"] for e in train_data.values()]
    x_dev = [e["sentence"] for e in dev_data.values()]
    x_test = [e["sentence"] for e in test_data.values()]

    # Shuffle Train Set
    c = list(zip(x_train, y_train))
    random.shuffle(c)
    x_train, y_train = zip(*c)

    # Shuffle Test Set
    c = list(zip(x_test, y_test))
    random.shuffle(c)
    x_test, y_test = zip(*c)

    # Shuffle Dev Set
    c = list(zip(x_dev, y_dev))
    random.shuffle(c)
    x_dev, y_dev = zip(*c)

    all_labels = set()
    all_labels.update(y_train)
    all_labels.update(y_dev)
    all_labels.update(y_test)
    # Tokenize data
    tokenized_x_train = tokenizer.texts_to_sequences(x_train)
    tokenized_x_dev = tokenizer.texts_to_sequences(x_dev)
    tokenized_x_test = tokenizer.texts_to_sequences(x_test)

    max_length = max([max([len(e) for e in tokenized_x_train])], [max([len(e) for e in tokenized_x_dev])],
                     [max([len(e) for e in tokenized_x_test])])[0]

    # Pad sequences
    padded_x_train = tf.keras.preprocessing.sequence.pad_sequences(tokenized_x_train, padding='post', maxlen=max_length)
    padded_x_dev = tf.keras.preprocessing.sequence.pad_sequences(tokenized_x_dev, padding='post', maxlen=max_length)
    padded_x_test = tf.keras.preprocessing.sequence.pad_sequences(tokenized_x_test, padding='post', maxlen=max_length)

    # Convert to tensors
    tensor_x_train = tf.convert_to_tensor(padded_x_train)
    tensor_x_dev = tf.convert_to_tensor(padded_x_dev)
    tensor_x_test = tf.convert_to_tensor(padded_x_test)
    tensor_y_train = tf.convert_to_tensor(y_train)
    tensor_y_dev = tf.convert_to_tensor(y_dev)
    tensor_y_test = tf.convert_to_tensor(y_test)

    # Build embedding matrix
    embedding_matrix = vectors_util.get_embedding_matrix(vectors, tokenizer)

    return tensor_x_train, tensor_y_train, tensor_x_dev, tensor_y_dev, tensor_x_test, tensor_y_test, embedding_matrix, vocab_size, max_length, x_test, y_test, all_labels

def get_tensors2(vectors, data, binary_classes):

    vocab_size = len(vectors)
    vocabulary = list(vectors.keys())

    # Build Tokenizer on embeddings vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, filters='', lower=False)
    tokenizer.fit_on_texts(vocabulary)

    sentences = [e["sentence"] for e in data.values()]

    if binary_classes:
        labels = [0 if int(e["label"]) == NON_VIOLENT_CLASS else 1 for e in data.values()]
    else:
        labels = [int(e["label"]) for e in data.values()]

    # Shuffle data
    c = list(zip(sentences, labels))
    random.shuffle(c)
    sentences, labels = zip(*c)

    all_labels = set()
    all_labels.update(labels)

    # Tokenize data
    tokenized_sentences = tokenizer.texts_to_sequences(sentences)

    max_length = max([len(e) for e in tokenized_sentences])

    # Pad sequences
    padded_sentences = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sentences, padding='post', maxlen=max_length)

    # Convert to tensors
    tensor_sentences = tf.convert_to_tensor(padded_sentences)
    tensor_labels = tf.convert_to_tensor(labels)

    # Build embedding matrix
    embedding_matrix = vectors_util.get_embedding_matrix(vectors, tokenizer)

    return tensor_sentences, tensor_labels, embedding_matrix, vocab_size, max_length, all_labels

def train_model_v_nv(train_data, dev_data, vectors, split, setting):

    tensor_x_train, tensor_y_train, _, _, max_length_train, _ = get_tensors2(vectors, train_data, True)
    tensor_x_dev, tensor_y_dev, embedding_matrix, vocab_size, max_length_dev, _ = get_tensors2(vectors, dev_data, True)
    max_length = max(max_length_train, max_length_dev)

    # Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=300,
            # Use masking to handle the variable sequence lengths
            mask_zero=True,
            input_length=max_length,
            weights=[embedding_matrix]
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(1)
    ])

    print(model.summary())

    # Compile Model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    # Start training
    history = model.fit(tensor_x_train, tensor_y_train, epochs=10)
                        #validation_data=(tensor_x_dev, tensor_y_dev),
                        #validation_steps=30)

    models_dir = "../resources/data/output/1_v_nv/models/"
    model_out_folder = models_dir + setting + "_" + split + "/"
    os.makedirs(model_out_folder, exist_ok=True)
    model.save(model_out_folder + "model.keras")

def run_model_v_nv(train_data, dev_data, test_data, vectors_fp, split, setting):

    # Get tokenized data
    tensor_x_train, tensor_y_train, tensor_x_dev, tensor_y_dev, tensor_x_test, tensor_y_test, embedding_matrix, vocab_size, max_length, x_test, y_test, labels_set = get_tensors(vectors_fp, train_data, dev_data, test_data, True)

    # Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=300,
            # Use masking to handle the variable sequence lengths
            mask_zero=True,
            input_length=max_length,
            weights=[embedding_matrix]
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(1)
    ])
    print(model.summary())

    # Compile Model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    # Start training
    history = model.fit(tensor_x_train, tensor_y_train, epochs=10)
                        #validation_data=(tensor_x_dev, tensor_y_dev),
                        #validation_steps=30)

    test_loss, test_acc = model.evaluate(tensor_x_test, tensor_y_test)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    y_pred = model.predict(tensor_x_test)
    threshold = 0.5
    y_pred = np.where(y_pred > threshold, 1,0)
    print(classification_report(y_test, y_pred))

    print("-"*50)
    print("Counts")
    counts = Counter(y_test)
    for c in counts:
        print(c, counts[c])

    models_dir = "../resources/data/output/1_v_nv/models/"
    model_out_folder = models_dir + setting + "_" + split + "/"
    os.makedirs(model_out_folder, exist_ok=True)
    model.save(model_out_folder + "model.keras")

    data_out_folder = "../resources/data/output/1_v_nv/" + setting + "_" + split + "/"
    os.makedirs(data_out_folder, exist_ok=True)

    test_file = ""
    for i, text in enumerate(x_test):
        test_file += text + "\t" + str(y_pred[i][0]) + "\t" + str(y_test[i]) + "\n"

    open(data_out_folder + "test_prediction.txt", "w").write(test_file)

    prediction_ids = []
    nv_prediction_ids = []
    for i, text in enumerate(x_test):
        sentence = x_test[i]
        label = y_pred[i]

        if label > 0:
            for entry_id in test_data:
                if test_data[entry_id]['sentence'] == sentence:
                    prediction_ids.append(entry_id + "\t" + test_data[entry_id]["label"])
        else:
            for entry_id in test_data:
                if test_data[entry_id]['sentence'] == sentence:
                    nv_prediction_ids.append(entry_id + "\t" + test_data[entry_id]["label"])

    open(data_out_folder + "violent_ids.txt", "w").write("\n".join(prediction_ids))
    open(data_out_folder + "non_violent_ids.txt", "w").write("\n".join(nv_prediction_ids))

def run_model_all(train_data, dev_data, test_data, vectors_fp, split, setting):

    # Get tokenized data
    tensor_x_train, tensor_y_train, tensor_x_dev, tensor_y_dev, tensor_x_test, tensor_y_test, embedding_matrix, vocab_size, max_length, x_test, y_test, labels_set = get_tensors(vectors_fp, train_data, dev_data, test_data, False)

    # Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=300,
            # Use masking to handle the variable sequence lengths
            mask_zero=True,
            input_length=max_length,
            weights=[embedding_matrix]
        ),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Conv1D(64, 5, activation='relu'),
        # tf.keras.layers.MaxPooling1D(pool_size=4),
        #tf.keras.layers.LSTM(100),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(len(labels_set), activation='softmax')
    ])
    print(model.summary())

    # Compile Model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    # Start training
    history = model.fit(tensor_x_train, tensor_y_train, epochs=20,
                        validation_data=(tensor_x_dev, tensor_y_dev),
                        validation_steps=30)

    test_loss, test_acc = model.evaluate(tensor_x_test, tensor_y_test)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    y_pred = model.predict(tensor_x_test)
    print("-"*50)
    print(y_pred)
    print("-"*50)
    # threshold = 0.5
    # y_pred = np.where(y_pred > threshold, 1,0)
    y_pred = [list(l).index(max(list(l))) for l in y_pred]
    print(classification_report(y_test, y_pred))

    print("-"*50)
    print("Counts")
    counts = Counter(y_test)
    for c in counts:
        print(c, counts[c])

    # --------------
    models_dir = "../resources/data/output/2_agent_classification/models/"
    model_out_folder = models_dir + setting + "_" + split + "/"
    os.makedirs(model_out_folder, exist_ok=True)
    model.save(model_out_folder + "model")

    data_out_folder = "../resources/data/output/2_agent_classification/" + setting + "_" + split + "/"
    os.makedirs(data_out_folder, exist_ok=True)

    test_file = ""
    for i, text in enumerate(x_test):
        test_file += text + "\t" + str(y_pred[i]) + "\t" + str(y_test[i]) + "\n"

    open(data_out_folder + "test_prediction.txt", "w").write(test_file)

    prediction_ids = []
    nv_prediction_ids = []
    for i, text in enumerate(x_test):
        sentence = x_test[i]
        label = y_pred[i]

        if label != NON_VIOLENT_CLASS:
            for entry_id in test_data:
                if test_data[entry_id]['sentence'] == sentence:
                    prediction_ids.append(entry_id + "\t" + test_data[entry_id]["label"])
        else:
            for entry_id in test_data:
                if test_data[entry_id]['sentence'] == sentence:
                    nv_prediction_ids.append(entry_id + "\t" + test_data[entry_id]["label"])

    open(data_out_folder + "violent_ids.txt", "w").write("\n".join(prediction_ids))
    open(data_out_folder + "non_violent_ids.txt", "w").write("\n".join(nv_prediction_ids))

def cv_run_model_v_nv(folds, data, vectors, split, setting):
    """
        Run the model using n-fold Cross-validation.
    """

    enable_cnn = "CNN" in split

    # Set seed for reproducibility
    tf.random.set_seed(92)
    np.random.seed(92)
    random.seed(92)

    losses = []
    accuracies = []
    nv_precisions = []
    v_precisions = []
    nv_recalls = []
    v_recalls = []
    nv_f1s = []
    v_f1s = []
    class_accuracies = []

    # Get tokenized data
    tensor_x, tensor_y, embedding_matrix, vocab_size, max_length, _ = get_tensors2(vectors, data, True)

    # Use sklearn to performe n-fold CV
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=92)

    n_fold = 0

    # Compute folds indices (when seed is set explicitly, these should be deterministic)
    fold_indices = []
    for train_idx, test_idx in kfold.split(tensor_x, tensor_y):
        fold_indices.append((train_idx, test_idx))

    for train, test in fold_indices:

        # Slice tensors according to current fold's split
        train_x = tf.gather(tensor_x, train)
        train_y = tf.gather(tensor_y, train)
        test_x = tf.gather(tensor_x, test)
        test_y = tf.gather(tensor_y, test)

        # Prepend a CNN to the LSTM
        if enable_cnn:

            input_layer = tf.keras.layers.Input(shape=(max_length,))

            embedding_layer = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=300,
                mask_zero=False,
                input_length=max_length,
                weights=[embedding_matrix],
                trainable=False
            )(input_layer)
            dropout_emb = tf.keras.layers.SpatialDropout1D(0.2)(embedding_layer)

            conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(dropout_emb)
            conv2 = tf.keras.layers.Conv1D(64, 4, activation='relu', padding='same')(dropout_emb)
            conv3 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(dropout_emb)
            concatenated = tf.keras.layers.Concatenate(axis=-1)([conv1, conv2, conv3])

            normalized = tf.keras.layers.BatchNormalization()(concatenated)
            pooled = tf.keras.layers.MaxPooling1D(pool_size=4)(normalized)

            lstm_output = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)
            )(pooled)

            dense = tf.keras.layers.Dense(64, activation='relu')(lstm_output)
            dense = tf.keras.layers.Dropout(0.5)(dense)

            output_layer = tf.keras.layers.Dense(1)(dense)
            model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        else:

            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=300,
                    mask_zero=True,
                    input_length=max_length,
                    weights=[embedding_matrix],
                    trainable=False # Freeze embeddings
                ),
                tf.keras.layers.SpatialDropout1D(0.2),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)
                ),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1) # Output layer (logits)
            ])

        # Compile Model
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])

        print(f"Training on fold #{n_fold+1}")

        start_time = datetime.datetime.now()

        # Start training
        history = model.fit(train_x, train_y, epochs=10, validation_split=0.1)

        end_time = datetime.datetime.now()
        time_delta = end_time - start_time
        print(f"Training time: {time_delta}")

        # Compute performance metrics on current fold
        test_loss, test_acc = model.evaluate(test_x, test_y)
        losses.append(test_loss)
        accuracies.append(test_acc)

        start_time = datetime.datetime.now()

        logits = model.predict(test_x, verbose=0)
        probabilities = tf.nn.sigmoid(logits).numpy().flatten()  # Use TF's sigmoid
        bin_predictions = (probabilities > 0.5).astype(int)
        test_y = test_y.numpy()

        end_time = datetime.datetime.now()
        time_delta = end_time - start_time
        print(f"Prediction time: {time_delta}")

        # Precision
        precision = precision_score(test_y, bin_predictions, average=None, zero_division=np.nan)
        nv_precisions.append(precision[0])
        v_precisions.append(precision[1])

        # Recall
        recall = recall_score(test_y, bin_predictions, average=None, zero_division=np.nan)
        nv_recalls.append(recall[0])
        v_recalls.append(recall[1])

        # F1
        f1 = f1_score(test_y, bin_predictions, average=None, zero_division=np.nan)
        nv_f1s.append(f1[0])
        v_f1s.append(f1[1])

        # Class accuracy
        accuracy = accuracy_score(test_y, bin_predictions)
        class_accuracies.append(accuracy)

        # Clean up to prevent memory leaks
        del model
        tf.keras.backend.clear_session()
        gc.collect()

        n_fold += 1

    print("\n Cross validation completed")
    print(f"Average loss: {sum(losses) / folds}")
    print(f"Average accuracy: {sum(accuracies) / folds}")
    print(f"NV precisions: {nv_precisions}")
    print(f"V precisions: {v_precisions}")
    print(f"NV Recalls: {nv_recalls}")
    print(f"V Recalls: {v_recalls}")
    print(f"NV F1s: {nv_f1s}")
    print(f"V F1s: {v_f1s}")
    print(f"Class accuracy: {class_accuracies}")

def cv_run_model_all(folds, data, vectors, split, setting, output_dir, violent_only = False):
    """
    Run LSTM model using n-fold Cross-validation for multiclass classification.

    """

    enable_cnn = "CNN" in split

    if violent_only:
        l2id = label2idViolent
        id2l = id2labelViolent
        NUM_CLASSES = 8
    else:
        l2id = label2idAll
        id2l = id2labelAll

    # Set seed for reproducibility
    tf.random.set_seed(92)
    np.random.seed(92)
    random.seed(92)

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    class_accuracies = []

    # Check if a file with previous misclassifcations for the current model exists, and if so delete it
    misclassification_file_dir = output_dir + "multiclass_misclassification/" + split
    os.makedirs(misclassification_file_dir, exist_ok=True)
    if os.path.isfile(misclassification_file_dir + "/ids.txt"): os.remove(misclassification_file_dir + "/ids.txt")
    if os.path.isfile(misclassification_file_dir + "/labels.txt"): os.remove(misclassification_file_dir + "/labels.txt")

    # Use sklearn to perform n-fold CV
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=92)

    n_fold = 0

    # Get tokenized data
    tensor_x, tensor_y, embedding_matrix, vocab_size, max_length, data_set_labels = get_tensors2(vectors, data, False)

    # Since classes are highly unbalanced, sklearn is used to mitigate this aspect
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(list(l2id.values())),
        y=tensor_y.numpy())
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Compute folds indices (when seed is set explicitly, these should be deterministic)
    fold_indices = []
    for train_idx, test_idx in kfold.split(tensor_x, tensor_y):
        fold_indices.append((train_idx, test_idx))

    for train, test in fold_indices:

        # Slice tensors according to current fold's split
        train_x = tf.gather(tensor_x, train)
        train_y = tf.gather(tensor_y, train)
        test_x = tf.gather(tensor_x, test)
        test_y = tf.gather(tensor_y, test)

        # Prepend a CNN to the LSTM
        if enable_cnn:

            input_layer = tf.keras.layers.Input(shape=(max_length,))

            embedding_layer = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=300,
                mask_zero=False,
                input_length=max_length,
                weights=[embedding_matrix],
                trainable=False
            )(input_layer)
            dropout_emb = tf.keras.layers.SpatialDropout1D(0.2)(embedding_layer)

            conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(dropout_emb)
            conv2 = tf.keras.layers.Conv1D(64, 4, activation='relu', padding='same')(dropout_emb)
            conv3 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(dropout_emb)
            concatenated = tf.keras.layers.Concatenate(axis=-1)([conv1, conv2, conv3])

            normalized = tf.keras.layers.BatchNormalization()(concatenated)
            pooled = tf.keras.layers.MaxPooling1D(pool_size=4)(normalized)

            lstm_output = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)
            )(pooled)

            dense = tf.keras.layers.Dense(64, activation='relu')(lstm_output)
            dense = tf.keras.layers.Dropout(0.5)(dense)

            output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(dense)
            model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        else:

            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=300,
                    mask_zero=True,
                    input_length=max_length,
                    weights=[embedding_matrix],
                    trainable=False  # Freeze embeddings
                ),
                tf.keras.layers.SpatialDropout1D(0.2),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)
                ),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Multiclass output
            ])

        # Compile Model
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy']
        )

        print(f"Training on fold #{n_fold+1}")
        start_time = datetime.datetime.now()

        # Start training with fixed number of epochs
        _ = model.fit(
            train_x,
            train_y,
            epochs=10,
            class_weight=class_weight_dict,
            validation_split=0.1,
            )

        end_time = datetime.datetime.now()
        time_delta = end_time - start_time
        print(f"Training time: {time_delta}")

        # Compute performance metrics on current fold
        test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
        losses.append(test_loss)
        accuracies.append(test_acc)

        start_time = datetime.datetime.now()

        predictions = model.predict(test_x)
        predictions = np.argmax(predictions, axis=1) # Convert probabilities to class

        end_time = datetime.datetime.now()
        time_delta = end_time - start_time
        print(f"Prediction time: {time_delta}")

        precision = precision_score(test_y, predictions, average=None, zero_division=np.nan)
        precisions.append(precision)
        recall = recall_score(test_y, predictions, average=None, zero_division=np.nan)
        recalls.append(recall)
        f1 = f1_score(test_y, predictions, average=None, zero_division=np.nan)
        f1s.append(f1)

        print(f"Accuracy: {accuracies}")
        print(f"Loss: {losses}")
        print(f"Precisions: {precisions}")
        print(f"Recalls: {recalls}")
        print(f"F1s: {f1s}")

        # Store wrong predictions for later analysis
        wrong_prediction_ids = []
        wrong_prediction_labels = []

        write_mode = 'a' if os.path.isfile(misclassification_file_dir + "/ids.txt") else 'w'

        for entry_id, test_entry, prediction in zip(test, test_y, predictions):
            if test_entry != prediction:
                wrong_prediction_ids.append(entry_id)
                wrong_prediction_labels.append(id2l[int(test_entry)] + "\t" + id2l[int(prediction)])
        with open(misclassification_file_dir + "/ids.txt", write_mode) as misclassification_file:
            for entry_id in wrong_prediction_ids: misclassification_file.write(f"{entry_id}\n")
        with open(misclassification_file_dir + "/labels.txt", write_mode) as misclassification_file:
            for label in wrong_prediction_labels: misclassification_file.write(f"{label}\n")

        # Clean up to prevent memory leaks
        del model
        tf.keras.backend.clear_session()
        gc.collect()

        n_fold += 1

    print("\n\nCross validation completed")
    print(f"Average loss: {sum(losses) / folds}")
    print(f"Final Average accuracy: {sum(accuracies) / folds}")
    print(f"Final Precisions: {precisions}")
    print(f"Final Recalls: {recalls}")
    print(f"Final F1s: {f1s}")
