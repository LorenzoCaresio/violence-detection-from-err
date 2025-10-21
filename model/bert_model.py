from transformers import AutoTokenizer
from transformers import RobertaTokenizer
from transformers import AutoConfig
from transformers import create_optimizer
from transformers import TFAutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset

from run import compute_results

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

import os
import gc
import random

import datetime

import tensorflow as tf
import numpy as np

# Binary classification
id2label = {0: "NON-VIOLENT", 1: "VIOLENT"}
label2id = {"NON-VIOLENT": 0, "VIOLENT": 1}

# Agent classification
NUM_CLASSES = 10 # 10 for all classes, 8 for violent only
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

def run_model_v_nv(train_data, dev_data, test_data, split, setting):
    """
        Run a BERT model to perform binary classification on given data.
        This is not used in paper's experiments, and should not be considered for replication.
    """

    batch_size = 16
    num_epochs = 3

    batches_per_epoch = len(train_data) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained("indigo-ai/BERTino")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # Import pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "indigo-ai/BERTino",
        num_labels=2, # V-NV
        id2label=id2label,
        label2id=label2id
    )

    # Prepare train set to be used to train the model
    train_data_dict = [{'sentence': report['sentence'], 'label': report['label']} for _, report in train_data.items()]
    train_set = Dataset.from_list(train_data_dict)
    tokenized_train_set = tokenizer(train_set["sentence"], padding="max_length", return_tensors="np", truncation=True)
    train_set_labels = np.array(train_set["label"])
    train_set_labels = np.where(train_set_labels == '9', 0, 1) # From multi-label to binary category

    """
    tf_train_set = model.prepare_tf_dataset(
        tokenized_train_set,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )
    """

    # Compile Model
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
    model.compile(optimizer=optimizer, metrics=['accuracy'])

    # Start training
    history = model.fit(tokenized_train_set, train_set_labels, epochs=num_epochs)

    """
    # Store trained model
    models_dir = "../resources/data/output/1_v_nv/models/"
    model_out_folder = models_dir + setting + "_" + split + "/"
    os.makedirs(model_out_folder, exist_ok=True)
    #model.save_pretrained(model_out_folder)
    #tokenizer.save_pretrained(model_out_folder) # Store the tokenizer too

    # Load the model
    model = TFAutoModelForSequenceClassification.from_pretrained(model_out_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_out_folder)
    model.compile(optimizer=optimizer, metrics=['accuracy'])
    """

    # Prepare test set to be used to evaluate trained model
    test_data_dict = [{'sentence': report['sentence'], 'label': report['label']} for _, report in test_data.items()][:200]
    test_set = Dataset.from_list(test_data_dict)
    tokenized_test_set = tokenizer(test_set['sentence'], padding="max_length", return_tensors="np", truncation=True)
    test_set_labels = np.array(test_set['label'])
    test_set_labels = np.where(test_set_labels == '9', 0, 1) # From multi-label to binary category

    """
    tf_test_set = model.prepare_tf_dataset(
        test_set,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )
    """

    # Evaluate model
    test_loss, test_acc = model.evaluate(tokenized_test_set, test_set_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    # Get model predictions
    start_time = datetime.datetime.now()
    y_pred = model.predict(tokenized_test_set)
    logits = y_pred.logits
    probabilities = tf.sigmoid(logits).numpy().squeeze()
    threshold = 0.5
    y_pred = [1 if p[1] > threshold else 0 for p in probabilities]

    end_time = datetime.datetime.now()
    time_delta = end_time - start_time
    print(f"Total execution time: {time_delta}")

    print(classification_report(test_set_labels, y_pred))

    print("-"*50)
    print("Counts")
    counts = Counter(test_set_labels)
    for c in counts:
        print(c, counts[c])

    data_out_folder = "../resources/data/output/1_v_nv/" + setting + "_" + split + "/"
    os.makedirs(data_out_folder, exist_ok=True)

    test_set_texts = np.array(test_set["sentence"])

    # Store test predictions
    test_file = ""
    for i, text in enumerate(test_set_texts):
        test_file += text + "\t" + str(y_pred[i]) + "\t" + str(test_set_labels[i]) + "\n"

    open(data_out_folder + "test_prediction.txt", "w").write(test_file)

    prediction_ids = []
    nv_prediction_ids = []

    for i, text in enumerate(test_set_texts):
        sentence = test_set_texts[i]
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

def run_model_all(train_data, dev_data, test_data, split, setting):
    """
        Run a BERT model to perform multi-class classification on given data
        This is not used in paper's experiments, and should not be considered for replication.
    """

    batch_size = 16
    num_epochs = 3

    batches_per_epoch = len(train_data) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained("indigo-ai/BERTino")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # Import pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "indigo-ai/BERTino",
        num_labels=9,
        id2label=id2labelAll,
        label2id=label2idAll
    )

    # Prepare train set to be used to train the model
    train_data_dict = [{'sentence': report['sentence'], 'label': report['label']} for _, report in train_data.items()]
    train_set = Dataset.from_list(train_data_dict)
    tokenized_train_set = tokenizer(train_set["sentence"], padding="max_length", return_tensors="np", truncation=True)
    train_set_labels = np.array(train_set["label"])

    """
    tf_train_set = model.prepare_tf_dataset(
        tokenized_train_set,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )
    """

    # Compile Model
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
    model.compile(
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=['accuracy'])

    # Start training
    history = model.fit(tokenized_train_set, train_set_labels, epochs=num_epochs)

    """
    # Store trained model
    models_dir = "../resources/data/output/1_v_nv/models/"
    model_out_folder = models_dir + setting + "_" + split + "/"
    os.makedirs(model_out_folder, exist_ok=True)
    model.save_pretrained(model_out_folder)
    tokenizer.save_pretrained(model_out_folder) # Store the tokenizer too

    # Load the model
    model = TFAutoModelForSequenceClassification.from_pretrained(model_out_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_out_folder)
    model.compile(optimizer=optimizer, metrics=['accuracy'])
    """

    # Prepare test set to be used to evaluate trained model
    test_data_dict = [{'sentence': report['sentence'], 'label': report['label']} for _, report in test_data.items()]
    test_set = Dataset.from_list(test_data_dict)
    tokenized_test_set = tokenizer(test_set['sentence'], padding="max_length", return_tensors="np", truncation=True)
    test_set_labels = np.array(test_set['label'])

    """
    tf_test_set = model.prepare_tf_dataset(
        test_set,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )
    """

    # Evaluate model
    test_loss, test_acc = model.evaluate(tokenized_test_set, test_set_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    # Get model predictions
    y_pred = model.predict(tokenized_test_set)
    y_pred = [list(l).index(max(list(l))) for l in y_pred]

    print(classification_report(test_set_labels, y_pred))

    print("-"*50)
    print("Counts")
    counts = Counter(test_set_labels)
    for c in counts:
        print(c, counts[c])

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

def train_model_v_nv(train_data, dev_data, split, setting):
    """
        Train and store a BERT model to perform binary classification on given data.
        This is not used in paper's experiments, and should not be considered for replication.
    """

    batch_size = 16
    num_epochs = 3

    batches_per_epoch = len(train_data) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained("indigo-ai/BERTino")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # Import pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "indigo-ai/BERTino",
        num_labels=2, # V-NV
        id2label=id2label,
        label2id=label2id
    )

    # Prepare train set to be used to train the model
    train_data_dict = [{'sentence': report['sentence'], 'label': report['label']} for _, report in train_data.items()]
    train_set = Dataset.from_list(train_data_dict)
    tokenized_train_set = tokenizer(train_set["sentence"], padding="max_length", return_tensors="np", truncation=True)
    train_set_labels = np.array(train_set["label"])
    train_set_labels = np.where(train_set_labels == '9', 0, 1) # From multi-label to binary category

    """
    tf_train_set = model.prepare_tf_dataset(
        tokenized_train_set,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )
    """

    # Compile Model
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
    model.compile(optimizer=optimizer, metrics=['accuracy'])

    # Start training
    history = model.fit(tokenized_train_set, train_set_labels, epochs=num_epochs)

    # Store trained model
    models_dir = "../resources/data/output/1_v_nv/models/"
    model_out_folder = models_dir + setting + "_" + split + "/"
    os.makedirs(model_out_folder, exist_ok=True)
    model.save_pretrained(model_out_folder)
    tokenizer.save_pretrained(model_out_folder) # Store the tokenizer too

def cv_run_model_v_nv(folds, data, split, setting):
    """
        Run the model using n-fold Cross-validation.
        This is used in paper's experiments, and SHOULD be considered for replication.
    """

    # Set seed for reproducibility
    tf.random.set_seed(92)
    np.random.seed(92)

    losses = []
    accuracies = []
    nv_precisions = []
    v_precisions = []
    nv_recalls = []
    v_recalls = []
    nv_f1s = []
    v_f1s = []
    class_accuracies = []

    batch_size = 16
    num_epochs = 3

    # Use sklearn to performe n-fold CV
    #kfold = KFold(n_splits=folds, shuffle=True)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)

    n_fold = 0

    # Prepare tokenizer
    if "BERT-BASE" in split:
        model_name = "dbmdz/bert-base-italian-uncased"
        uncased_model = True
    else:
        model_name = "indigo-ai/BERTino"
        uncased_model = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=uncased_model)

    # Prepare train set to be used to train the model
    data_dict = [{'sentence': report['sentence'], 'label': report['label']} for _, report in data.items()]
    data_set = Dataset.from_list(data_dict)
    tokenized_data_set = tokenizer(data_set['sentence'], padding="max_length", return_tensors="np", truncation=True)
    tokenized_data_set = tokenized_data_set['input_ids']
    data_set_labels = np.array(data_set['label'])
    data_set_labels = np.where(data_set_labels == '9', 0, 1) # From multi-label to binary category

    # Compute folds indices (when seed is set explicitly, these should be deterministic)
    fold_indices = []
    for train_idx, test_idx in kfold.split(tokenized_data_set, data_set_labels):
        fold_indices.append((train_idx, test_idx))

    for train, test in fold_indices:

        # Slice tensors according to current fold's split
        train_x = tf.gather(tokenized_data_set, train)
        train_y = tf.gather(data_set_labels, train)
        test_x = tf.gather(tokenized_data_set, test)
        test_y = tf.gather(data_set_labels, test)

        batches_per_epoch = len(train_y) // batch_size
        total_train_steps = int(batches_per_epoch * num_epochs)

        # Pick corresponding model
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2, # V-NV
            id2label=id2label,
            label2id=label2id,
        )

        # Compile Model
        optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        print(f"Training on fold #{n_fold+1}")

        start_time = datetime.datetime.now()

        history = model.fit(train_x, train_y, epochs=num_epochs, validation_split=0.1)

        end_time = datetime.datetime.now()
        time_delta = end_time - start_time
        print(f"Training time: {time_delta}")

        # Compute metrics on current fold
        test_loss, test_acc = model.evaluate(test_x, test_y)

        losses.append(test_loss)
        accuracies.append(test_acc)

        start_time = datetime.datetime.now()

        all_predictions = []
        batch_size = 100
        for i in range(0, len(test_x), batch_size):
            batch_x = test_x[i:i + batch_size]
            batch_predictions = model.predict(batch_x)
            all_predictions.extend(batch_predictions.logits)

        # Convert logits to predictions
        predictions = tf.nn.softmax(all_predictions, axis=1).numpy()
        bin_predictions = np.argmax(predictions, axis=1)

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

def cv_run_model_all(folds, data, split, setting, output_dir, heads_finetuning=False, violent_only=False):
    """
        Run the model using n-fold Cross-validation for multi-class classification.
        This is used in paper's experiments, and SHOULD be considered for replication.
        heads_finetuning indicates whether to use a two-phase fine-tuning strategy or not,
        this technique is a refinement that we didn't include in the paper.
    """

    if violent_only:
        l2id = label2idViolent
        id2l = id2labelViolent
        NUM_CLASSES = 8
    else:
        l2id = label2idAll

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

    batch_size = 16
    num_epochs = 3

    # Check if a file with previous misclassifcations for the current model exists, and if so delete it
    misclassification_file_dir = output_dir + "multiclass_misclassification/" + split
    os.makedirs(misclassification_file_dir, exist_ok=True)
    if os.path.isfile(misclassification_file_dir + "/ids.txt"): os.remove(misclassification_file_dir + "/ids.txt")

    # Use sklearn to performe n-fold CV
    #kfold = KFold(n_splits=folds, shuffle=True)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=92)

    n_fold = 0

    # Prepare tokenizer
    if "BERT-BASE" in split:
        model_name = "dbmdz/bert-base-italian-uncased"
        uncased_model = True
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=uncased_model)
    else:
        model_name = "indigo-ai/BERTino"
        uncased_model = False
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=uncased_model)

    # Prepare train set to be used to train the model
    data_dict = [{'sentence': report['sentence'], 'label': int(report['label'])} for _, report in data.items()]
    data_set = Dataset.from_list(data_dict)
    tokenized_data_set = tokenizer(data_set['sentence'], padding="max_length", return_tensors="np", truncation=True)
    tokenized_data_set = tokenized_data_set['input_ids']
    data_set_labels = np.array(data_set['label'])

    # Since classes are highly unbalanced, sklearn is used to mitigate this aspect
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(list(l2id.values())),
        y=data_set_labels)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Compute folds indices (when seed is set explicitly, these should be deterministic)
    fold_indices = []
    for train_idx, test_idx in kfold.split(tokenized_data_set, data_set_labels):
        fold_indices.append((train_idx, test_idx))

    for train, test in fold_indices:

        # Slice tensors according to current fold's split
        train_x = tf.gather(tokenized_data_set, train)
        train_y = tf.gather(data_set_labels, train)
        test_x = tf.gather(tokenized_data_set, test)
        test_y = tf.gather(data_set_labels, test)

        # Convert labels to one-hot encoding
        train_y_one_hot = tf.keras.utils.to_categorical(train_y, num_classes=NUM_CLASSES)
        test_y_one_hot = tf.keras.utils.to_categorical(test_y, num_classes=NUM_CLASSES)

        batches_per_epoch = len(train_y) // batch_size
        total_train_steps = int(batches_per_epoch * num_epochs)

        if "BERT-BASE" in split:
            pytorch_weights = True
        else:
            pytorch_weights = False

        # Pick corresponding model
        original_model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True # (!) Necessary for newer version of TF
        )

        if heads_finetuning:

            # Implement a two-phase fine-tuining strategy
            # 1. Freeze the base model and train only the classification head
            # 2. Unfreeze the entire model and fine-tune with a lower learning rate

            # Since the original model is pre-trained is on binary-classification, its architecutre has to be modified
            model_base = original_model.layers[0]
            model_base.trainable = False # Freeze the model for first fine-tuning phase
            input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
            sequence_output = model_base(input_ids)[0] # Get the last hidden state
            pooled_output = sequence_output[:, 0, :] # CLS token
            dropout = tf.keras.layers.Dropout(0.1)(pooled_output) # Add a dropout layer for regularization
            outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name="classifier")(dropout) # Add a multi-class classification head
            model = tf.keras.Model(inputs=input_ids, outputs=outputs) # Create the new model

            print(f"Training on fold #{n_fold+1} (with heads finetuning)")

            # Compile the model for phase 1
            ph1_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4) # Higher LR
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                optimizer=ph1_optimizer,
                metrics=['accuracy'])

            start_time = datetime.datetime.now()

            # Fine-tune the classification head
            _ = model.fit(
                train_x,
                train_y_one_hot,
                epochs=2,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                validation_split=0.1,
            )

            # Unfreeze the model for phase 2
            model_base.trainable = True

            # Compile the model for phase 2
            ph2_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6) # Lower LR
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                optimizer=ph2_optimizer,
                metrics=['accuracy'])

            # Fine-tune the whole model
            _ = model.fit(
                train_x,
                train_y_one_hot,
                initial_epoch=2,
                epochs=num_epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                validation_split=0.1,
            )

            end_time = datetime.datetime.now()
            time_delta = end_time - start_time
            print(f"Training time: {time_delta}")

        else: # Single-phase finetuning

            # Since the original model is pre-trained is on binary-classification, its architecutre has to be modified
            model_base = original_model.layers[0]
            input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
            sequence_output = model_base(input_ids)[0] # Get the last hidden state
            pooled_output = sequence_output[:, 0, :] # CLS token
            dropout = tf.keras.layers.Dropout(0.1)(pooled_output) # Add a dropout layer for regularization
            outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name="classifier")(dropout) # Add a multi-class classification head
            model = tf.keras.Model(inputs=input_ids, outputs=outputs) # Create the new model

            optimizer, schedule = create_optimizer(
                init_lr=1e-5,
                num_warmup_steps=0,
                num_train_steps=total_train_steps
            )

            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                optimizer=optimizer,
                metrics=['accuracy'])

            print(f"Training on fold #{n_fold+1}")

            start_time = datetime.datetime.now()

            # Fine-tune the whole model
            _ = model.fit(
                train_x,
                train_y_one_hot,
                epochs=num_epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                validation_split=0.1,
            )

            end_time = datetime.datetime.now()
            time_delta = end_time - start_time
            print(f"Training time: {time_delta}")

        # Compute metrics on current fold
        test_loss, test_acc = model.evaluate(test_x, test_y_one_hot)
        losses.append(test_loss)
        accuracies.append(test_acc)

        # Make predictions in batches to avoid memory issues
        all_predictions = []
        pred_batch_size = 16

        start_time = datetime.datetime.now()

        for i in range(0, len(test_x), pred_batch_size):
            batch_x = test_x[i:i + pred_batch_size]
            batch_predictions = model.predict(batch_x, verbose=0)
            all_predictions.extend(batch_predictions)
        predictions = np.argmax(all_predictions, axis=1) # Convert probabilities to class

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

        # Deallocate model
        del model, original_model, model_base
        tf.keras.backend.clear_session()
        gc.collect()

        n_fold += 1

    print("\n\nCross validation completed")
    print(f"Average loss: {sum(losses) / folds}")
    print(f"Final Average accuracy: {sum(accuracies) / folds}")
    print(f"Final Precisions: {precisions}")
    print(f"Final Recalls: {recalls}")
    print(f"Final F1s: {f1s}")
