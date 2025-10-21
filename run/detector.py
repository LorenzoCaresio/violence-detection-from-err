from utils import mauriziano_utils
from utils import vectors_util
from model import lstm_model
from model import bert_model
from run import compute_results

# Used to load fine-tuned BERT
from transformers import create_optimizer
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

import math
import time
import json
import os
from itertools import combinations

import numpy as np
import tensorflow as tf

LABELS = {"EX_SPOUSE_PARTNER": 0,
        "KNOWNPERSON_FRIEND": 1,
        "OTHER_RELATIONSHIP": 2,
        "RELATIVE": 3,
        "SELF": 4,
        "SPOUSE_PARTNER": 5,
        "THIEF": 6,
        "UNKNOWN_PERSON": 7,
        "UNSPECIFIED_RELATIONSHIP": 8,
        "NON-VIOLENT": 9}

v_nv = True
confidence_threshold = 0.9
mauriziano_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

def main(setting, split, mode):
    print("#"*100)
    print(setting, split, "v_nv -> " + str(v_nv))
    print("#"*100)

    mauriziano_input_directory = "../resources/data/input/mauriziano_finals/"
    input_directory = mauriziano_input_directory + split + "/"

    if "BERT" in split:
        model_filepath = "../resources/data/output/1_v_nv/models/" + setting + "_" + split + "/"
    else:
        model_filepath = "../resources/data/output/1_v_nv/models/" + setting + "_" + split + "/model.keras"

    # Create split dir if absent
    if not os.path.exists(input_directory):
        os.mkdir(input_directory)

    # ------------------------------------ #
    #           2. Build utils             #
    # ------------------------------------ #

    # If LSTM model is used, embeddings must be loaded or created
    if "BERT" not in split:
        if ("I" in split and "M" in split) or "COMP" in split:
            all_text_fp = mauriziano_input_directory + "all_text_i_m.txt"
            if not os.path.exists(all_text_fp): # Merge plain texts to train embeddings
                iss_source = "../resources/data/input/all_text.txt"
                mauriziano_source = mauriziano_input_directory + "mauriziano_all_text.txt"
                mauriziano_utils.merge_embeddings_source(iss_source, mauriziano_source, all_text_fp)
            vectors_file_path = "../resources/data/vectors/fasttext_IM_300d.bin"
        elif "M" in split:
            all_text_fp = mauriziano_input_directory + "mauriziano_all_text.txt"
            vectors_file_path = "../resources/data/vectors/fasttext_M_300d.bin"
        else:
            all_text_fp = "../resources/data/input/all_text.txt"
            vectors_file_path = "../resources/data/vectors/fasttext_I_300d.bin"

        if not os.path.exists(vectors_file_path):
            # Build utils
            vectors_util.build_vectors(all_text_fp, vectors_file_path)
        else:
            print("Skipping word embedding generation!")
            print("File " + vectors_file_path + " already exists!")

    # If the model doesn't exist yet, it must be trained
    if not os.path.exists(model_filepath):

        train_file = input_directory + "train.txt"
        train_file_labels = input_directory + "train_labels.txt"
        train_file_ids = input_directory + "train_ids.txt"

        dev_file = input_directory + "dev.txt"
        dev_file_labels = input_directory + "dev_labels.txt"
        dev_file_ids = input_directory + "dev_ids.txt"

        # ------------------------------------ #
        #        Extract training data         #
        # ------------------------------------ #

        if split == "CV-M-0.5":
            print(f"Training a model over 1000 original examples, with class ratio 0.5...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, 1000, 0.5, 100, 'w')
        elif split == "CV-M-0.8":
            print(f"Training a model over 1000 original examples, with class ratio 0.8...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, 1000, 0.8, 100, 'w')
        elif split == "D-I-M":
            iss_input_directory = "../resources/data/input/agent_classification_files/balanced_bio_V_NV/80-20/"
            mauriziano_utils.merge_original_sets(iss_input_directory, input_directory)
        elif split == "D-M0.8":
            train_set_size, _ = mauriziano_utils.compute_sets_size(mauriziano_input_directory, 1, 0.8)
            print(f"Training a model over {train_set_size} examples, with class ratio 0.8...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, train_set_size, 0.8, 100, 'w')
        elif split == "D-M0.8+V":
            mauriziano_utils.merge_human_confirmed_to_train(mauriziano_input_directory, input_directory)
            train_set_size, _ = mauriziano_utils.compute_sets_size(mauriziano_input_directory, 1, 0.8)
            print(f"Training a model over {train_set_size} examples, with class ratio 0.8...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, train_set_size, 0.8, 100, 'w')
        elif split == "D-M0.5":
            train_set_size, _ = mauriziano_utils.compute_sets_size(mauriziano_input_directory, 1, 0.5)
            print(f"Training a model over {train_set_size} examples, with class ratio 0.5...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, train_set_size, 0.5, 100, 'w')
        elif split == "D-IM0.8":
            # Merge ISS train and test sets to be used as part of train set for the current model
            iss_input_directory = "../resources/data/input/agent_classification_files/balanced_bio_V_NV/80-20/"
            mauriziano_utils.merge_original_sets(iss_input_directory, input_directory)
            train_set_size, _ = mauriziano_utils.compute_sets_size(mauriziano_input_directory, 1, 0.8)
            print(f"Training a model over {train_set_size} original examples, with class ratio 0.8...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, train_set_size, 0.8, 100, 'a')
        elif split == "D-IM0.8+V":
            # Merge ISS train and test sets to be used as part of train set for the current model
            iss_input_directory = "../resources/data/input/agent_classification_files/balanced_bio_V_NV/80-20/"
            mauriziano_utils.merge_original_sets(iss_input_directory, input_directory)
            mauriziano_utils.merge_human_confirmed_to_train(mauriziano_input_directory, input_directory)
            train_set_size, _ = mauriziano_utils.compute_sets_size(mauriziano_input_directory, 1, 0.8)
            print(f"Training a model over {train_set_size} original examples, with class ratio 0.8...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, train_set_size, 0.8, 100, 'a')
        elif split == "D-IM0.5":
            # Merge ISS train and test sets to be used as train set as part of train set for the current model
            iss_input_directory = "../resources/data/input/agent_classification_files/balanced_bio_V_NV/50-50/"
            mauriziano_utils.merge_original_sets(iss_input_directory, input_directory)
            train_set_size, _ = mauriziano_utils.compute_sets_size(mauriziano_input_directory, 1, 0.5)
            print(f"Training a model over {train_set_size} original examples, with class ratio 0.5...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, train_set_size, 0.5, 100, 'a')
        elif split == "LSTM-COMP":
            class_ratio = 0.8
            iss_input_directory = "../resources/data/input/agent_classification_files/balanced_bio_V_NV/80-20/"
            mauriziano_utils.merge_original_sets(iss_input_directory, input_directory) # Add ISS Tr+Te to curent Tr
            mauriziano_utils.merge_human_confirmed_to_train(mauriziano_input_directory, input_directory) # Add Mauriziano confirmed V to current Tr
            mauriziano_utils.merge_v_set_to_train(mauriziano_input_directory, input_directory) # Add Mauriziano V to current Tr
            with open(input_directory + "composition.json", 'r') as f: composition = json.load(f)
            v_composition = int(composition['v'])
            nv_proportion = (v_composition * class_ratio) // (1 - class_ratio)
            remaining_nv_mauriziano = int((nv_proportion // 2) - composition['gold_nv'])
            remaining_nv_iss = int((nv_proportion // 2) - composition['original_nv'])
            mauriziano_utils.merge_n_original_nv_reports(iss_input_directory, input_directory, remaining_nv_iss, 100) # Add ISS NV to current Tr
            mauriziano_utils.merge_n_nv_reports(mauriziano_input_directory, input_directory, remaining_nv_mauriziano, 100) # Add Mauriziano NV to current Tr
            mauriziano_utils.convert_txt_to_bio(input_directory + "unparsed_train.txt", input_directory + "train.txt", 'a')

        elif split == "BERT-I-M":
            iss_input_directory = "../resources/data/input/agent_classification_files/balanced_bio_V_NV/80-20/"
            mauriziano_utils.merge_original_sets(iss_input_directory, input_directory)
            _, test_set_size = mauriziano_utils.compute_sets_size(mauriziano_input_directory, 0, 0.5)
            mauriziano_utils.extract_test_set(mauriziano_input_directory, input_directory, test_set_size, 0, 0.5, 100)
        elif split == "BERT-M0.8":
            train_set_size, test_set_size = mauriziano_utils.compute_sets_size(mauriziano_input_directory, 0.8, 0.8)
            print(f"Training a model over {train_set_size} examples, with class ratio 0.8...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, train_set_size, 0.8, 100, 'w')
            mauriziano_utils.extract_test_set(mauriziano_input_directory, input_directory, test_set_size, 0.8, 0.8, 100)
        elif split == "BERT-M0.5":
            train_set_size, test_set_size = mauriziano_utils.compute_sets_size(mauriziano_input_directory, 0.8, 0.5)
            print(f"Training a model over {train_set_size} examples, with class ratio 0.5...")
            mauriziano_utils.extract_train_set(mauriziano_input_directory, input_directory, train_set_size, 0.5, 100, 'w')
            mauriziano_utils.extract_test_set(mauriziano_input_directory, input_directory, test_set_size, 0.8, 0.5, 100)
        elif split == "BERT-COMP":
            class_ratio = 0.8
            iss_input_directory = "../resources/data/input/agent_classification_files/balanced_bio_V_NV/80-20/"
            mauriziano_utils.merge_original_sets(iss_input_directory, input_directory) # Add ISS Tr+Te to curent Tr
            mauriziano_utils.merge_human_confirmed_to_train(mauriziano_input_directory, input_directory) # Add Mauriziano confirmed V to current Tr
            mauriziano_utils.merge_v_set_to_train(mauriziano_input_directory, input_directory) # Add Mauriziano V to current Tr
            with open(input_directory + "composition.json", 'r') as f: composition = json.load(f)
            v_composition = int(composition['v'])
            nv_proportion = (v_composition * class_ratio) // (1 - class_ratio)
            remaining_nv_mauriziano = int((nv_proportion // 2) - composition['gold_nv'])
            remaining_nv_iss = int((nv_proportion // 2) - composition['original_nv'])
            mauriziano_utils.merge_n_original_nv_reports(iss_input_directory, input_directory, remaining_nv_iss, 100) # Add ISS NV to current Tr
            mauriziano_utils.merge_n_nv_reports(mauriziano_input_directory, input_directory, remaining_nv_mauriziano, 100) # Add Mauriziano NV to current Tr
            mauriziano_utils.convert_txt_to_bio(input_directory + "unparsed_train.txt", input_directory + "train.txt", 'a')
        else:
            print("Specified split is not handled!")
            return;

        mauriziano_utils.create_dev_set(input_directory)

         # Load data
        train_data = load_data_bio_file(train_file, train_file_labels, train_file_ids)
        dev_data = load_data_bio_file(dev_file, dev_file_labels, dev_file_ids)

        # ------------------------------------ #
        #            Train V/NV model          #
        # ------------------------------------ #

        if "BERT" in split:
            bert_model.train_model_v_nv(train_data, dev_data, split, setting)
        else:
            vectors = vectors_util.load_vectors(vectors_file_path.replace(".bin", ".vec"))
            lstm_model.train_model_v_nv(train_data, dev_data, vectors, split, setting)
    else:
        print("Skipping model training!")
        print("File " + model_filepath + " already exists!")

    # Load model according to model type
    if "BERT" in split:
        model = TFAutoModelForSequenceClassification.from_pretrained(model_filepath)
    else:
        model = tf.keras.models.load_model(model_filepath)

    if mode == "VD": # Detect violent reports

        data_out_folder = "../resources/data/output/1_v_nv/" + setting + "_" + split + "/"
        predictions_to_verify = data_out_folder + "predictions_to_verify.txt"
        predictions_to_verify_ids = data_out_folder + "predictions_to_verify_ids.txt"
        os.makedirs(data_out_folder, exist_ok=True)

        # Wipe previous predictions if stored
        processed_example_n = 0
        nv_set_splits = mauriziano_utils.split_nv_set(mauriziano_input_directory, 10000)
        total_example_n = sum(nv_set_split.size for nv_set_split in nv_set_splits)

        for nv_set_split in nv_set_splits:

            processed_example_n += len(nv_set_split)
            print(str(processed_example_n) + " of " + str(total_example_n))

            # Extract data onto which the model will be executed
            mauriziano_utils.extract_nv_set(mauriziano_input_directory, input_directory, nv_set_split, 5)

            # Load data
            nv_set  = input_directory + "nv.txt"
            nv_set_labels = input_directory + "nv_labels.txt"
            nv_set_ids = input_directory + "nv_ids.txt"

            nv_data = load_data_bio_file(nv_set, nv_set_labels, nv_set_ids)

            # Tokenizes nv data
            if "BERT" in split:
                tokenizer = AutoTokenizer.from_pretrained(model_filepath)
                nv_sentences = [e["sentence"] for e in nv_data.values()]
                nv_tensor = tokenizer(nv_sentences, padding="max_length", return_tensors="np", truncation=True)
            else:
                vectors = vectors_util.load_vectors(vectors_file_path.replace(".bin", ".vec"))
                nv_tensor, _, _, _, _, _ = lstm_model.get_tensors2(vectors, nv_data, True)

                # Get the dim of embeddings used during training
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.Embedding):
                        embeddings_size = layer._batch_input_shape[1]
                        break

                # Adjust nv tensor size according to the one used during training
                if nv_tensor.shape[1] > embeddings_size: # Truncate sentence size
                    nv_tensor = nv_tensor[:, :embeddings_size]
                elif nv_tensor.shape[1] < embeddings_size: # Pad sentence size
                    nv_tensor = tf.keras.preprocessing.sequence.pad_sequences(nv_tensor, padding='post', maxlen=embeddings_size)

            predictions = model.predict(nv_tensor)

            # Convert to binary predictions according to model type
            if "BERT" in split:
                logits = predictions.logits
                probabilities = tf.sigmoid(logits).numpy().squeeze() # From logit to probabilities using a sigmoid
                bin_predictions = [[1] if p[1] > 0.85 else [0] for p in probabilities] # From probabilities to binary classes
            else:
                predictions = 1 / (1 + np.exp(-predictions)) # From logit to probabilities using a sigmoid
                bin_predictions = np.where(predictions > 0.85, 1, 0) # From probabilities to binary classes

            # Output current model predictions to file if current predictions is labelled as violent
            data_out_folder = "../resources/data/output/1_v_nv/" + setting + "_" + split + "/"
            os.makedirs(data_out_folder, exist_ok=True)
            pred_file = ""
            pred_ids_file = ""
            for r_id, pred in zip(nv_data, bin_predictions):
                if pred[0]: # Store predictions iff they are predicted as violent
                    pred_file += nv_data[r_id]["sentence"] + "\t" + str(pred[0]) + "\t0\n" # 0 since gold label = non-violent
                    pred_ids_file += r_id + "\n"
            open(predictions_to_verify, "a").write(pred_file)
            open(predictions_to_verify_ids, "a").write(pred_ids_file)

    elif mode == "PT": # Test model over (tagged by humans) violent reports

        mauriziano_utils.extract_v_set(mauriziano_input_directory, input_directory)

        data_out_folder = "../resources/data/output/muariziano_finals/" + setting + "_" + split + "/"

        # Load data
        v_set  = input_directory + "v.txt"
        v_set_labels = input_directory + "v_labels.txt"
        v_set_ids = input_directory + "v_ids.txt"

        v_data = load_data_bio_file(v_set, v_set_labels, v_set_ids)

        # Load tokenizer according to model type
        if "BERT" in split:
            tokenizer = AutoTokenizer.from_pretrained(model_filepath)
            v_sentences = [e["sentence"] for e in v_data.values()]
            v_tensor = tokenizer(v_sentences, padding="max_length", return_tensors="np", truncation=True)
        else:
            vectors = vectors_util.load_vectors(vectors_file_path.replace(".bin", ".vec"))
            v_tensor, _, _, _, _, _ = lstm_model.get_tensors2(vectors, v_data, True)

            # Get the dim of embeddings used during training
            embeddings_size = 0
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Embedding):
                    embeddings_size = layer._batch_input_shape[1]
                    break

            # Adjust v tensor size according to the one used during training
            if v_tensor.shape[1] > embeddings_size: # Truncate sentence size
                v_tensor = v_tensor[:, :embeddings_size]
            elif v_tensor.shape[1] < embeddings_size: # Pad sentence size
                v_tensor = tf.keras.preprocessing.sequence.pad_sequences(v_tensor, padding='post', maxlen=embeddings_size)

        print("Running the model over report tagged as violent (by humans) to test model accuracy…")
        predictions = model.predict(v_tensor)
        threshold = 0.5

        # Convert to binary predictions according to model type
        if "BERT" in split:
            logits = predictions.logits
            probabilities = tf.sigmoid(logits).numpy().squeeze() # From logit to probabilities using a sigmoid
            bin_predictions = [[1] if p[1] > threshold else [0] for p in probabilities] # From probabilities to binary classes
        else:
            predictions = 1 / (1 + np.exp(-predictions)) # From logit to probabilities using a sigmoid
            bin_predictions = np.where(predictions > threshold, 1, 0) # From probabilities to binary classes

        # Output current model predictions to file
        data_out_folder = "../resources/data/output/1_v_nv/" + setting + "_" + split + "/"
        os.makedirs(data_out_folder, exist_ok=True)
        pred_file = ""
        for r_id, pred in zip(v_data, bin_predictions):
            pred_file += v_data[r_id]["sentence"] + "\t" + str(pred[0]) + "\t1\n" # 1 since gold label = violent
        open(data_out_folder + "v_prediction.txt", "w").write(pred_file)

        # Print out results
        lines = open(data_out_folder + "v_prediction.txt").read().split("\n")
        gold_labels = [l.split("\t")[-1] for l in lines if l != ""]
        pred_labels = [l.split("\t")[-2] for l in lines if l != ""]
        gold_lines = ["index\tlabel"]
        pred_lines = ["index\tlabel"]
        for index, label in enumerate(gold_labels):
            gold_lines.append(str(index) + "\t" + str(label))
        for index, label in enumerate(pred_labels):
            pred_lines.append(str(index) + "\t" + str(label))
        open(data_out_folder + "gold.txt", 'w').write("\n".join(gold_lines))
        open(data_out_folder + "pred.txt", 'w').write("\n".join(pred_lines))
        compute_results.compute_results(data_out_folder + "gold.txt", data_out_folder + "pred.txt", "LSTM Results " + setting + " - " + split, data_out_folder + "v_results_table.txt")

def check_predictions(setting, split):
    """
        Check if model predictions are accurate by prompting the user to confirm them
    """

    confirmed_v_ids = []
    confirmed_v_labels = []
    confirmed_v_texts = []

    confirmed_nv_ids = []
    ambiguous_ids = []
    spurious_ids = []

    data_out_folder = "../resources/data/output/1_v_nv/" + setting + "_" + split + "/"
    predictions_to_verify_ids_file = data_out_folder + "predictions_to_verify_ids.txt"
    verification_out_folder = "../resources/data/output/1_v_nv/human_verification/"
    os.makedirs(data_out_folder, exist_ok=True)
    os.makedirs(verification_out_folder, exist_ok=True)

    if not os.path.isfile(predictions_to_verify_ids_file):
        print("No reports to be verified.")
    else:
        ids_to_be_verified = {r_id.strip() for r_id in open(predictions_to_verify_ids_file)}
        reports_verified = 0 # Use to track how many reports are verified in the current session
        total_reviewing_time = 0  # Use to track total reviewing time for the current session

        # Parse stored v/nv/ambiguous ids to avoid re-reviewing same reports
        if os.path.isfile(verification_out_folder + "confirmed_v_ids.txt"):
            stored_confirmed_v_ids = {r_id.strip() for r_id in open(verification_out_folder + "confirmed_v_ids.txt")}
            ids_to_be_verified = ids_to_be_verified.difference(stored_confirmed_v_ids)
        if os.path.isfile(verification_out_folder + "confirmed_nv_ids.txt"):
            stored_confirmed_nv_ids = {r_id.strip() for r_id in open(verification_out_folder + "confirmed_nv_ids.txt")}
            ids_to_be_verified = ids_to_be_verified.difference(stored_confirmed_nv_ids)
        if os.path.isfile(verification_out_folder + "ambiguous_ids.txt"):
            stored_ambiguous_ids = {r_id.strip() for r_id in open(verification_out_folder + "ambiguous_ids.txt")}
            ids_to_be_verified = ids_to_be_verified.difference(stored_ambiguous_ids)
        if os.path.isfile(verification_out_folder + "spurious_ids.txt"):
            stored_spurious_ids = {r_id.strip() for r_id in open(verification_out_folder + "spurious_ids.txt")}
            ids_to_be_verified = ids_to_be_verified.difference(stored_spurious_ids)

        # Ask the user for confirmation about model's predictions
        print("\n" + "#"*80)
        print("Conferma dell'effettiva violenza dei referti predetti come violenti")
        print("#"*80 + "\n")
        for id_tbv in ids_to_be_verified:

            r_ids = id_tbv.split("_") # Ids are composed by creation year + report id (multiple ids for composite records)
            report = mauriziano_utils.get_report_by_ids(mauriziano_input_directory, r_ids[0], r_ids[1:])

            if report:
                mauriziano_utils.visualize_report(report)
                print("Conferma violenza [s/n/x]? ")

                init_time = time.perf_counter()

                answer = input()

                if answer == 's':
                    print("Seleziona il tipo di violenza:")
                    for label, label_id in compute_results.LABELS.items():
                        if label_id < 9: print(str(label_id) + ". " + label)
                        else: print("Inserisci x per ignorare l'attuale scelta e proseguire")
                    violence_type = input()
                    if violence_type != 'x' and violence_type.isdigit() and int(violence_type) < 9:
                        confirmed_v_ids.append(id_tbv)
                        confirmed_v_labels.append(violence_type)
                        confirmed_v_texts.append(report.triage_note)
                        reports_verified += 1
                elif answer == 'n':
                    confirmed_nv_ids.append(id_tbv)
                    reports_verified += 1
                elif answer == 'x':
                    ambiguous_ids.append(id_tbv)
                else:
                    print("\nUscita forzata dal processo di conferma.\n")
                    break
            else:
               spurious_ids.append(id_tbv)

            final_time = time.perf_counter()
            total_reviewing_time += final_time - init_time

        # Store confirmed nv/v ids to be used during training in future re-training
        if len(confirmed_v_ids) > 0:
            write_mode = 'a' if os.path.isfile(verification_out_folder + "confirmed_v_ids.txt") else 'w'
            with open(verification_out_folder + "confirmed_v.txt", write_mode) as f:
                for r_sentence in confirmed_v_texts: f.write(f"{r_sentence}\n")
            with open(verification_out_folder + "confirmed_v_ids.txt", write_mode) as f:
                for r_id in confirmed_v_ids: f.write(f"{r_id}\n")
            with open(verification_out_folder + "confirmed_v_labels.txt", write_mode) as f:
                for r_label in confirmed_v_labels: f.write(f"{compute_results.ID_TO_LABEL[r_label]}\n")
        if len(confirmed_nv_ids) > 0:
            write_mode = 'a' if os.path.isfile(verification_out_folder + "confirmed_nv_ids.txt") else 'w'
            with open(verification_out_folder + "confirmed_nv_ids.txt", write_mode) as f:
                for r_id in confirmed_nv_ids: f.write(f"{r_id}\n")
        if len(ambiguous_ids) > 0:
            write_mode = 'a' if os.path.isfile(verification_out_folder + "ambiguous_ids.txt") else 'w'
            with open(verification_out_folder + "ambiguous_ids.txt", write_mode) as f:
                for r_id in ambiguous_ids: f.write(f"{r_id}\n")
        if len(spurious_ids) > 0:
            write_mode = 'a' if os.path.isfile(verification_out_folder + "spurious_ids.txt") else 'w'
            with open(verification_out_folder + "spurious_ids.txt", write_mode) as f:
                for r_id in spurious_ids: f.write(f"{r_id}\n")

        # Store reviewing stats
        if reports_verified > 0:
            write_mode = 'a' if os.path.isfile(verification_out_folder + "reviewing_stats.txt") else 'w'
            with open(verification_out_folder + "reviewing_stats.txt", write_mode) as f:
                f.write(f"{reports_verified}\t{total_reviewing_time:0.1f}\n")

        # Print reviewing stats
        print(f"Hai verificato {reports_verified} report in {math.ceil(total_reviewing_time)} secondi.")
        stats_entries = [stats_entry.strip() for stats_entry in open(verification_out_folder + "reviewing_stats.txt")]
        if len(stats_entries) > 0:
            # Each stats entry is composed as: reviewed_count<TAB>time_in_sec
            reviewed_counts, reviewing_times = zip(*(stats_entry.split("\t") for stats_entry in stats_entries))
            cumulative_reviewed_counts = sum([int(rc) for rc in reviewed_counts])
            cumulative_reviewing_time = sum([float(rt) for rt in reviewing_times])

            print(f"Cumulativamente, hai verificato {cumulative_reviewed_counts} report in {math.ceil(cumulative_reviewing_time / 60)} minuti e {math.ceil(cumulative_reviewing_time) % 60} secondi")
            print(f"Report spuri: {len(spurious_ids)}")

            confirmed_nv_count = 0
            confirmed_v_count = 0
            if os.path.isfile(verification_out_folder + "confirmed_nv_ids.txt"):
                confirmed_nv_count = len(open(verification_out_folder + "confirmed_nv_ids.txt").readlines())
            if os.path.isfile(verification_out_folder + "confirmed_v_ids.txt"):
                confirmed_v_count = len(open(verification_out_folder + "confirmed_v_ids.txt").readlines())
            print(f"\t con {confirmed_v_count} confermati violenti e {confirmed_nv_count} confermati non violenti.")


def check_predictions_overlap(settings, prompt_user=False):
    """
       Compute overlap between reports predicted as violent by several models
    """

    mauriziano_input_directory = "../resources/data/input/mauriziano/"
    data_out_folder = "../resources/data/output/1_v_nv/"
    verification_out_folder = data_out_folder + "human_verification/"
    overlap_ids = set()

    models_predictions = {}
    confirmed_to_be_violent = []

    # Retrieves the stored ids of reports predicted as violent by several models
    for directory in next(os.walk(data_out_folder))[1]:
        predictions_to_verify_ids_file = data_out_folder + directory + "/predictions_to_verify_ids.txt"
        if os.path.isfile(predictions_to_verify_ids_file):
            model_name = directory.replace(settings + '_', '')
            models_predictions[model_name] = [r_id.strip() for r_id in open(predictions_to_verify_ids_file)]

    # Retrieves the stored ids of reports confirmed to be violent by a human user
    if os.path.isfile(verification_out_folder + "confirmed_v_ids.txt"):
        stored_confirmed_v_ids = [r_id.strip() for r_id in open(verification_out_folder + "confirmed_v_ids.txt")]

    # Evaluate the models pair-wise and compute the overlap (counts and percentages)
    models = list(models_predictions.keys())
    models_couples = list(combinations(models, 2))

    for mc in models_couples:

        overlap = np.intersect1d(models_predictions[mc[0]], models_predictions[mc[1]], assume_unique=True)
        overlap_with_verified = np.intersect1d(overlap, stored_confirmed_v_ids, assume_unique=True)

        # Evaluate only overlaps where at least one model is trained over verified reports
        if "+V" in mc[0] and "+V" in mc[1]:
            overlap_ids.update(overlap)

        print("\n" + "#"*80)
        print(f"Overlap between {mc[0]} and {mc[1]}:")
        print(f"\t{mc[0]} size: {len(models_predictions[mc[0]])}")
        print(f"\t{mc[1]} size: {len(models_predictions[mc[1]])}")
        print(f"\tOverlap size: {len(overlap)}")
        print(f"\tOverlap with verified size: {len(overlap_with_verified)}")

    if prompt_user:

        confirmed_v_ids = []
        confirmed_v_labels = []
        confirmed_nv_ids = []
        ambiguous_ids = []

        reports_verified = 0 # Use to track how many reports are verified in the current session
        total_reviewing_time = 0  # Use to track total reviewing time for the current session

        # Parse stored v/nv/ambiguous ids to avoid re-reviewing same reports
        if os.path.isfile(verification_out_folder + "confirmed_v_ids.txt"):
            stored_confirmed_v_ids = {r_id.strip() for r_id in open(verification_out_folder + "confirmed_v_ids.txt")}
            overlap_ids = overlap_ids.difference(stored_confirmed_v_ids)
        if os.path.isfile(verification_out_folder + "confirmed_nv_ids.txt"):
            stored_confirmed_nv_ids = {r_id.strip() for r_id in open(verification_out_folder + "confirmed_nv_ids.txt")}
            overlap_ids = overlap_ids.difference(stored_confirmed_nv_ids)
        if os.path.isfile(verification_out_folder + "ambiguous_ids.txt"):
            stored_ambiguous_ids = {r_id.strip() for r_id in open(verification_out_folder + "ambiguous_ids.txt")}
            overlap_ids = overlap_ids.difference(stored_ambiguous_ids)

        # Ask the user for confirmation about model's predictions
        print("\n" + "#"*80)
        print("Conferma dell'effettiva violenza dei referti predetti come violenti da diversi modelli")
        print("#"*80 + "\n")
        for id_tbv in overlap_ids:
            r_ids = id_tbv.split("_") # Ids are composed by creation year + report id (multiple ids for composite records)
            report = mauriziano_utils.get_report_by_ids(mauriziano_input_directory, r_ids[0], r_ids[1:])
            mauriziano_utils.visualize_report(report)
            print("Conferma violenza [s/n/x]? ")
            init_time = time.perf_counter()
            answer = input()
            if answer == 's':
                print("Seleziona il tipo di violenza:")
                for label, label_id in compute_results.LABELS.items():
                    if label_id < 9: print(str(label_id) + ". " + label)
                    else: print("Inserisci x per ignorare l'attuale scelta e proseguire")
                violence_type = input()
                if violence_type != 'x' and violence_type.isdigit() and int(violence_type) < 9:
                    confirmed_v_ids.append(id_tbv)
                    confirmed_v_labels.append(violence_type)
                    reports_verified += 1
            elif answer == 'n':
                confirmed_nv_ids.append(id_tbv)
                reports_verified += 1
            elif answer == 'x':
                ambiguous_ids.append(id_tbv)
            else:
                print("\nUscita forzata dal processo di conferma.\n")
                break

            final_time = time.perf_counter()
            total_reviewing_time += final_time - init_time

        # Store confirmed nv/v ids to be used during training in future re-training
        if len(confirmed_v_ids) > 0:
            write_mode = 'a' if os.path.isfile(verification_out_folder + "confirmed_v_ids.txt") else 'w'
            with open(verification_out_folder + "confirmed_v_ids.txt", write_mode) as f:
                for r_id in confirmed_v_ids: f.write(f"{r_id}\n")
            with open(verification_out_folder + "confirmed_v_labels.txt", write_mode) as f:
                for r_label in confirmed_v_labels: f.write(f"{compute_results.ID_TO_LABEL[r_label]}\n")
        if len(confirmed_nv_ids) > 0:
            write_mode = 'a' if os.path.isfile(verification_out_folder + "confirmed_nv_ids.txt") else 'w'
            with open(verification_out_folder + "confirmed_nv_ids.txt", write_mode) as f:
                for r_id in confirmed_nv_ids: f.write(f"{r_id}\n")
        if len(ambiguous_ids) > 0:
            write_mode = 'a' if os.path.isfile(verification_out_folder + "ambiguous_ids.txt") else 'w'
            with open(verification_out_folder + "ambiguous_ids.txt", write_mode) as f:
                for r_id in ambiguous_ids: f.write(f"{r_id}\n")

def compose_gold_set(set_size, length_threshold):
    """
        Compose a gold dataset by prompting the user on human verified non-violent
        reports asking about their overall quality
    """

    nv_ids_to_be_verified = []
    good_nv_ids = []
    good_nv_lines = []
    bad_nv_ids = []

    mauriziano_input_directory = "../resources/data/input/mauriziano/"
    verification_out_folder = "../resources/data/output/1_v_nv/human_verification/"

    if os.path.isfile(verification_out_folder + "confirmed_nv_ids.txt"):
        nv_ids_to_be_verified = {r_id.strip() for r_id in open(verification_out_folder + "confirmed_nv_ids.txt")}

    # Parse stored good/bad nv ids to avoid re-reviewing same reports
    if os.path.isfile(verification_out_folder + "good_nv_ids.txt"):
        stored_good_nv_ids = {r_id.strip() for r_id in open(verification_out_folder + "good_nv_ids.txt")}
        nv_ids_to_be_verified = nv_ids_to_be_verified.difference(stored_good_nv_ids)
    if os.path.isfile(verification_out_folder + "bad_nv_ids.txt"):
        stored_bad_nv_ids = {r_id.strip() for r_id in open(verification_out_folder + "bad_nv_ids.txt")}
        nv_ids_to_be_verified = nv_ids_to_be_verified.difference(stored_bad_nv_ids)

    # Ask the user for confirmation about nv report quality
    print("\n" + "#"*80)
    print("Gold dataset creation")
    print("#"*80 + "\n")
    for cid in nv_ids_to_be_verified:
        r_ids = cid.split("_") # Ids are composed by creation year + report id (multiple ids for composite records)
        report = mauriziano_utils.get_report_by_ids(mauriziano_input_directory, r_ids[0], r_ids[1:])
        if len(report.triage_note) > length_threshold:
            mauriziano_utils.visualize_report(report)
            print("Include this report in the gold dataset? [y/n]")
            answer = input()
            if answer == 'y':
                good_nv_ids.append(cid)
                good_nv_lines.append(report.triage_note)
            elif answer == 'n':
                bad_nv_ids.append(cid)
            else:
                print("\nForced exit from the gold set creation process.\n")
                break

    # Store confirmed good/bad nv ids to file
    if len(good_nv_ids) > 0:
        write_mode = 'a' if os.path.isfile(verification_out_folder + "good_nv_ids.txt") else 'w'
        with open(verification_out_folder + "good_nv_ids.txt", write_mode) as f:
            for r_id in good_nv_ids: f.write(f"{r_id}\n")
        write_mode = 'a' if os.path.isfile(verification_out_folder + "good_nv.txt") else 'w'
        with open(verification_out_folder + "good_nv.txt", write_mode) as f:
            for line in good_nv_lines: f.write(f"{line}\n")
    if len(bad_nv_ids) > 0:
        write_mode = 'a' if os.path.isfile(verification_out_folder + "bad_nv_ids.txt") else 'w'
        with open(verification_out_folder + "bad_nv_ids.txt", write_mode) as f:
            for r_id in bad_nv_ids: f.write(f"{r_id}\n")

def load_data_bio_file(file_path, labels_file_path, ids_file_path):
    data = {}
    labels = [str(compute_results.LABELS[l]) for l in open(labels_file_path).read().split("\n") if l != ""]
    ids = [l for l in open(ids_file_path).read().split("\n") if l != ""]
    lines = [l for l in open(file_path).read().split("\n\n") if l != ""]

    print(len(labels), len(ids), len(lines))
    assert len(labels) == len(ids) == len(lines)

    for i,l in enumerate(lines):
        sentence = " ".join([fields.split(" ")[0] for fields in l.split("\n")])
        label = labels[i]
        data[ids[i]] = {"sentence": sentence.replace("[id] -> ", ""), "label": label}
    return data

def load_data_text_file(file_path, ids_file_path):
    data = {}
    ids = [l for l in open(ids_file_path).read().split("\n") if l != ""]
    lines = [l for l in open(file_path).read().split("\n") if l != ""]

    assert len(ids) == len(lines)

    for i,l in enumerate(lines):
        data[ids[i]] = {"sentence": l, "label": str(LABELS["NON-VIOLENT"])}
    return data

if __name__ == '__main__':
    """
    Execute model over several given splits.

    Available splits for detection:
        LSTM:
            D-I-M: training set composed by ISS's data only
            D-M0.5: training set composed by Mauriziano's data only with class ratio 0.5
            D-M0.8: training set composed by Mauriziano's data only with class ratio 0.8
            D-M0.8+V: training set composed by Mauriziano's data only with class ratio 0.8 + human verified reports
            D-IM0.5: training set composed by ISS+Mauriziano's data with class ratio 0.5
            D-IM0.8: training set composed by ISS+Mauriziano's data with class ratio 0.8
            D-IM0.8+V: training set composed by ISS+Mauriziano's data with class ratio 0.8 + human verified reports
            D-IM0.8: training set composed by ISS+Mauriziano's data with class ratio 0.8 + human verified reports
            LSTM-COMP: composite dataset with class ratio 0.8
        BERT:
            BERT-M0.5: training set composed by Mauriziano's data only with class ratio 0.5
            BERT-M0.8: training set composed by Mauriziano's data only with class ratio 0.8
            BERT-COMP: composite dataset with class ratio 0.8

    Available modes:
        VD: run the model over examples reported as non-violent and prompt the user
            to check model predictions (with over 90% confidence).
        PT: run the model over examples positively reported as violent (which should
            be properly tagged).
        CP: prompt the user to check if model predictions (over non-violent reports)
            are accurate.
        CO: compute overlap between reports predicted as violent by several models
            and optionally prompt the user about it
        GC: compose gold dataset by prompting the user on human verified non-violent
            reports
    """

    mauriziano_input_directory = "../resources/data/input/mauriziano_finals/"

    # General settings
    settings = ["balanced_bio_V_NV"]
    splits = ["BERT-COMP"]
    mode = "VD"

    match mode:
        case "VD":
            for setting in settings:
                for split in splits:
                    main(setting, split, mode)
        case "CP":
            for setting in settings:
                for split in splits:
                    check_predictions(setting, split)
        case "CO":
            for setting in settings:
                check_predictions_overlap(setting, True)
        case "GC":
            compose_gold_set(100, 100)
        case _:
            for setting in settings:
                for split in splits:
                    main(setting, split, mode)