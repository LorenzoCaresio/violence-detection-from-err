from run import compute_results
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_cpp import Llama

import os
import gc
import tqdm
import random
import datetime
import torch
import multiprocessing

import numpy as np

model_name = 'Llama-3.2-3B-Instruct-Q4_K_M.gguf'
model_path_dir = '../resources/data/models/'

label2idViolent = {
        "EX_SPOUSE_PARTNER": 0,
        "KNOWNPERSON_FRIEND": 1,
        "OTHER_RELATIONSHIP": 2,
        "RELATIVE": 3,
        "SPOUSE_PARTNER": 4,
        "THIEF": 5,
        "UNKNOWN_PERSON": 6,
        "UNSPECIFIED_RELATIONSHIP": 7}

prompt = """
<|system|>
You are a BINARY CLASSIFIER used to verify if emergency room reports are caused by VIOLENT events. Do not mark traumatic but NON-VIOLENT reports (e.g. car accidents) as violent.

The text to classify will be in Italian.

Your response should be a SINGLE WORD LABEL:
- VIOLENT: for reports caused by violent acts (assaults, thefts, fights, self-inflicted violence).
- NONVIOLENT: for other types of reports

Default to NONVIOLENT if you are not confident enough.

EXAMPLES OF VIOLENT REPORTS:
- "Riferisce percosse da parte della figlia durante litigio familiare avvenuto in casa. Si evidenzia ematoma avambraccio dx."
- "Ennesimo episodio di aggressione da parte del compagno. Trauma cranico, e trauma arti."
- "RIF ABUSO ETILICO, IN SEGUITO LITE E COLLUTAZIONE CON PERSONA NON NOTA CON TRAUMA CRANICO FACCIALE E CONTUSIONI DIFFUSE. IN TRIAGE VOMITO."

EXAMPLES OF NON-VIOLENT REPORTS:
- "Riferisce: 'Era a lavoro, mentre puliva si ferisce il polso sx. No copertura antitetanica.'. Medicata in triage."
- "Trauma 5 gg fa contro uno spigolo. Deambula con zoppia. Recente diagnosi di Ca mammario in attesa di RT."
- "Riferisce cardiopalmo mentre si recava in PS, ora regredito. Eseguito ECG e fatto visionare, presenza di RS. APR: pregressa ablazione cardiaca."

HERE IS THE REPORT TO CLASSIFY:

"{report}"

Remember to respond EXCLUSIVELY with a SINGLE WORD LABEL (NONVIOLENT or VIOLENT), nothing else.

<|assistant|>
"""

prompt_multi = """
<|system|>
You are a specialized CLASSIFIER for emergency room reports involving VIOLENCE. Your task is to identify the relationship between the victim and perpetrator.

CLASSIFICATION LABELS:
- SPOUSE_PARTNER: Current husband/wife/partner of the victim
- EX_SPOUSE_PARTNER: Former husband/wife/partner of the victim
- RELATIVE: Family member of the victim
- KNOWNPERSON_FRIEND: Friend or acquaintance of the victim (*persona nota* in italian)
- OTHER_RELATIONSHIP: Other specific relationship (e.g. neighbor, employer, employee, customers)
- THIEF: Theft-related perpetrator, burglars, etc
- UNKNOWN_PERSON: Stranger or unidentified person
- UNSPECIFIED_RELATIONSHIP: if the ER report doesn't specify the mechanics of the violent act

IMPORTANT RULES:
- Select EXACTLY ONE label from the list above
- Default to UNSPECIFIED_RELATIONSHIP if you are not confident enough
- Look for specific relationship terms like "marito", "ex moglie", "padre", "persona nota", etc.

CLASSIFICATION TASK:
Report to analyze: "{report}"

Your response must be ONLY ONE of these labels:
SPOUSE_PARTNER, EX_SPOUSE_PARTNER, KNOWNPERSON_FRIEND, RELATIVE, OTHER_RELATIONSHIP, THIEF, UNKNOWN_PERSON, UNSPECIFIED_RELATIONSHIP

NO PREAMBLE, just provide the label.

<|assistant|>
"""

prompt_rewrite = """You will be given the triage note of an emergency room report. You have to REWRITE it in a more comprehensible way.

The text to rewrite will be in Italian and your OUTPUT must be in ITALIAN too.

Remember to RETAIN ALL THE INFORMATION and DON'T ADD ANY.

Keep in mind that the DOMAIN of these texts is the MEDICAL DOMAIN.

DO NOT BE VERBOSE: the length of the rewritten text SHOULD NOT EXCEED 10-15% more or less than the length of the original text. In general DON'T EXCEED 400 CHARACTERS.

HERE IS THE TEXT TO REWRITE IN A MORE COMPREHENSIBLE WAY:

"{report}"

Remember to respond with the REWRITTEN TEXT ONLY, nothing else, NO PREAMBLE of any sort.
"""

def run_model_v_nv(data, split, setting):
    """
        Run (test-only) a LLM to perform binary classification on given data
    """

    # Load model and tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Create a text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        min_new_tokens=1
    )

    # Prepare dataset to be used to test the model
    data_texts = np.array([report['sentence'] for _, report in data.items()])
    data_labels = np.array([report['label'] for _, report in data.items()])
    data_labels = np.where(data_labels == '9', 0, 1) # From multi-label to binary category

    start_time = datetime.datetime.now()

    # Get model predictions
    y_pred = []
    for record in tqdm.tqdm(data_texts):
        pred = ""
        while pred != "VIOLENT" and pred != "NONVIOLENT":
            current_prompt = prompt.format(report=record)
            response = generator(current_prompt, return_full_text=False)[0]['generated_text']
            print(response)
            #pred = response[len(current_prompt):].strip() # Remove the prompt
            pred = response.strip()
        bin_pred = 0 if pred == "NONVIOLENT" else 1
        y_pred.append(bin_pred)

    end_time = datetime.datetime.now()
    time_delta = end_time - start_time

    print(f"Total execution time: {time_delta}")

    print(classification_report(data_labels, y_pred))

    print("-"*50)
    print("Counts")
    counts = Counter(data_labels)
    for c in counts:
        print(c, counts[c])

    data_out_folder = "../resources/data/output/1_v_nv/" + setting + "_" + split + "/"
    os.makedirs(data_out_folder, exist_ok=True)

    # Store predictions
    pred_file = ""
    for i, text in enumerate(data_texts):
        pred_file += text + "\t" + str(y_pred[i]) + "\t" + str(data_labels[i]) + "\n"

    open(data_out_folder + "test_prediction.txt", "w").write(pred_file)

    prediction_ids = []
    nv_prediction_ids = []

    for i, text in enumerate(data_texts):
        sentence = data_texts[i]
        label = y_pred[i]

        if label > 0:
            for entry_id in data:
                if data[entry_id]['sentence'] == sentence:
                    prediction_ids.append(entry_id + "\t" + data[entry_id]["label"])
        else:
            for entry_id in data:
                if data[entry_id]['sentence'] == sentence:
                    nv_prediction_ids.append(entry_id + "\t" + data[entry_id]["label"])

    open(data_out_folder + "violent_ids.txt", "w").write("\n".join(prediction_ids))
    open(data_out_folder + "non_violent_ids.txt", "w").write("\n".join(nv_prediction_ids))

def run_model_v_nv_with_rewrite(data, n_rewrite, split, setting):
    """
        Run (test-only) a LLM to perform binary classification on given data by
        taking majority voting on n rewritten description (rewritten by the LLM).
        This technique is a refinement that we didn't include in the paper.
    """

    # Store how many times the prediction over the original triage note
    # is different to the prediction over majority vote
    prediction_mismatch = 0
    mismatches_with_right_predictions = 0

    # Prepare dataset to be used to test the model
    data_texts = np.array([report['sentence'] for _, report in data.items()][:250])
    data_labels = np.array([report['label'] for _, report in data.items()][:250])
    data_labels = np.where(data_labels == '9', 0, 1) # From multi-label to binary category

    # Get model predictions
    y_pred = []
    for (label, record) in tqdm.tqdm(zip(data_labels, data_texts)):

        # Predict over the original triage note
        pred = ""
        while pred != "VIOLENT" and pred != "NONVIOLENT":
            current_prompt = prompt.format(report=record)
            response: GenerateResponse = generate(model=model, prompt=current_prompt)
            pred = response.response.strip()
        original_bin_pred = 0 if pred == "NONVIOLENT" else 1

        # Rewrite the triage note several times, and predict violence over these rewrites
        rewritten_texts = []

        for _ in range(n_rewrite):
            current_prompt = prompt_rewrite.format(report=record)
            response: GenerateResponse = generate(model=model, prompt=current_prompt)
            rewrite = response.response
            rewritten_texts.append(rewrite)

        # Predict over rewritten text and then take majority voting
        current_predictions = []
        pred = ""
        for rewritten_text in rewritten_texts:
            while pred != "VIOLENT" and pred != "NONVIOLENT":
                current_prompt = prompt.format(report=rewritten_text)
                response: GenerateResponse = generate(model=model, prompt=current_prompt)
                pred = response.response.strip()
            current_bin_prediction = 0 if pred == "NONVIOLENT" else 1
            current_predictions.append(current_bin_prediction)

        # Majority voting
        counts = Counter(current_predictions)
        bin_pred = counts.most_common(1)[0][0]
        y_pred.append(bin_pred)

        # Increase prediction mismatch if the original prediction != prediction on rewritten text
        if original_bin_pred != bin_pred:
            prediction_mismatch += 1
            if bin_pred == label:
                mismatches_with_right_predictions += 1

    print("\nPrediction mismatch: " + str(prediction_mismatch) + "\n")
    print("\nMismatch with right predictions: " + str(mismatches_with_right_predictions) + "\n")

    print(classification_report(data_labels, y_pred))

    print("-"*50)
    print("Counts")
    counts = Counter(data_labels)
    for c in counts:
        print(c, counts[c])

    data_out_folder = "../resources/data/output/1_v_nv/" + setting + "_" + split + "/"
    os.makedirs(data_out_folder, exist_ok=True)

    # Store predictions
    pred_file = ""
    for i, text in enumerate(data_texts):
        pred_file += text + "\t" + str(y_pred[i]) + "\t" + str(data_labels[i]) + "\n"

    open(data_out_folder + "test_prediction.txt", "w").write(pred_file)

    prediction_ids = []
    nv_prediction_ids = []

    for i, text in enumerate(data_texts):
        sentence = data_texts[i]
        label = y_pred[i]

        if label > 0:
            for entry_id in data:
                if data[entry_id]['sentence'] == sentence:
                    prediction_ids.append(entry_id + "\t" + data[entry_id]["label"])
        else:
            for entry_id in data:
                if data[entry_id]['sentence'] == sentence:
                    nv_prediction_ids.append(entry_id + "\t" + data[entry_id]["label"])

def cv_run_model_v_nv(folds, data, split, setting, output_dir):
    """
        Run the model using n-fold Cross-validation.
        This is used in paper's experiments, and SHOULD be considered for replication.
    """

    # Set seed for reproducibility
    np.random.seed(92)
    random.seed(92)

    accuracies = []
    nv_precisions = []
    v_precisions = []
    nv_recalls = []
    v_recalls = []
    nv_f1s = []
    v_f1s = []
    class_accuracies = []

    # Check if a file with previous misclassifcations for the current model exists, and if so delete it
    misclassification_file_dir = output_dir + "binclass_misclassification/" + split
    os.makedirs(misclassification_file_dir, exist_ok=True)
    if os.path.isfile(misclassification_file_dir + "/ids.txt"): os.remove(misclassification_file_dir + "/ids.txt")
    if os.path.isfile(misclassification_file_dir + "/labels.txt"): os.remove(misclassification_file_dir + "/labels.txt")
    if os.path.isfile(misclassification_file_dir + "/texts.txt"): os.remove(misclassification_file_dir + "/texts.txt")

    if "llama" in model_name.lower(): # Llama-specific

        # if Verify model file exists
        if not os.path.exists(model_path_dir + model_name):
            raise FileNotFoundError(f"Model not found at the specified path: {model_path}. Download it first using huggingface-cli")

        num_physical_cores = multiprocessing.cpu_count()
        optimal_threads = max(1, num_physical_cores - 1) # -1 to avoid saturating the cpu

        model = Llama(
            model_path=model_path_dir + model_name,
            n_ctx=2048,
            n_batch=512,
            n_threads=optimal_threads,
            n_gpu_layers=0,
            verbose=False,
            seed=92
        )

    else: # HuggingFace

        # Load model and tokenizer from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16)

        # Create a text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            min_new_tokens=1,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # Prepare dataset to be used to test the model
    data_texts = np.array([report['sentence'] for _, report in data.items()])
    data_labels = np.array([report['label'] for _, report in data.items()])
    data_labels = np.where(data_labels == '9', 0, 1) # From multi-label to binary category

    # Use sklearn to performe n-fold CV
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=92)

    n_fold = 0

    # Compute folds indices (when seed is set explicitly, these should be deterministic)
    fold_indices = []
    for train_idx, test_idx in kfold.split(data_texts, data_labels):
        fold_indices.append((train_idx, test_idx))

    for train, test in fold_indices:

        test_x = data_texts[test_idx]
        test_y = data_labels[test_idx]

        start_time = datetime.datetime.now()

        bin_predictions = []
        for _, record in enumerate(tqdm.tqdm(test_x)):

            tries = 0
            pred = ""

            while pred != "VIOLENT" and pred != "NONVIOLENT" and tries <= 5:

                current_prompt = prompt.format(report=record)

                if "llama" in model_name.lower(): # Llama-specific handling
                    response = model(
                        current_prompt,
                        max_tokens=15,
                        temperature=0.1,
                        stop=[".", "\n", " ", ","],
                        echo=False)

                    pred = response["choices"][0]["text"].strip().upper()

                else: # HuggingFace Transformer
                    response = generator(current_prompt,
                                return_full_text=False,
                                pad_token_id=generator.tokenizer.eos_token_id
                            )[0]['generated_text']

                    pred = response.strip().upper()

                tries += 1

            if tries > 5:
                pred = "NONVIOLENT" # Used to avoid loops

            bin_pred = 0 if pred == "NONVIOLENT" else 1
            bin_predictions.append(bin_pred)

        end_time = datetime.datetime.now()
        time_delta = end_time - start_time
        print(f"Prediction time: {time_delta}")

        # Precision
        precision = precision_score(test_y, bin_predictions, average=None, zero_division=np.nan)
        nv_precisions.append(float(precision[0]))
        v_precisions.append(float(precision[1]))

        # Recall
        recall = recall_score(test_y, bin_predictions, average=None, zero_division=np.nan)
        nv_recalls.append(float(recall[0]))
        v_recalls.append(float(recall[1]))

        # F1
        f1 = f1_score(test_y, bin_predictions, average=None, zero_division=np.nan)
        nv_f1s.append(float(f1[0]))
        v_f1s.append(float(f1[1]))

        # Class accuracy
        accuracy = accuracy_score(test_y, bin_predictions)
        class_accuracies.append(float(accuracy))

        print(f"NV precisions: {nv_precisions}")
        print(f"V precisions: {v_precisions}")
        print(f"NV Recalls: {nv_recalls}")
        print(f"V Recalls: {v_recalls}")
        print(f"NV F1s: {nv_f1s}")
        print(f"V F1s: {v_f1s}")
        print(f"Class accuracy: {class_accuracies}")

        # Store wrong predictions for later analysis
        wrong_prediction_ids = []
        wrong_prediction_labels = []
        wrong_prediction_texts = []

        write_mode = 'a' if os.path.isfile(misclassification_file_dir + "/ids.txt") else 'w'

        for entry_id, test_text, test_label, prediction in zip(test, test_x, test_y, bin_predictions):
            if test_label != prediction:
                wrong_prediction_ids.append(entry_id)
                wrong_prediction_labels.append(f"{test_label}\t{prediction}")
                wrong_prediction_texts.append(test_text)
        with open(misclassification_file_dir + "/ids.txt", write_mode) as misclassification_file:
            for entry_id in wrong_prediction_ids: misclassification_file.write(f"{entry_id}\n")
        with open(misclassification_file_dir + "/labels.txt", write_mode) as misclassification_file:
            for label in wrong_prediction_labels: misclassification_file.write(f"{label}\n")
        with open(misclassification_file_dir + "/texts.txt", write_mode) as misclassification_file:
            for text in wrong_prediction_texts: misclassification_file.write(f"{text}\n")

        gc.collect()

        n_fold += 1

    print("\n Cross validation completed")
    print(f"NV precisions: {nv_precisions}")
    print(f"V precisions: {v_precisions}")
    print(f"NV Recalls: {nv_recalls}")
    print(f"V Recalls: {v_recalls}")
    print(f"NV F1s: {nv_f1s}")
    print(f"V F1s: {v_f1s}")
    print(f"Class accuracy: {class_accuracies}")

def cv_run_model_all(folds, data, split, setting, output_dir):
    """
        Run the model using n-fold Cross-validation for multiclass classification.
        This is used in paper's experiments, and SHOULD be considered for replication.
    """

    # Set seed for reproducibility
    np.random.seed(92)
    random.seed(92)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # Check if a file with previous misclassifcations for the current model exists, and if so delete it
    misclassification_file_dir = output_dir + "multiclass_misclassification/" + split
    os.makedirs(misclassification_file_dir, exist_ok=True)
    if os.path.isfile(misclassification_file_dir + "/ids.txt"): os.remove(misclassification_file_dir + "/ids.txt")
    if os.path.isfile(misclassification_file_dir + "/labels.txt"): os.remove(misclassification_file_dir + "/labels.txt")
    if os.path.isfile(misclassification_file_dir + "/texts.txt"): os.remove(misclassification_file_dir + "/texts.txt")

    if "llama" in model_name.lower(): # Llama-specific

        # if Verify model file exists
        if not os.path.exists(model_path_dir + model_name):
            raise FileNotFoundError(f"Model not found at the specified path: {model_path}. Download it first using huggingface-cli")

        num_physical_cores = multiprocessing.cpu_count()
        optimal_threads = max(1, num_physical_cores - 1) # -1 to avoid saturating the cpu

        model = Llama(
            model_path=model_path_dir + model_name,
            n_ctx=2048,
            n_batch=512,
            n_threads=optimal_threads,
            n_gpu_layers=0,
            verbose=False,
            seed=92
        )

    else: # HuggingFace

        # Load model and tokenizer from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16)

        # Create a text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            min_new_tokens=1,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # Prepare dataset to be used to test the model
    data_texts = np.array([report['sentence'] for _, report in data.items()])
    data_labels = np.array([int(report['label']) for _, report in data.items()])

    # Use sklearn to performe n-fold CV
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=92)

    n_fold = 0

    # Compute folds indices (when seed is set explicitly, these should be deterministic)
    fold_indices = []
    for train_idx, test_idx in kfold.split(data_texts, data_labels):
        fold_indices.append((train_idx, test_idx))

    for train, test in fold_indices:

        test_x = data_texts[test_idx]
        test_y = data_labels[test_idx]

        start_time = datetime.datetime.now()

        predictions = []

        print(f"Training on fold #{n_fold+1}")

        for _, record in enumerate(tqdm.tqdm(test_x)):

            tries = 0
            pred = ""

            while pred not in label2idViolent.keys() and tries <= 5:

                current_prompt = prompt_multi.format(report=record)

                if "llama" in model_name.lower(): # Llama-specific handling
                    response = model(
                        current_prompt,
                        max_tokens=20,
                        temperature=0.1,
                        stop=[".", "\n", " ", ","],
                        echo=False)

                    pred = response["choices"][0]["text"].strip().upper()

                else: # HuggingFace Transformer
                    response = generator(current_prompt,
                                return_full_text=False,
                                pad_token_id=generator.tokenizer.eos_token_id
                            )[0]['generated_text']

                    pred = response.strip().upper()

                tries += 1

            if pred in label2idViolent.keys():
                final_pred = label2idViolent[pred]
            else:
                final_pred = 7 # Fallback label

            predictions.append(final_pred)

        end_time = datetime.datetime.now()
        time_delta = end_time - start_time
        print(f"Prediction time: {time_delta}")

        accuracy = accuracy_score(test_y, predictions)
        accuracies.append(float(accuracy))
        precision = precision_score(test_y, predictions, average=None, zero_division=np.nan)
        precision = [float(p) for p in precision]
        precisions.append(precision)
        recall = recall_score(test_y, predictions, average=None, zero_division=np.nan)
        recall = [float(r) for r in recall]
        recalls.append(recall)
        f1 = f1_score(test_y, predictions, average=None, zero_division=np.nan)
        f1 = [float(f) for f in f1]
        f1s.append(f1)

        print(f"Accuracy: {accuracies}")
        print(f"Precisions: {precisions}")
        print(f"Recalls: {recalls}")
        print(f"F1s: {f1s}")

        # Store wrong predictions for later analysis
        wrong_prediction_ids = []
        wrong_prediction_labels = []
        wrong_prediction_texts = []

        write_mode = 'a' if os.path.isfile(misclassification_file_dir + "/ids.txt") else 'w'

        for entry_id, test_text, test_label, prediction in zip(test, test_x, test_y, predictions):
            if test_label != prediction:
                wrong_prediction_ids.append(entry_id)
                wrong_prediction_labels.append(f"{test_label}\t{prediction}")
                wrong_prediction_texts.append(test_text)
        with open(misclassification_file_dir + "/ids.txt", write_mode) as misclassification_file:
            for entry_id in wrong_prediction_ids: misclassification_file.write(f"{entry_id}\n")
        with open(misclassification_file_dir + "/labels.txt", write_mode) as misclassification_file:
            for label in wrong_prediction_labels: misclassification_file.write(f"{label}\n")
        with open(misclassification_file_dir + "/texts.txt", write_mode) as misclassification_file:
            for text in wrong_prediction_texts: misclassification_file.write(f"{text}\n")

        gc.collect()

        n_fold += 1

    print("\n\nCross validation completed")
    print(f"Final Average accuracy: {sum(accuracies) / folds}")
    print(f"Final Precisions: {precisions}")
    print(f"Final Recalls: {recalls}")
    print(f"Final F1s: {f1s}")
