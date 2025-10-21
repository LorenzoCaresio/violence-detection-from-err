from utils import vectors_util
from model import llm_model
from run import compute_results

import os

# Comment following when running LLM experiments (vd-llm env)
#from model import bert_model
#from model import lstm_model

heads_finetuning = False
violent_only = True

def main(setting, split, folds):
    print("#"*100)
    print(setting, split, "V-NV")
    print("#"*100)

    input_directory = "../resources/data/input/metroxrai_experiments/"
    output_directory = "../resources/data/output/metroxrai_experiments/"

    data_file = input_directory + "train.txt"
    data_file_labels = input_directory + "train_labels.txt"
    data_file_ids = input_directory + "train_ids.txt"

    # Load data
    data = load_data_bio_file(data_file, data_file_labels, data_file_ids)

    if "LSTM" in split:
        all_text_fp = input_directory + "all_text_i_m.txt"
        vectors_file_path = "../resources/data/vectors/fasttext_IM_300d.bin"
        if not os.path.exists(vectors_file_path):
            vectors_util.build_vectors(all_text_fp, vectors_file_path)
        else:
            print("Skipping word embedding generation!")
            print("File " + vectors_file_path + " already exists!")
        vectors = vectors_util.load_vectors(vectors_file_path.replace(".bin", ".vec"))
        lstm_model.cv_run_model_v_nv(folds, data, vectors, split, setting)
    elif "BERT" in split:
        bert_model.cv_run_model_v_nv(folds, data, split, setting)
    elif "LLM" in split:
        llm_model.cv_run_model_v_nv(folds, data, split, setting, output_directory)

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

if __name__ == '__main__':
    """
    Execute CV over a (given) multi-class dataset.

    Available splits in cross-validation:
        CV-LSTM-COMP-BIN: composite dataset with class ratio 0.8, V-NV classification
        CV-CNN+LSTM-COMP-BIN: composite dataset with class ratio 0.8, V-NV classification
        CV-BERTino-COMP-BIN: composite dataset with class ratio 0.8, V-NV classification
        CV-BERT-BASE-COMP-BIN: composite dataset with class ratio 0.8, V-NV classification
        CV-LLM-BIN: composite dataset with class ratio 0.8, V-NV classification
    """

    folds = 10

    setting = "balanced_bio_V_NV"
    split = "CV-LLM-COMP-BIN"

    main(setting, split, folds)