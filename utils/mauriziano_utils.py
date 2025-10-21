import linecache
import json
import os
import random
import textwrap
import time
import math

from datetime import datetime
import numpy as np
from nltk.tokenize import word_tokenize

random.seed(92)

class Report:

    def __init__(self, r_id, year, age, gender, violence, triage_note, entrance_visit):
        self.r_ids = [r_id]
        self.year = year
        self.age = age
        self.gender = gender
        self.violence = violence
        self.triage_note = triage_note
        self.entrance_visit = entrance_visit

    def __eq__(self, other_report):
        if isinstance(other_report, self.__class__):
            return (self.age == other_report.age and
                   self.gender == other_report.gender and
                   self.entrance_visit == other_report.entrance_visit)
        return False

def get_report_by_ids(input_directory, year, ids):
    """
        Get a complete report given its ids and its creation year.
    """

    json_file_path = input_directory + "mauriziano_" + str(year) + ".json"
    with open(json_file_path, 'r') as f:
        input_data = json.load(f)

    if len(ids) > 0:
        # Get first id in the list and compose a Report object using stored data
        if ids[0] in input_data.keys():
            report = input_data[ids[0]]
            current_violence = 0 if report["violence"] == "NULL" else int(report["violence"]) # Violence can be "NULL" too
            current_report = Report(ids[0], year, report["age"], report["gender"], current_violence, report["triage_note"], report["entrance_visit"])
        else:
            return None # Used to check requested report validity
    else:
        raise ValueError("Ids list shouldn't be empty.")

    # Parse data from reports corresponding to the remaining ids
    for r_id in ids[1:]:
        current_report.r_ids = current_report.r_ids + [r_id]
        report = input_data[r_id]
        if report["triage_note"] not in current_report.triage_note:
            current_report.triage_note = current_report.triage_note + " | " + report["triage_note"]
        if report["entrance_visit"] not in current_report.entrance_visit:
            current_report.entrance_visit = current_report.entrance_visit + " | " + report["entrance_visit"]
    return current_report

def visualize_report(report):
    """
        Visualize (printing to stdout) a given report by displaying its fields.
    """

    print("\n" + "-"*80)

    if(len(report.r_ids) > 1):
        print(" "*32 + "REPORT COMPOSITO")
        print("\n  Identificativi: " + " / ".join(report.r_ids) + " (" + report.year + ")")
    elif(len(report.r_ids) == 1):
        print(" "*33 + "REPORT SINGOLO")
        print("\n  Identificativo: " + str(report.r_ids[0]) + " (" + report.year + ")")
    else:
        raise ValueError("Ids list shouldn't be empty.")

    print("  Genere: " + report.gender)
    print("  Età: " + report.age)
    print("  Violenza: " + str(report.violence))

    if report.triage_note:
        print("\n  Triage note:\n")
        wrapped_triage_note = '\n'.join(textwrap.wrap(report.triage_note, width=80))
        print(wrapped_triage_note)
    if report.entrance_visit:
        print("\n  Entrance visit:\n")
        wrapped_entrance_visit = '\n'.join(textwrap.wrap(report.entrance_visit, width=80))
        print(wrapped_entrance_visit)

    print("\n" + "-"*80)

def extract_train_set(input_directory, output_directory, train_set_size, class_ratio, length_threshold, write_mode):
    """
        Compose a training set according to given parametrization.

        Parameters:
            train_set_size: the size of the generated train set (most commonly computed via compute_sets_size())
            train_test_ratio (float [0, 1]): train-test split (e.g. with 0.80, train-test 80-20)
            class_ratio (float [0, 1]): v-nv ratio (e.g. with 0.97, 97% nv and 3% v)
            length_treshold: minimum length for a nv report to be included in the new sub-set
    """

    nv_size = 0; # The amount of nv entries in the final train set
    v_size = 0; # The amount of v entries in the final train set
    v_actual_size = 0; # The actual size of v dataset
    nv_actual_size = 0; # The actual size of nv dataset

    nv_ids = [] # Ids of nv reports choosen for the train set
    nv_lines = [] # Descriptors of reports choosen for the train set
    v_ids = [] # Ids of v reports choosen for the train set
    v_lines = [] # Descriptors of reports choosen for the train set

    randomly_picked_ids = []

    input_file_v = input_directory + "mauriziano_v_texts.txt"
    input_file_nv = input_directory + "mauriziano_nv_texts.txt"
    input_file_v_ids = input_directory + "mauriziano_v_ids.txt"
    input_file_nv_ids = input_directory + "mauriziano_nv_ids.txt"
    file_stats = input_directory + "mauriziano_stats.json"

    output_file_train_set = output_directory + "train.txt"
    output_file_train_set_ids = output_directory + "train_ids.txt"
    output_file_train_set_labels = output_directory + "train_labels.txt"
    output_file_unparsed_train_set = output_directory + "unparsed_train.txt"

    # Parse v and nv dataset sizes from stats file
    if os.path.isfile(file_stats):
        with open(file_stats, 'r') as f: json_data = json.load(f)
        v_actual_size = int(json_data['violent'])
        nv_actual_size = int(json_data['non_violent'])
    else:
        # If the stats file doesn't exists (it should), compute lengths from files (intensive)
        v_actual_size = len(open(input_file_v_ids).readlines())
        nv_actual_size = len(open(input_file_nv_ids).readlines())

    nv_size = int(train_set_size * class_ratio)
    v_size = int(train_set_size * (1 - class_ratio))

    if v_size > v_actual_size:
        raise Exception('Invalid parametrization - Too few examples')

    # Pick randomly #nv_size nv reports to be included in the train set
    while len(nv_ids) <= nv_size:
        rnd_fileline = random.randint(1, nv_actual_size)
        if rnd_fileline not in randomly_picked_filelines:
            current_descriptor = linecache.getline(input_file_nv, rnd_fileline)
            if(len(current_descriptor) >= length_threshold):
                 nv_lines.append(current_descriptor)
                 current_id = linecache.getline(input_file_nv_ids, rnd_fileline)
                 nv_ids.append(current_id.strip()) # linecache doesn't remove final new line
        randomly_picked_filelines.append(rnd_fileline)

    randomly_picked_filelines = []

    # Pick randomly #v_size v reports to be included in the train set
    while len(v_ids) <= v_size:
        rnd_fileline = random.randint(1, v_actual_size)
        if rnd_fileline not in randomly_picked_filelines:
            current_descriptor = linecache.getline(input_file_v, rnd_fileline)
            v_lines.append(current_descriptor)
            current_id = linecache.getline(input_file_v_ids, rnd_fileline)
            v_ids.append(current_id.strip()) # linecache doesn't remove final new line
        randomly_picked_filelines.append(rnd_fileline)

    # Write data to corresponding files
    with open(output_file_unparsed_train_set, write_mode) as f:
        for line in nv_lines: f.write(f"{line}")
        for line in v_lines: f.write(f"{line}")
    with open(output_file_train_set_ids, write_mode) as f:
        for r_id in nv_ids: f.write(f"{r_id}\n")
        for r_id in v_ids: f.write(f"{r_id}\n")
    with open(output_file_train_set_labels, write_mode) as f:
        for _ in nv_ids: f.write(f"NON-VIOLENT\n")
        for _ in v_ids: f.write(f"UNSPECIFIED_RELATIONSHIP\n")

    # Convert from plain text to BIO tagged data
    convert_txt_to_bio(output_file_unparsed_train_set, output_file_train_set, write_mode)

def extract_test_set(input_directory, output_directory, test_set_size, train_test_ratio, class_ratio, length_threshold):
    """
        Compose a test set according to given parametrization.

        Parameters:
            test_set_size: the size of the generated test set (most commonly computed via compute_sets_size())
            train_test_ratio (float [0, 1]): train-test split (e.g. with 0.80, train-test 80-20)
            class_ratio (float [0, 1]): v-nv ratio (e.g. with 0.97, 97% nv and 3% v)
            length_treshold: minimum length for a nv report to be included in the new sub-set
    """

    nv_size = 0; # The amount of nv entries in the final test set
    v_size = 0; # The amount of v entries in the final test set
    v_actual_size = 0; # The actual size of v dataset
    nv_actual_size = 0; # The actual size of nv dataset

    nv_ids = [] # Ids of nv reports choosen for the test set
    nv_lines = [] # Descriptors of nv reports choosen for the test set
    v_ids = [] # Ids of v reports choosen for the test set
    v_lines = [] # Descriptors of v reports choosen for the test set

    randomly_picked_ids = []

    input_file_v = input_directory + "mauriziano_v_texts.txt"
    input_file_nv = input_directory + "mauriziano_nv_texts.txt"
    input_file_v_ids = input_directory + "mauriziano_v_ids.txt"
    input_file_nv_ids = input_directory + "mauriziano_nv_ids.txt"
    file_stats = input_directory + "mauriziano_stats.json"

    file_train_set_ids = output_directory + "train_ids.txt"

    output_file_test_set = output_directory + "test.txt"
    output_file_test_set_ids = output_directory + "test_ids.txt"
    output_file_test_set_labels = output_directory + "test_labels.txt"
    output_file_unparsed_test_set = output_directory + "unparsed_test.txt"

    # Parse v and nv datasets sizes from stats file
    if os.path.isfile(file_stats):
        with open(file_stats, 'r') as f: json_data = json.load(f)
        v_actual_size = int(json_data['violent'])
        nv_actual_size = int(json_data['non_violent'])
    else:
        # If the stats file doesn't exists (it should), compute lengths from files (intensive)
        v_actual_size = len(open(input_file_v_ids).readlines())
        nv_actual_size = len(open(input_file_nv_ids).readlines())

    train_ids = open(file_train_set_ids).readlines()

    nv_size = int(test_set_size * class_ratio)
    v_size = int(test_set_size * (1 - class_ratio))

    if(v_size > v_actual_size):
        raise Exception('Invalid parametrization')

    # Pick randomly #nv_size nv reports to be included in the test set
    while len(nv_ids) != nv_size:
        rnd_fileline = random.randint(1, nv_actual_size)
        if rnd_fileline not in randomly_picked_ids:
            current_id = linecache.getline(input_file_nv_ids, rnd_fileline).replace('\n', '')
            if current_id not in train_ids: # Check if the current entry isn't already part of the training set
                current_descriptor = linecache.getline(input_file_nv, rnd_fileline)
                if(len(current_descriptor) >= length_threshold):
                    nv_lines.append(current_descriptor)
                    nv_ids.append(current_id) # linecache doesn't remove final new line
        randomly_picked_ids.append(rnd_fileline)

    randomly_picked_ids = []

    # Pick randomly #v_size v reports to be included in the test set
    while len(v_ids) != v_size:
        rnd_fileline = random.randint(1, v_actual_size)
        if rnd_fileline not in randomly_picked_ids:
            current_id = linecache.getline(input_file_v_ids, rnd_fileline).replace('\n', '')
            if current_id not in train_ids: # Check if the current entry isn't already part of the training set
                current_descriptor = linecache.getline(input_file_v, rnd_fileline)
                v_lines.append(current_descriptor)
                v_ids.append(current_id)
        randomly_picked_ids.append(rnd_fileline)

    # Write data to corresponding files
    with open(output_file_unparsed_test_set, 'w') as f:
        for line in nv_lines: f.write(f"{line}")
        for line in v_lines: f.write(f"{line}")
    with open(output_file_test_set_ids, 'w') as f:
        for r_id in nv_ids: f.write(f"{r_id}\n")
        for r_id in v_ids: f.write(f"{r_id}\n")
    with open(output_file_test_set_labels, 'w') as f:
        for _ in nv_ids: f.write(f"NON-VIOLENT\n")
        for _ in v_ids: f.write(f"UNSPECIFIED_RELATIONSHIP\n")

    # Convert from plain text to BIO tagged data
    convert_txt_to_bio(output_file_unparsed_test_set, output_file_test_set, 'w')

def extract_nv_set(input_directory, output_directory, nv_ids_linenums, length_threshold):
    """
        Generate a sub-set of the non-violent dataset according to the given parametrizations.
        This sub-set can be used, e.g., during inference by a model.

        Parameters:
            input_directory: directory where the non-violent set is stored
            output_directory: directory where the current subset of the nv set will be stored
            nv_ids_linenums: list of file lines numbers corresponding to nv report (generated split_nv_set())
            length_treshold: minimum length for a nv report to be included in the new sub-set
    """

    nv_actual_size = 0; # The actual size of nv dataset

    nv_ids = [] # Ids of nv reports choosen to be
    nv_lines = [] # Descriptors of nv reports choosen for the test set

    randomly_picked_ids = []

    input_file_nv = input_directory + "mauriziano_nv_texts.txt"
    input_file_nv_ids = input_directory + "mauriziano_nv_ids.txt"
    file_stats = input_directory + "mauriziano_stats.json"

    file_train_set_ids = output_directory + "train_ids.txt"

    output_file_nv_set = output_directory + "nv.txt"
    output_file_unparsed_nv_set = output_directory + "unparsed_nv.txt"
    output_file_nv_set_ids = output_directory + "nv_ids.txt"
    output_file_nv_set_labels = output_directory + "nv_labels.txt"

    # Parse nv dataset size from stats file
    if os.path.isfile(file_stats):
        with open(file_stats, 'r') as f: json_data = json.load(f)
        nv_actual_size = int(json_data['non_violent'])
    else:
        # If the stats file doesn't exists (it should), compute lengths from files (intensive)
        nv_actual_size = len(open(input_file_nv_ids).readlines())

    train_ids = [r_id.strip() for r_id in open(file_train_set_ids)]

    # Pick randomly #nv_size nv reports to be included in the nv set
    for linenum in nv_ids_linenums:
        current_id = linecache.getline(input_file_nv_ids, linenum).strip()
         # Check if the current entry isn't already part of the training set or already reviewed by human
        if current_id not in train_ids:
            current_descriptor = linecache.getline(input_file_nv, linenum)
            if(len(current_descriptor) >= length_threshold):
                nv_lines.append(current_descriptor)
                nv_ids.append(current_id) # linecache doesn't remove final new line

    # Write data to corresponding files
    with open(output_file_unparsed_nv_set, 'w') as f:
        for line in nv_lines: f.write(f"{line}")
    with open(output_file_nv_set_ids, 'w') as f:
        for r_id in nv_ids: f.write(f"{r_id}\n")
    with open(output_file_nv_set_labels, 'w') as f:
        for _ in nv_ids: f.write(f"NON-VIOLENT\n")

    convert_txt_to_bio(output_file_unparsed_nv_set, output_file_nv_set, 'w')

def split_nv_set(input_directory, split_size):
    """
        Splits nv reports according to split_size.
     """

    input_file_nv_ids = input_directory + "mauriziano_nv_ids.txt"
    file_stats = input_directory + "mauriziano_stats.json"

    if os.path.isfile(file_stats):
        with open(file_stats, 'r') as f: json_data = json.load(f)
        nv_actual_size = int(json_data['non_violent'])
    else:
        # If the stats file doesn't exists (it should), compute lengths from files (intensive)
        nv_actual_size = len(open(input_file_nv_ids).readlines())

    # Instead of directly using the ids, use file lines (to avoid expensive file search later)
    ids_linenum = np.arange(nv_actual_size)
    np.random.shuffle(ids_linenum)
    num_of_splits = math.ceil(nv_actual_size / split_size)

    return np.array_split(ids_linenum, num_of_splits)

def extract_v_set(input_directory, output_directory, write_mode='w'):
    """
        Generate a set composed by violent report only, to be used e.g. for validation.
    """

    v_actual_size = 0; # The actual size of v dataset

    v_ids = []
    v_lines = []

    input_file_v = input_directory + "mauriziano_v_texts.txt"
    input_file_v_ids = input_directory + "mauriziano_v_ids.txt"
    file_stats = input_directory + "mauriziano_stats.json"

    file_train_set_ids = output_directory + "train_ids.txt"

    output_file_v_set = output_directory + "v.txt"
    output_file_unparsed_v_set = output_directory + "unparsed_v.txt"
    output_file_v_set_ids = output_directory + "v_ids.txt"
    output_file_v_set_labels = output_directory + "v_labels.txt"

    v_ids = open(input_file_v_ids).readlines()
    v_lines = open(input_file_v).readlines()

    # Write data to corresponding files
    with open(output_file_unparsed_v_set, write_mode) as f:
        for line in v_lines: f.write(f"{line}")
    with open(output_file_v_set_ids, write_mode) as f:
        for r_id in v_ids: f.write(f"{r_id}\n")
    with open(output_file_v_set_labels, write_mode) as f:
        for _ in v_ids: f.write(f"UNSPECIFIED_RELATIONSHIP\n")

    convert_txt_to_bio(output_file_unparsed_v_set, output_file_v_set, 'w')

def compute_sets_size(input_directory, split, class_ratio, source_human=False):
    """
        Compute train and set size according to train-test split and class ratio.

        Parameters:
            input_directory: directory where the stats file is stored
            split (float [0, 1]): proportion of dataset to be used as training and set sets
            class_ratio (float [0, 1]): v-nv ratio (e.g. with 0.8, 80% nv and 20% v)
            source_human (bool): flag to source or not human verified v/nv reports in the training

        Returns: Training set size, Test set size
    """

    input_file_v_ids = input_directory + "mauriziano_v_ids.txt"
    input_file_nv_ids = input_directory + "mauriziano_nv_ids.txt"
    file_stats = input_directory + "mauriziano_stats.json"

    # Parse v and nv datasets sizes from stats file
    if os.path.isfile(file_stats):
        with open(file_stats, 'r') as f: json_data = json.load(f)
        v_size = int(json_data['violent'])
        nv_actual_size = int(json_data['non_violent'])
    else:
        # If the stats file doesn't exists (it should), compute lengths from files (intensive)
        v_size = len(open(input_file_v_ids).readlines())
        nv_actual_size = len(open(input_file_nv_ids).readlines())

    # source human verified v/nv reports
    if source_human:
        confirmed_v_ids_file = input_file_nv_ids + "gold_v_ids.txt"
        confirmed_nv_ids_file = input_file_nv_ids + "gold_nv_ids.txt"
        if os.path.isfile(confirmed_v_ids_file) and os.path.isfile(confirmed_nv_ids_file):
            confirmed_v_size = len(open(confirmed_v_ids_file).readlines())
            confirmed_nv_size = len(open(confirmed_nv_ids_file).readlines())
        else:
            print("Unable to source human verified file during sets size computation")
    else:
        confirmed_v_size = 0
        confirmed_nv_size = 0

    # Compute training and test sets size according to split and class ratio
    train_v_size = int(v_size * split)
    train_nv_size = int((train_v_size * class_ratio) / (1 - class_ratio))
    test_v_size = v_size - train_v_size
    test_nv_size = int((test_v_size * class_ratio) / (1 - class_ratio))

    # print(train_v_size, train_nv_size, test_v_size, test_nv_size)

    return train_v_size + train_nv_size + confirmed_v_size + confirmed_nv_size, test_v_size + test_nv_size

def convert_txt_to_bio(input_file, output_file, write_mode):
    """
        Convert plain text to BIO-tagged data.

        Current implementation doesn't perform *actual* BIO tagging, since BIO-tagging is
        not part of this project's scope (and its a reminiscente from previous stages of the project)
        this part is simply bypassed. Further exploratons may benefit from a proper implementation.
    """

    lines = open(input_file).readlines()

    with open(output_file, write_mode) as f:
        for line in lines:
            tokenized_line = word_tokenize(line, language='italian')
            f.write("[id] O O O\n-> O O O\n")
            for word in tokenized_line:
                f.write(f"{word} O O O\n")
            f.write("\n")

def merge_original_sets(original_input_directory, new_input_directory, write_mode='w'):
    """
        Merge ISS train and test sets to be used as a compsoite train set for a new model.
    """

    train_file = original_input_directory + "train.txt"
    train_file_ids = original_input_directory + "train_ids.txt"
    train_file_labels = original_input_directory + "train_labels.txt"

    test_file = original_input_directory + "test.txt"
    test_file_ids = original_input_directory + "test_ids.txt"
    test_file_labels = original_input_directory + "test_labels.txt"

    # Store previous entries verbatim
    with open(new_input_directory + "train.txt", write_mode) as new_train:
        with open(train_file, 'r') as old_train: new_train.write(old_train.read())
        with open(test_file, 'r') as old_test: new_train.write(old_test.read())

    with open(new_input_directory + "train_ids.txt", write_mode) as new_train_ids:
        with open(train_file_ids, 'r') as old_train_ids: new_train_ids.write(old_train_ids.read())
        with open(test_file_ids, 'r') as old_test_ids: new_train_ids.write(old_test_ids.read())

    with open(new_input_directory + "train_labels.txt", write_mode) as new_train_labels:
        with open(train_file_labels, 'r') as old_train_labels: new_train_labels.write(old_train_labels.read())
        with open(test_file_labels, 'r') as old_test_labels: new_train_labels.write(old_test_labels.read())

    # Store the dataset composition (total, v, nv) in a file
    original_labels = [r_label.strip() for r_label in open(train_file_labels)]
    original_labels = original_labels + [r_label.strip() for r_label in open(test_file_labels)]
    original_v = len([r_label for r_label in original_labels if r_label != "NON-VIOLENT"])
    original_nv = len([r_label for r_label in original_labels if r_label == "NON-VIOLENT"])

    if os.path.isfile(new_input_directory + "composition.json"):
        with open(new_input_directory + "composition.json", 'r') as f: composition = json.load(f)
    else:
        composition = {}

    composition['original'] = original_v + original_nv
    composition['original_v'] = original_v
    composition['original_nv'] = original_nv
    composition['v'] = original_v + composition['v'] if 'v' in composition else original_v
    composition['nv'] = original_nv + composition['nv'] if 'nv' in composition else original_nv

    json_data = json.dumps(composition)
    with open(new_input_directory + "composition.json", 'w') as f: f.write(json_data)

def copy_original_train_set(original_input_directory, new_input_directory):
    """
        Copy ISS train data to be used as train set for a new model.
    """

    train_file = original_input_directory + "train.txt"
    train_file_ids = original_input_directory + "train_ids.txt"
    train_file_labels = original_input_directory + "train_labels.txt"

    with open(new_input_directory + "train.txt", 'w') as new_train:
        with open(train_file, 'r') as old_train: new_train.write(old_train.read())
    with open(new_input_directory + "train_ids.txt", 'w') as new_train_ids:
        with open(train_file_ids, 'r') as old_train_ids: new_train_ids.write(old_train_ids.read())
    with open(new_input_directory + "train_labels.txt", 'w') as new_train_labels:
        with open(train_file_labels, 'r') as old_train_labels: new_train_labels.write(old_train_labels.read())

def merge_human_confirmed_to_train(original_input_directory, new_input_directory):
    """
        Merge human verified v and nv reports to compose a new training set.
    """

    confirmed_v_file = original_input_directory + "gold_v.txt"
    confirmed_v_file_ids = original_input_directory + "gold_v_ids.txt"
    confirmed_v_file_labels = original_input_directory + "gold_v_labels.txt"

    confirmed_nv_file = original_input_directory + "gold_nv.txt"
    confirmed_nv_file_ids = original_input_directory + "gold_nv_ids.txt"

    # Since confirmed_v.txt contains non BIO-tagged texts, its entries are stored in
    # output_file_unparsed_train_set and then parsed later (during extract_train_set() execution)
    write_mode = 'a' if os.path.isfile(new_input_directory + "unparsed_train.txt") else 'w'
    with open(new_input_directory + "unparsed_train.txt", write_mode) as new_train:
        with open(confirmed_v_file, 'r') as confirmed_v: new_train.write(confirmed_v.read())
        with open(confirmed_nv_file, 'r') as confirmed_nv: new_train.write(confirmed_nv.read())

    write_mode = 'a' if os.path.isfile(new_input_directory + "train_ids.txt") else 'w'
    with open(new_input_directory + "train_ids.txt", write_mode) as new_train_ids:
        with open(confirmed_v_file_ids, 'r') as confirmed_v_ids: new_train_ids.write(confirmed_v_ids.read())
        with open(confirmed_nv_file_ids, 'r') as confirmed_nv_ids: new_train_ids.write(confirmed_nv_ids.read())

    write_mode = 'a' if os.path.isfile(new_input_directory + "train_labels.txt") else 'w'
    with open(new_input_directory + "train_labels.txt", write_mode) as new_train_labels:
        with open(confirmed_v_file_labels, 'r') as confirmed_v_labels: new_train_labels.write(confirmed_v_labels.read())
        with open(confirmed_nv_file_ids, 'r') as confirmed_nv_ids:
            for _ in confirmed_nv_ids.readlines(): new_train_labels.write(f"NON-VIOLENT\n")

    # Store the dataset composition (total, v, nv) in a file
    gold_v = len(open(confirmed_v_file_ids).readlines())
    gold_nv = len(open(confirmed_nv_file_ids).readlines())

    if os.path.isfile(new_input_directory + "composition.json"):
        with open(new_input_directory + "composition.json", 'r') as f: composition = json.load(f)
    else:
        composition = {}

    composition['gold'] = gold_v + gold_nv
    composition['gold_v'] = gold_v
    composition['gold_nv'] = gold_nv
    composition['v'] = gold_v + composition['v'] if 'v' in composition else gold_v
    composition['nv'] = gold_nv + composition['nv'] if 'nv' in composition else gold_nv

    json_data = json.dumps(composition)
    with open(new_input_directory + "composition.json", 'w') as f: f.write(json_data)

def merge_v_set_to_train(original_input_directory, new_input_directory):
    """
        Merge v reports from to compose a new training set.
    """

    v_actual_size = 0; # The actual size of v dataset

    v_ids = [] # Ids of v reports
    v_lines = [] # Descriptors of v reports
    v_labels = [] # Label of v reports

    input_file_v = original_input_directory + "mauriziano_v_texts.txt"
    input_file_v_ids = original_input_directory + "mauriziano_v_ids.txt"
    input_file_v_labels = original_input_directory + "mauriziano_v_labels.txt"

    v_lines = [line.strip() for line in open(input_file_v)]
    v_ids = [r_id.strip() for r_id in open(input_file_v_ids)]
    v_labels = [label.strip() for label in open(input_file_v_labels)]

    # Since v texts are non BIO-tagged texts, those entries are stored in
    # output_file_unparsed_train_set and then parsed later (during extract_train_set() execution)
    write_mode = 'a' if os.path.isfile(new_input_directory + "unparsed_train.txt") else 'w'
    with open(new_input_directory + "unparsed_train.txt", write_mode) as new_train:
        for line in v_lines: new_train.write(f"{line}\n")
    write_mode = 'a' if os.path.isfile(new_input_directory + "train_ids.txt") else 'w'
    with open(new_input_directory + "train_ids.txt", write_mode) as new_train_ids:
        for r_id in v_ids: new_train_ids.write(f"{r_id}\n")
    write_mode = 'a' if os.path.isfile(new_input_directory + "train_labels.txt") else 'w'
    with open(new_input_directory + "train_labels.txt", write_mode) as new_train_labels:
        for label in v_labels: new_train_labels.write(f"{label}\n")

    # Store the dataset composition (total, v, nv) in a file
    if os.path.isfile(new_input_directory + "composition.json"):
        with open(new_input_directory + "composition.json", 'r') as f: composition = json.load(f)
    else:
        composition = {}

    composition['mau_v'] = len(v_ids)
    composition['v'] = len(v_ids) + composition['v'] if 'v' in composition else len(v_ids)

    json_data = json.dumps(composition)
    with open(new_input_directory + "composition.json", 'w') as f: f.write(json_data)

def merge_n_nv_reports(input_directory, output_directory, reports_number, length_threshold):
    """
        Select a given number of nv reports from Mauriziano's nv set to compose a new training set.
        The selection is performed randomly but proportionally on each year cardinality.

        Parameters:
            input_directory: directory where the non-violent set is stored
            output_directory: directory where the current subset of the nv set will be stored
            reports_number: the number of nv reports to be selected and merged
            length_treshold: minimum length for a nv report to be included in the new sub-set
    """

    nv_actual_size = 0; # The actual size of nv dataset

    nv_ids = [] # Ids of nv reports that will be merged
    nv_lines = [] # Descriptors of nv reports that will be merged

    randomly_picked_ids = []

    input_file_nv = input_directory + "mauriziano_nv_texts.txt"
    input_file_nv_ids = input_directory + "mauriziano_nv_ids.txt"
    file_stats = input_directory + "mauriziano_stats.json"

    file_train_set_ids = output_directory + "train_ids.txt"

    # Parse nv dataset size from stats file and each year cardinality
    if os.path.isfile(file_stats):
        with open(file_stats, 'r') as f: json_data = json.load(f)
        nv_actual_size = int(json_data['non_violent'])
        years_cardinality = json_data['years_cardinality']
    else:
        # If the stats file doesn't exists (it should), compute lengths from files (intensive)
        nv_actual_size = len(open(input_file_nv_ids).readlines())

    train_ids = [r_id.strip() for r_id in open(file_train_set_ids)]
    randomly_picked_filelines = []

    # Pick randomly #nv_size nv reports to be included in the nv set according to yearly distribution
    total_reports = sum(years_cardinality.values())
    reports_per_year = {year: int(reports_number * cardinality / total_reports) for year, cardinality in years_cardinality.items()}

    for year in reports_per_year.keys():

        current_reports_per_year = 0
        randomly_picked_filelines = []

        while current_reports_per_year <= reports_per_year[year]:
            rnd_fileline = random.randint(1, nv_actual_size)
            if rnd_fileline not in randomly_picked_filelines:
                current_id = linecache.getline(input_file_nv_ids, rnd_fileline).strip() # linecache doesn't remove final new line
                r_ids = current_id.split("_") # Ids are composed by creation year + report id (multiple ids for composite records)
                # Check if the current entry isn't already part of the training set and belongs to the current year
                if current_id not in train_ids and r_ids[0] == year:
                    current_descriptor = linecache.getline(input_file_nv, rnd_fileline)
                    if(len(current_descriptor) >= length_threshold):
                        nv_lines.append(current_descriptor)
                        nv_ids.append(current_id)
                        current_reports_per_year += 1
                randomly_picked_filelines.append(rnd_fileline)

    # Since nv texts are non BIO-tagged texts, those entries are stored in
    # output_file_unparsed_train_set and then parsed later (during extract_train_set() execution)
    write_mode = 'a' if os.path.isfile(output_directory + "unparsed_train.txt") else 'w'
    with open(output_directory + "unparsed_train.txt", write_mode) as new_train:
        for line in nv_lines: new_train.write(f"{line}")
    write_mode = 'a' if os.path.isfile(output_directory + "train_ids.txt") else 'w'
    with open(output_directory + "train_ids.txt", write_mode) as new_train_ids:
        for r_id in nv_ids: new_train_ids.write(f"{r_id}\n")
    write_mode = 'a' if os.path.isfile(output_directory + "train_labels.txt") else 'w'
    with open(output_directory + "train_labels.txt", write_mode) as new_train_labels:
        for _ in nv_ids: new_train_labels.write(f"NON-VIOLENT\n")

    # Store the new dataset composition (nv) in a file
    if os.path.isfile(output_directory + "composition.json"):
        with open(output_directory + "composition.json", 'r') as f: composition = json.load(f)
    else:
        composition = {}

    composition['mau_nv'] = len(nv_ids)
    composition['nv'] = len(nv_ids) + composition['nv'] if 'nv' in composition else len(nv_ids)

    json_data = json.dumps(composition)
    with open(output_directory + "composition.json", 'w') as f: f.write(json_data)

def merge_n_original_nv_reports(input_directory, output_directory, reports_number, length_threshold):
    """
        Select a given number of nv reports from ISS's nv set to compose a new training set.

        Parameters:
            input_directory: directory where the non-violent set is stored
            output_directory: directory where the current subset of the nv set will be stored
            reports_number: the number of nv reports to be selected and merged
            length_treshold: minimum length for a nv report to be included in the new sub-set
    """

    nv_actual_size = 0; # The actual size of nv dataset

    nv_ids = [] # Ids of nv reports choo
    nv_lines = [] # Descriptors of nv reports choosen

    randomly_picked_filelines = []

    input_file_nv = input_directory + "nv.txt"
    input_file_nv_ids = input_directory + "nv_ids.txt"

    current_train_set_ids = output_directory + "train_ids.txt"

    nv_actual_size = len(open(input_file_nv_ids).readlines())

    train_ids = [r_id.strip() for r_id in open(current_train_set_ids)]

    # Pick randomly #nv_size nv reports to be included in the nv set
    while len(nv_ids) <= reports_number:
        rnd_fileline = random.randint(1, nv_actual_size)
        if rnd_fileline not in randomly_picked_filelines: # ISS and Mauriziano's ids have different format (no overlap possible)
            current_id = linecache.getline(input_file_nv_ids, rnd_fileline).strip() # linecache doesn't remove final new line
            # Check if the current entry isn't already part of the training set or already reviewed by human
            if current_id not in train_ids:
                current_descriptor = linecache.getline(input_file_nv, rnd_fileline)
                if(len(current_descriptor) >= length_threshold):
                    nv_lines.append(current_descriptor)
                    nv_ids.append(current_id)
        randomly_picked_filelines.append(rnd_fileline)

    # Since nv texts are non BIO-tagged texts, those entries are stored in
    # output_file_unparsed_train_set and then parsed later (during extract_train_set() execution)
    write_mode = 'a' if os.path.isfile(output_directory + "unparsed_train.txt") else 'w'
    with open(output_directory + "unparsed_train.txt", write_mode) as new_train:
        for line in nv_lines: new_train.write(f"{line}")
    write_mode = 'a' if os.path.isfile(output_directory + "train_ids.txt") else 'w'
    with open(output_directory + "train_ids.txt", write_mode) as new_train_ids:
        for r_id in nv_ids: new_train_ids.write(f"{r_id}\n")
    write_mode = 'a' if os.path.isfile(output_directory + "train_labels.txt") else 'w'
    with open(output_directory + "train_labels.txt", write_mode) as new_train_labels:
        for _ in nv_ids: new_train_labels.write(f"NON-VIOLENT\n")

    # Store the new dataset composition (nv) in a file
    if os.path.isfile(output_directory + "composition.json"):
        with open(output_directory + "composition.json", 'r') as f: composition = json.load(f)
    else:
        composition = {}

    composition['iss_nv'] = len(nv_ids)
    composition['nv'] = len(nv_ids) + composition['nv'] if 'nv' in composition else len(nv_ids)

    json_data = json.dumps(composition)
    with open(output_directory + "composition.json", 'w') as f: f.write(json_data)

def create_dev_set(input_directory):
    """
        Dev set generation can be bypassed for current implementation
    """

    with open(input_directory + "dev.txt", 'w') as dev:
        with open(input_directory + "train.txt", 'r') as train: dev.write(train.read())
    with open(input_directory + "dev_ids.txt", 'w') as dev_ids:
        with open(input_directory + "train_ids.txt", 'r') as train_ids: dev_ids.write(train_ids.read())
    with open(input_directory + "dev_labels.txt", 'w') as dev_labels:
        with open(input_directory + "train_labels.txt", 'r') as train_labels: dev_labels.write(train_labels.read())

def merge_embeddings_source(first_source, second_source, final_file):
    """
        Merge (verbatim) two sources for the later creation of embeddings.
    """
    with open(final_file, 'w') as output_file:
        with open(first_source, 'r') as input_file: output_file.write(input_file.read())
        with open(second_source, 'r') as input_file: output_file.write(input_file.read())