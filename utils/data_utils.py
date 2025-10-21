from conf.configuration import *
import json

dizionario = {"EX_SPOUSE_PARTNER": 0,
        "KNOWNPERSON_FRIEND": 1,
        "OTHER_RELATIONSHIP": 2,
        "RELATIVE": 3,
        "SELF": 4,
        "SPOUSE_PARTNER": 5,
        "THIEF": 6,
        "UNKNOWN_PERSON": 7,
        "UNSPECIFIED_RELATIONSHIP": 8,
        "NON-VIOLENT": 9}

#non violent
def load_dataset_only_nonviolent(path):

    texts = []
    labels = []

    with open(path, 'r') as f:
        texts = f.read().split("\n")
        texts = [x for x in texts if x != ""]
        labels = [9]*len(texts)
    return texts,labels

#violent senza ner
def load_dataset_only_violent_add_sep(path1, path2):

    texts = []
    labels = []

    with open(path2, 'r') as f2:
        lines = f2.read().split("\n")
        for l in lines:
            if l != "":
                labels.append(dizionario[l])
    with open(path1, 'r') as f1:
            lines = f1.read().split("\n")[1:]
            actual_frase = []
            post_frase = []
            ner_seq = []
            for l_complete in lines:
                if l_complete == "" and len(actual_frase) != 0:

                    if len(ner_seq) != 0:
                        ner_seq.append("\"")
                        actual_frase.extend(ner_seq)
                        post_frase.extend(ner_seq)
                        actual_frase.append(l)
                        ner_seq = []

                    texts.append(" ".join(actual_frase) + " [SEP] " + " ".join(post_frase).replace('"', ''))
                    actual_frase = []
                    post_frase = []
                elif l_complete == "" and len(actual_frase) == 0:
                    continue
                else:
                    l = l_complete.split(" ")[0]
                    ner = l_complete.split(" ")[-1]
                    if ner == "B-AGENT" or ner == "I-AGENT":
                        if len(ner_seq) == 0:
                            ner_seq.append("\"")
                            ner_seq.append(l)
                        else:
                            ner_seq.append(l)
                    else:
                        if len(ner_seq) == 0:
                            actual_frase.append(l)
                        else:
                            ner_seq.append("\"")
                            actual_frase.extend(ner_seq)
                            post_frase.extend(ner_seq)
                            actual_frase.append(l)
                            ner_seq = []
    return texts, labels

# Violent with NER
def load_dataset_only_violent(path1, path2):

    texts = []
    labels = []

    with open(path2, 'r') as f2:
        lines = f2.read().split("\n")
        for l in lines:
            if l != "":
                labels.append(dizionario[l])
    with open(path1, 'r') as f1:
            lines = f1.read().split("\n")[1:]
            actual_frase = []

            for l_complete in lines:
                if l_complete == "" and len(actual_frase) != 0:
                    texts.append(" ".join(actual_frase))
                    actual_frase = []
                elif l_complete == "" and len(actual_frase) == 0:
                    continue
                else:
                    l = l_complete.split(" ")[0]
                    actual_frase.append(l)
    return texts, labels