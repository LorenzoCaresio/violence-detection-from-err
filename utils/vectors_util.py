import fasttext
from tqdm import tqdm
import numpy as np

def build_vectors(all_text_fp, model_out_fp):

    # Skipgram model
    model = fasttext.train_unsupervised(all_text_fp, model='skipgram', dim=300, minCount=1)
    model.save_model(model_out_fp)

    # Bin to Vec
    vec_fp = model_out_fp.replace(".bin", ".vec")
    vec_str = ""
    words = model.get_words()

    for w in tqdm(words, desc="Building vec format"):
        v = model.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        vec_str += w + vstr + "\n"

    open(vec_fp, 'w').write(vec_str)


def load_vectors(vectors_fp):

    print("Loading vectors from " + vectors_fp)

    # Extract word embeddings from the file mapping them to tokenizer dictionary
    vectors = dict()

    lines = open(vectors_fp, encoding="utf-8").read().split("\n")
    for line in tqdm(lines, desc="Loading vectors"):
        if line != "":
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            vectors[word] = coefs
    return vectors

def get_embedding_matrix(vectors, tokenizer):
    # Create a weight matrix
    embedding_matrix = np.zeros((len(vectors), 300))
    for word, index in tokenizer.word_index.items():
        if index > len(vectors)-1:
            break
        else:
            if word in vectors:
                embedding_vector = vectors[word]
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
            else:
                embedding_matrix[index] = np.zeros((1,300))

    return embedding_matrix