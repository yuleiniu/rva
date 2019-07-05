import argparse
from tqdm import tqdm
import yaml
import numpy as np
from visdialch.data.dataset import VisDialDataset
from visdialch.data.vocabulary import Vocabulary

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml", default="configs/rva.yml",
    help="Path to a config file listing reader, model and solver parameters."
)
parser.add_argument(
    "--pretrained-txt", default="data/glove.6B.300d.txt",
    help="Path to GloVe pretrained word vectors."
)
parser.add_argument(
    "--save-npy", default="data/glove.npy",
    help="Path to save word embeddings."
)

# ================================================================================================
#   INPUT ARGUMENTS AND CONFIG
# ================================================================================================

args = parser.parse_args(args=[])

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))

# ================================================================================================
#   SETUP DATASET
# ================================================================================================

vocabulary = Vocabulary(
    config["dataset"]["word_counts_json"], min_count=config["dataset"]["vocab_min_count"]
)

def loadGloveModel(gloveFile):
    print("Loading pretrained word vectors...")
    with open(gloveFile,'r') as f:
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

glove = loadGloveModel(args.pretrained_txt)

vocab_size = len(vocabulary.index2word)
glove_data = np.zeros(shape=[vocab_size, 300], dtype=np.float32)
for i in range(0, vocab_size):
    word = vocabulary.index2word[i]
    if word in ['<PAD>', '<S>', '</S>']:
        continue
    if word in glove:
        glove_data[i] = glove[word]
    else:
        glove_data[i] = glove['unk']
np.save(args.save_npy, glove_data) 