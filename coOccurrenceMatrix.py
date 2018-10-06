from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer as wnl
import nltk, gensim, re, string, enchant, glob
from itertools import islice, compress
import itertools
import matplotlib.pyplot as plt
import numpy
import networkx as nx

model = "./enwiki_5_ner.txt"
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(model, binary=False)

#################################################
# Initialize, config & define helpful functions #
#################################################

stop_words = nltk.corpus.stopwords.words('english')
newStopWords = ['fig', 'figure', 'table', 'may', 'also', 'zero', 'one', 'two', 'three', 'four', 'five', 'six',
                'seven', 'eight', 'nine', 'ten', 'across', 'among', 'beside', 'however', 'yet', 'within']
stop_words.extend(newStopWords)
translator = str.maketrans('', '', string.punctuation.replace('-', '')) #filters punctuation except dash
lemmatizeCondition = 1
lemmatizer = wnl()

# Function for finding index of words of interest, like 'references'

def find(target):
    for i, word in enumerate(sents):
        try:
            j = word.index(target)
        except ValueError:
            continue
        yield i

# Function for handling the input for gensim word2vec

class FileToSent(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, 'r'):
            ll = line.strip().split(",")
            ll = [''.join(c for c in s if c not in string.punctuation) for s in ll]
            ll = [num.strip() for num in ll]
            yield ll

# Function for looking for element x occurs at least n times in list

def check_list(lst, x, n):
    gen = (True for i in lst if i==x)
    return next(islice(gen, n-1, None), False)


###################################################
# Read in .txt file(s) from a specified directory #
###################################################

IDs = glob.glob('./text/*')
IDs_subIDs = []
for ID in IDs:
    IDs_subIDs += glob.glob(ID + '/*.txt')
print(len(IDs))  # Print number of files read

####################
# Clean, lemmatize #
####################

for ID in IDs:  # loop through papers
    print(ID)
    with open(ID) as paper:
        text = paper.read()

        ############
        # Cleaning #
        ############

        text = re.sub("\u2013|\u2014", "-", text)  # Replace em-dashes
        sents = sent_tokenize(text)  # Split into sentences
        sents = [word_tokenize(s) for s in sents]
        sents = [[w.translate(translator) for w in s] for s in sents]  # filter punctuation
        sents = [[re.sub(r'\d+', 'numeric', w) for w in s] for s in
                 sents]  # replace all numerals with the holder "number"
        sents = [[w for w in s if re.search('[^a-zA-Z-]+', w) is None] for s in
                 sents]  # trips everything but alphabetic
        sents = [[w.lower() for w in s] for s in sents]  # make lower case
        sents = [s for s in sents if len(s) > 0]  # remove empty lines
        sents = [[w for w in s if not w in stop_words] for s in sents]  # filter stop words
        sents = [[w for w in s if len(w) > 1] for s in sents]  # filters out variables, etc
        sents = [[w for w in s if len(w) > 2] for s in sents]  # filters out variables, etc
        sents = [[w for w in s if len(w) > 3] for s in sents]  # filters out variables and abbreviations
        sents = [s for s in sents if len(s) > 0]  # remove empty lines
        words = [[lemmatizer.lemmatize(w) for w in s if lemmatizeCondition == 1] for s in sents]  # lemmatize
        words = list(itertools.chain.from_iterable(words))  # join list of lists

        # Write cleaned text to file
        with open('./cleanText/cleanedText.txt', 'w') as f:
            for _list in words:
                f.write(str(_list) + ' ')

###############################
# Construct semantic networks #
###############################

"""
Code to make a network out of the shortest N cosine-distances (or, equivalently, the strongest N associations)
between a set of words in a gensim word2vec model.
"""

model = word_vectors # load

# Specify words
###############

my_words = []

text = open('./cleanText/cleanedText.txt').read()

for word in word_tokenize(text):  # append unique words in the whole corpus
    print(word)
    if word in my_words:
        continue
    else:
        my_words.append(word)

# filter out words not in model
my_words = [word for word in my_words if word in model]

# The number of connections we want: either as a factor of the number of words or a set number
num_top_conns = len(my_words) * 19

# Make a list of all word-to-word distances [each as a tuple of (word1,word2,dist)]
dists=[]

# Find similarity distances between each word pair

for i1,word1 in enumerate(my_words):
    for i2,word2 in enumerate(my_words):
        if i1>=i2: continue
        cosine_similarity = model.similarity(word1,word2)
        cosine_distance = 1 - cosine_similarity
        dist = (word1, word2, cosine_distance)
        dists.append(dist)

# Sort the list by ascending distance
dists.sort(key=lambda _tuple: _tuple[-1])

# Get the top connections
top_conns = dists[:num_top_conns]


# Make a network
g = nx.Graph()
for word1,word2,dist in top_conns:
    weight = 1 - dist # cosine similarity for weight
    g.add_edge(word1, word2, weight=float(weight))

# Write the network
nx.write_graphml(g, "./semanticNetwork/semanticNetwork.graphml") # Readable by Gephi

A = nx.adjacency_matrix(g)
adjmat = A.todense()

numpy.savetxt("./semanticNetwork/semanticNetworkAdjmat.txt", adjmat, delimiter = ' ')

###########################
# reload and clean text without lemmatization and without
# spell-checking to leave words as original as possible

for ID in IDs:  # loop through papers
    print(ID)
    with open(ID) as paper:
        text = paper.read()

        ############
        # Cleaning #
        ############

        text = re.sub("\u2013|\u2014", "-", text)  # Replace em-dashes
        sents = sent_tokenize(text)  # Split into sentences
        sents = [word_tokenize(s) for s in sents]
        sents = [[w.translate(translator) for w in s] for s in sents]  # filter punctuation
        sents = [[re.sub(r'\d+', 'numeric', w) for w in s] for s in
                 sents]  # replace all numerals with the holder "number"
        sents = [[w for w in s if re.search('[^a-zA-Z-]+', w) is None] for s in
                 sents]  # trips everything but alphabetic
        sents = [[w.lower() for w in s] for s in sents]  # make lower case
        sents = [s for s in sents if len(s) > 0]  # remove empty lines
        sents = [[w for w in s if not w in stop_words] for s in sents]  # filter stop words
        sents = [[w for w in s if len(w) > 1] for s in sents]  # filters out variables, etc
        sents = [[w for w in s if len(w) > 2] for s in sents]  # filters out variables, etc
        sents = [[w for w in s if len(w) > 3] for s in sents]  # filters out variables and abbreviations
        sents = [s for s in sents if len(s) > 0]  # remove empty lines
        words = [[w for w in s] for s in sents]  # lemmatize
        words = list(itertools.chain.from_iterable(words))  # join list of lists

from collections import OrderedDict

windows = []

# For each sentence, retrieve 5-gram windows
for sent in sents:
    print(sent)
    for i in list(range(len(sent))):
        if len(sent) <= 5: # if the sentence is less than 5 words, just return that sentence
            window_slice = sent
            windows.append(window_slice)
            break
        else:
            window_slice = sent[i:i + 5] # otherwise, return as many 5-grams as possible
            if len(window_slice) == 5:
                windows.append(window_slice)

my_words = []

for word in words:  # append unique words
    print(word)
    if word in my_words:
        continue
    else:
        my_words.append(word)

# Create an ordered dictionary that counts the occurrence of words
# in a 5-gram sliding window

document = windows
names = my_words

occurrences = OrderedDict((name, OrderedDict((name, 0) for name in names)) for name in names)

# Find the co-occurrences:
for l in document:
    for i in range(len(l)):
        for item in l[:i] + l[i + 1:]:
            occurrences[l[i]][item] += 1

# Print the matrix:
print(' ', ' '.join(occurrences.keys()))
for name, values in occurrences.items():
    print(' '.join(str(i) for i in values.values()))


# Save the data
with open("/Users/dalezhou/Desktop/coOccurrenceMatrix.txt", "w") as text_file:
    for name, values in occurrences.items():
        print(', '.join(str(i) for i in values.values()), file = text_file)

with open("/Users/dalezhou/Desktop/coOccurrenceNodeLabels.txt", "w") as text_file:
    print(my_words, file = text_file)