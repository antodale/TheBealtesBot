import tarfile
import os
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from music21 import roman, key as m21key, chord as m21chord, pitch as m21pitch
import matplotlib.pyplot as plt
import numpy as np
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
import torch
import csv
from collections import defaultdict, Counter

# print("Current working directory:", os.getcwd())
# Ensure daft is installed and supports model.to_daft()
# If not, consider skipping the plotting or using networkx for structure plots
df = pd.read_csv("project/chords_dataframe.csv")
print(df.head())

def train_paul(df):
    paul_mcbayes = DiscreteBayesianNetwork([
        ('p1', 'deg'),
        ('p2', 'deg'),
        ('p1c', 'deg'),                   
        ('p2c', 'deg'),

        ('p1', 'color'),
        ('p2', 'color'),
        ('p1c', 'color'),
        ('p2c', 'color'),

        ('deg', 'color')
    ])

    paul_mcbayes.fit(
        df,
        estimator=BayesianEstimator,
        prior_type='BDeu',
        equivalent_sample_size=10
    )

    return paul_mcbayes

model_paul = train_paul(df)
print("MODEL PAUL COMPUTED")



def predict_paul(model, p2, c2, p1, c1, print_result=True):
    inference = VariableElimination(model)
    q = inference.query(
        variables=['deg', 'color'],
        evidence={
            'p2': p2,
            'p2c': c2,
            'p1': p1,
            'p1c': c1,
        }
    )

    # Extract variables and values
    assignments = q.state_names
    values = q.values.flatten()
    degs = assignments['deg']
    colors = assignments['color']

    results = []
    for i, prob in enumerate(values):
        deg_idx = i // len(colors)
        col_idx = i % len(colors)
        label = f"{degs[deg_idx]} ({colors[col_idx]})"
        results.append((label, prob))

    # Convert to dict and normalize
    prob_dict = {}
    for label, prob in results:
        prob_dict[label] = prob_dict.get(label, 0) + prob

    # Sort and get top 5
    top5 = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5])

    if print_result:
        for label, prob in top5.items():
            print(f"{label}: {prob:.4f}")

    return top5


def predict_george(model, p2, c2, p1, c1, print_result=True, sample_size=1000):
    sampler = BayesianModelSampling(model)

    try:
        evidence = [
            ('p1', p1),
            ('p1c', c1),
            ('p2', p2),
            ('p2c', c2)
        ]

        samples = sampler.likelihood_weighted_sample(
            evidence=evidence,
            size=sample_size
        )

        samples['label'] = samples['deg'] + ' (' + samples['color'] + ')'
        top = samples['label'].value_counts(normalize=True).head(5)

        if print_result:
            print(top)
        print(top.to_dict())
        return top.to_dict()

    except Exception as e:
        print("Sampling failed:", e)
        return {}

def markov_seq_generator(path):
    df = pd.read_csv(path)
    df["chord_token"] = df["deg"] + "_" + df["color"]
    valid_rows = df.dropna(subset=["p1", "p1c", "p2", "p2c", "prev3", "prev3_color"])
    sequences = []
    for _, group in valid_rows.groupby("file"):
        chord_seq = group["chord_token"].tolist()
        if len(chord_seq) >= 4:
            sequences.append(chord_seq)
    return sequences

sequences = markov_seq_generator("project/chords_dataframe.csv")


# Get the vocabulary of unique chord tokens
vocab = sorted(set(token for seq in sequences for token in seq))
# print(len(vocab))
vocab_index = {token: i for i, token in enumerate(vocab)}

# Convert sequences of tokens to sequences of indices (integers)
indexed_sequences = [[vocab_index[token] for token in seq] for seq in sequences]

formatted_sequences = [np.array(seq).reshape(-1, 1) for seq in indexed_sequences]

# print(indexed_sequences)
# Number of states and vocabulary size
n_components = 3
n_vocab = len(vocab)


# Create a list of Categorical distributions, one per state
distribution_tonic = Categorical([[1/n_vocab for _ in range(n_vocab)]])
print(f"Example of probability distribution of values wrt to a class: {distribution_tonic.probs}")
distribution_dominant = Categorical([[1/n_vocab for _ in range(n_vocab)]])
distribution_subdominant = Categorical([[1/n_vocab for _ in range(n_vocab)]])


start_probs = [1/n_components for _ in range(n_components)]
# print(start_probs)
# Initialize DenseHMM
model_ringo = DenseHMM(
    distributions=[distribution_tonic, distribution_dominant, distribution_subdominant],
    edges=[
        [0.25, 0.50, 0.25],
        [0.35, 0.25, 0.40],
        [0.50, 0.35, 0.15]
    ],
    starts=start_probs,
    ends=start_probs )


# Fit the model
model_ringo.fit(formatted_sequences)
print("MODEL RINGO COMPUTED")

index_to_vocab = {i: token for token, i in vocab_index.items()}

def predict_ringo(model, p2, c2, p1, c1, vocab_index = vocab_index, index_to_vocab=index_to_vocab, print_result=True):
    # Convert chords to indices
    chord1 = p1 + "_" + c1
    chord2 = p2 + "_" + c2
    try:
        idx1 = vocab_index[chord1]
        idx2 = vocab_index[chord2]
    except KeyError:
        raise ValueError(f"Unknown chord(s): {chord1}, {chord2}")

    # Create input tensor
    input_list = [[[idx1], [idx2]]]
    input_tensor = torch.tensor(input_list, dtype=torch.int64)

    # Run forward-backward algorithm
    transitions, responsibilities, starts, ends, logp = model.forward_backward(X=input_tensor)

    # Get most likely current state (at second position)
    current_state = torch.argmax(responsibilities[0, 1]).item()

    # Predict next state
    transition_probs = model.edges[current_state]
    next_state = torch.argmax(transition_probs).item()

    # Get top 5 emission probabilities for next state
    emission_probs = model.distributions[next_state].probs
    top_values, top_indices = torch.topk(emission_probs, 5)

    # Build dictionary: label → probability
    top5 = {}
    for i in range(5):
        label = index_to_vocab[int(top_indices[0][i])]
        prob = float(top_values[0][i])
        top5[label] = prob

    if print_result:
        for label, prob in top5.items():
            print(f"{label}: {prob:.4f}")
    print(top5)
    return top5

def generate_trigrams_from_file(filepath):

    trigrams = []
    current_sequence = []
    last_file = None

    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file = row['file']
            deg = row['deg']
            color = row['color']
            chord = f"{deg}_{color}"

            if file != last_file and len(current_sequence) >= 3:
                for i in range(len(current_sequence) - 2):
                    trigrams.append(tuple(current_sequence[i:i+3]))
                current_sequence = []

            current_sequence.append(chord)
            last_file = file

        if len(current_sequence) >= 3:
            for i in range(len(current_sequence) - 2):
                trigrams.append(tuple(current_sequence[i:i+3]))

    return trigrams

# Example usage:
trigrams = generate_trigrams_from_file("project/chords_dataframe.csv")
    
class MarkovChainPredictor:
    def __init__(self):
        self.transition_counts = defaultdict(Counter)

    def fit(self, trigrams):
        for chord1, chord2, chord3 in trigrams:
            self.transition_counts[(chord1, chord2)][chord3] += 1



def predict_john(model, p2, c2, p1, c1, top_k=5, print_result=False):
    bigram = (p2 + "_" + c2, p1 + "_" + c1)
    counts = model.transition_counts.get(bigram, None)

    if not counts:
        print("Bigram not found in training data.")
        return {}

    total = sum(counts.values())
    probs = [(chord, count / total) for chord, count in counts.items()]
    probs.sort(key=lambda x: x[1], reverse=True)
    top_probs = probs[:top_k]

    # Split each chord key at the underscore
    result_dict = {tuple(chord.split('_')): prob for chord, prob in top_probs}

    if print_result:
        for chord_tuple, prob in result_dict.items():
            print(f"{chord_tuple}: {prob:.4f}")
    print(result_dict)
    return result_dict



model_john = MarkovChainPredictor()
model_john.fit(trigrams)
print("MODEL JOHN COMPUTED")





print("****COMPUTATION COMPLETED*****")