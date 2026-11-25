import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical

SEQ_LEN = 50

with open("data/notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitch_names = sorted(set(notes))
n_vocab = len(pitch_names)
note_to_int = {n: i for i, n in enumerate(pitch_names)}
print("Unique notes:", n_vocab)

inputs = []
outputs = []
for i in range(len(notes) - SEQ_LEN):
    seq_in = notes[i:i + SEQ_LEN]
    seq_out = notes[i + SEQ_LEN]
    inputs.append([note_to_int[n] for n in seq_in])
    outputs.append(note_to_int[seq_out])

n_patterns = len(inputs)
print("Training examples:", n_patterns)

X = np.reshape(inputs, (n_patterns, SEQ_LEN, 1))
X = X / float(n_vocab)
y = to_categorical(outputs, num_classes=n_vocab)

print("X shape:", X.shape)
print("y shape:", y.shape)

with open("data/prepared.pkl", "wb") as f:
    pickle.dump((X, y, pitch_names, n_vocab), f)
print("Saved data/prepared.pkl")
