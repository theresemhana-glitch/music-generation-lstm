import pickle
import numpy as np
from tensorflow.keras.models import load_model
from music21 import note, chord, stream, instrument

SEQ_LEN = 50
NUM_NOTES = 300
TEMPERATURE = 0.8

with open("data/notes.pkl", "rb") as f:
    notes = pickle.load(f)
with open("data/prepared.pkl", "rb") as f:
    X, y, pitch_names, n_vocab = pickle.load(f)

note_to_int = {n: i for i, n in enumerate(pitch_names)}
int_to_note = {i: n for i, n in enumerate(pitch_names)}

model = load_model("data/model.h5")

start = np.random.randint(0, len(notes) - SEQ_LEN)
pattern = [note_to_int[n] for n in notes[start:start + SEQ_LEN]]

def sample(preds, temp):
    preds = np.log(preds + 1e-9) / temp
    probs = np.exp(preds)
    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)

generated = []
for i in range(NUM_NOTES):
    x = np.reshape(pattern, (1, SEQ_LEN, 1)) / float(n_vocab)
    preds = model.predict(x, verbose=0)[0]
    idx = sample(preds, TEMPERATURE)
    generated.append(int_to_note[idx])
    pattern.append(idx)
    pattern = pattern[1:]

output_notes = []
offset = 0
for p in generated:
    if "." in p or p.isdigit():
        chord_notes = []
        for s in p.split("."):
            n = note.Note(int(s) + 60)
            n.storedInstrument = instrument.Piano()
            chord_notes.append(n)
        new_item = chord.Chord(chord_notes)
    else:
        new_item = note.Note(p)
        new_item.storedInstrument = instrument.Piano()
    new_item.offset = offset
    output_notes.append(new_item)
    offset += 0.5

stream.Stream(output_notes).write("midi", fp="output.mid")
print("Saved output.mid")
print("First 20 generated notes:", generated[:20])