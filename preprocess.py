import glob
import pickle
from music21 import converter, note, chord

notes = []

files = glob.glob("data/midi/*.mid")
print("Found", len(files), "files")

for file in files:
    midi = converter.parse(file)
    count_before = len(notes)

    for element in midi.flat.notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))

    print("Done", file, "->", len(notes) - count_before, "notes")

with open("data/notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("Total notes:", len(notes))
print("First 20 notes:", notes[:20])
