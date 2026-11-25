import os
from music21 import corpus

os.makedirs("data/midi", exist_ok=True)

paths = corpus.getComposer("bach")[:100]   # take the first 100 pieces

for i, path in enumerate(paths):
    try:
        score = corpus.parse(path)
        score.write("midi", fp=f"data/midi/bach_{i}.mid")
        print("Saved", i)
    except Exception as e:
        print("Skipped", i, e)