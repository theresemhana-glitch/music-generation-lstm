import wave
import numpy as np
from music21 import converter

SR = 22050
SEC_PER_BEAT = 0.5

score = converter.parse("my_song_1.mid")

events = []
for el in score.flat.notes:
    if el.isChord:
        freqs = [p.frequency for p in el.pitches]
    else:
        freqs = [el.pitch.frequency]
    events.append((float(el.offset) * SEC_PER_BEAT,
                   float(el.quarterLength) * SEC_PER_BEAT,
                   freqs))

total = max(s + d for s, d, f in events) + 1.0
audio = np.zeros(int(total * SR))

for start, dur, freqs in events:
    n = int((dur + 0.5) * SR)
    t = np.arange(n) / SR
    env = np.exp(-3 * t)
    env[:200] *= np.linspace(0, 1, 200)
    i = int(start * SR)
    for f in freqs:
        tone = (np.sin(2 * np.pi * f * t)
                + 0.4 * np.sin(4 * np.pi * f * t)
                + 0.2 * np.sin(6 * np.pi * f * t))
        piece = tone * env * 0.2
        piece = piece[:len(audio) - i]
        audio[i:i + len(piece)] += piece

audio = audio / np.max(np.abs(audio))
data = (audio * 32767 * 0.9).astype(np.int16)

with wave.open("my_song_1.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(data.tobytes())

print("Saved my_song_1.wav")
