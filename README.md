\# Music Generation with AI (LSTM)



\## Goal

Train a neural network on classical music and use it to generate new piano music.



\## Steps



\### 1. Data collection (get\_data.py)

I used the built-in corpus of the music21 library and exported 100 Bach pieces as MIDI files into data/midi.



\### 2. Preprocessing (preprocess.py)

Each MIDI file was read with music21. Every note became a text label (for example "F4"), and every chord became its pitch numbers (for example "4.7.11"). In total I extracted 32,701 notes.



\### 3. Preparing training data (prepare.py)

Each unique note was mapped to a number. I then built training examples: 50 notes in a row as input and the next note as the answer, which gave 32,651 examples.



\### 4. Model (train.py)

The model is a Keras LSTM network:

\- LSTM (128 units) + Dropout 0.3

\- LSTM (128 units) + Dropout 0.3

\- Dense (128, ReLU)

\- Dense (softmax output, one unit per unique note)



It has 221,751 parameters. I trained it for 30 epochs with batch size 64, using the Adam optimizer and categorical cross-entropy loss. The loss went down during training, which shows the model learned patterns in the music. The best model was saved to data/model.h5.



\### 5. Music generation (generate.py)

I picked 50 random notes from the data as a starting point. The model predicts the next note, adds it to the sequence, and repeats this 300 times. I used temperature sampling (0.8) so the music is not too repetitive. The generated notes were written to a MIDI file (my\_song\_1.mid).



\### 6. Audio (make\_wav.py)

The MIDI file was converted to a WAV file (my\_song\_1.wav) using a simple synthesizer written with NumPy.



\## Results

The model generates piano music in a Bach-like style.



\## Limitations

All generated notes have the same length, because the model only learned which note comes next, not rhythm or duration. Future work: also learn note durations and train on more data.



\## How to run

python get\_data.py

python preprocess.py

python prepare.py

python train.py

python generate.py

python make\_wav.py

