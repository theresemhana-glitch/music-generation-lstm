import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

with open("data/prepared.pkl", "rb") as f:
    X, y, pitch_names, n_vocab = pickle.load(f)

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dense(n_vocab, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

checkpoint = ModelCheckpoint("data/model.h5", monitor="loss", save_best_only=True, verbose=1)

model.fit(X, y, epochs=30, batch_size=64, callbacks=[checkpoint])
print("Training finished. Model saved to data/model.h5")
