import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "I love coding with Python",
    "Natural Language Processing is amazing",
    "Generative models are fascinating",
    "Deep learning is revolutionizing NLP"
]

# Tokenize the text and convert to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure they have the same length
max_sequence_length = max(len(sequence) for sequence in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Prepare data for language modeling
X, y = [], []
for sequence in padded_sequences:
    for i in range(1, len(sequence)):
        X.append(sequence[:i])
        y.append(sequence[i:i+1])  # Change this line to store each y value as a separate list

# Convert X to a TensorFlow ragged tensor
X_ragged = tf.ragged.constant([sequence for sequence in X])

# Pad X to have the same length as the longest sequence
X_ragged = X_ragged.to_tensor()

# Pad y sequences to have the same length as X sequences
y_padded = pad_sequences(y, maxlen=max_sequence_length - 1)

# Define the dimensions of the LSTM layer
input_dim = len(tokenizer.word_index) + 1
embedding_dim = 32
lstm_units = 64

# Build the language model architecture
input_layer = Input(shape=(None,))
embedding_layer = Embedding(input_dim, embedding_dim)(input_layer)
lstm_layer = LSTM(lstm_units, return_sequences=True)(embedding_layer)
output_layer = Dense(input_dim, activation='softmax')(lstm_layer)

language_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with sparse_categorical_crossentropy loss
language_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Create TensorFlow datasets from the tensors
dataset = tf.data.Dataset.from_tensor_slices((X_ragged, y_padded))

# Shuffle and batch the dataset
batch_size = 1
dataset = dataset.shuffle(len(X)).batch(batch_size)

# Train the language model using the dataset
language_model.fit(dataset, epochs=100)

# Function to generate text using the trained language model
def generate_text(seed_text, max_length=10):
    seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    generated_sequence = tf.ragged.constant([seed_sequence])
    for _ in range(max_length):
        next_word_probs = language_model.predict(generated_sequence.to_tensor())
        next_token_index = np.random.choice(len(next_word_probs[0][-1]), p=next_word_probs[0][-1])
        next_token = tf.constant([[next_token_index]], dtype=tf.int32)
        generated_sequence = tf.concat([generated_sequence, next_token], axis=1)
    generated_text = tokenizer.sequences_to_texts(generated_sequence.to_tensor().numpy())[0]
    return generated_text

# Generate text using the trained language model
seed_text = "Deep learning"
generated_text = generate_text(seed_text, max_length=10)
print(f"Seed Text: {seed_text}")
print(f"Generated Text: {generated_text}")