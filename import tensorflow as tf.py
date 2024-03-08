import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

training_data = ["frase1",
                 "frase2",
                 "frase3"]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(training_data)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(training_data)
max_sequence_len = max([len(x) for x in sequences])
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

model = Sequential([
    Embedding(vocab_size, 10, input_length=max_sequence_len-1),
    LSTM(50),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(padded_sequences[:,:-1], tf.keras.utils.to_categorical(padded_sequences[:,-1], num_classes=vocab_size), epochs=50)

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='post')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

generated_text = generate_text("exemplo de", 5, model, max_sequence_len)

print(generated_text)
