import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# ------------------ Custom Attention Layer ------------------
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, hidden_dim)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.V = self.add_weight(name='att_var', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Compute score using tanh activation
        score = K.tanh(K.dot(inputs, self.W) + self.b)
        # Compute attention weights with softmax across the time dimension
        attention_weights = K.softmax(K.dot(score, self.V), axis=1)
        # Compute context vector as weighted sum of inputs
        context_vector = attention_weights * inputs
        context_vector = K.sum(context_vector, axis=1)
        return context_vector
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# ------------------ Updated Baseline Model with Attention ------------------
def build_baseline_model(vocab_size, embedding_dim=128, lstm_units=200, dropout_rate=0.05):
    inputs = Input(shape=(max_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    # Use Bidirectional LSTM with increased lstm_units and dropout; return sequences for attention
    bilstm = Bidirectional(LSTM(lstm_units, dropout=dropout_rate, return_sequences=True))(embedding)
    # Add the custom Attention layer
    attention = AttentionLayer()(bilstm)
    # Dense layer after attention
    dense = Dense(64, activation='relu')(attention)
    output = Dense(1, activation='sigmoid')(dense)  # Binary classification

    model = Model(inputs, output)
    # Updated optimizer with learning rate 0.001 as per the paper
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def preprocess_urls(urls, tokenizer, max_length):
    # Convert each URL to a list of characters and tokenize
    char_sequences = [list(url) for url in urls]
    sequences = tokenizer.texts_to_sequences(char_sequences)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded

def s_phs(url_list, model, tokenizer, max_length):
    inputs = preprocess_urls(url_list, tokenizer, max_length)
    predictions = (model.predict(inputs) > 0.5).astype("int32")
    # Here, if any URL is predicted as phishing (assumed to be 0), we flag the page.
    # Adjust logic if your label convention is different.
    # For now, we return 1 if any prediction is 1, meaning phishing detected.
    return int(any(predictions))

def mss(url_list, model, tokenizer, max_length, threshold=0.6):
    inputs = preprocess_urls(url_list, tokenizer, max_length)
    preds = model.predict(inputs).flatten()
    mean_score = np.mean(preds)
    return int(mean_score >= threshold)

def weas(url_dict, model, tokenizer, max_length, threshold=0.6):
    urls = list(url_dict.keys())
    weights = list(url_dict.values())
    inputs = preprocess_urls(urls, tokenizer, max_length)
    preds = model.predict(inputs).flatten()
    weighted_sum = np.sum(np.array(preds) * np.array(weights))
    total_weight = np.sum(weights)
    weas_score = weighted_sum / total_weight
    return int(weas_score >= threshold)

'''----------------------------MAIN-----------------------------------'''
# Load the dataset
print("Loading dataset...")
data = pd.read_csv('Cleaned_Phishing_Dataset.csv')
print("Dataset loaded successfully!\n")
print(data.head())

# Preprocess URLs: Convert each URL into a list of characters
print("Preprocessing URLs...")
data['processed_url'] = data['url'].astype(str).apply(list)
print("URL preprocessing complete.\n")

# Tokenization at character-level
print("Tokenizing data...")
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data['processed_url'])
sequences = tokenizer.texts_to_sequences(data['processed_url'])
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# Determine maximum sequence length and pad sequences
max_length = max(len(seq) for seq in sequences)
print(f"Maximum sequence length: {max_length}")
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = data['label'].values

# Split data into training (60%), validation (20%), and test (20%) sets
print("Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Train set size: {X_train.shape}, Validation set size: {X_val.shape}, Test set size: {X_test.shape}\n")

# Build and Train the Model using the validation set for tuning
baseline_model = build_baseline_model(vocab_size)
baseline_model.summary()  # Prints model architecture with attention
print("Training the model...")
# Updated number of epochs to 100 as per paper; you can adjust if needed
baseline_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# Predictions on the test set (unseen data)
print("Making predictions on the test set...")
y_pred_baseline = (baseline_model.predict(X_test) > 0.5).astype("int32")

# Evaluation
print("Baseline Model Performance on Test Data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline)}")
print(f"Precision: {precision_score(y_test, y_pred_baseline)}")
print(f"Recall: {recall_score(y_test, y_pred_baseline)}")
print(f"F1 Score: {f1_score(y_test, y_pred_baseline)}")

# Decision Strategies
print("\nDecision Strategies:")

urls = ['http://example.com', 'http://suspicious.biz/phish']
weights = {'http://example.com': 0.45, 'http://suspicious.biz/phish': 0.30}

print("SPhS:", s_phs(urls, baseline_model, tokenizer, max_length))
print("MSS:", mss(urls, baseline_model, tokenizer, max_length))
print("WeAS:", weas(weights, baseline_model, tokenizer, max_length))

import unittest

class TestPhishingRTDS(unittest.TestCase):
    def setUp(self):
        # Load and preprocess the new dataset ("new_dataset.csv")
        self.data = pd.read_csv('new_dataset.csv')
        self.data['processed_url'] = self.data['url'].astype(str).apply(list)
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(self.data['processed_url'])
        sequences = self.tokenizer.texts_to_sequences(self.data['processed_url'])
        self.vocab_size = len(self.tokenizer.word_index) + 1
        global max_length
        max_length = max(len(seq) for seq in sequences)
        self.X = pad_sequences(sequences, maxlen=max_length, padding='post')
        self.y = self.data['label'].values
        # Split data: 60% training, 20% validation, 20% test
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        self.X_train, self.y_train = X_train, y_train
        # Build the model as per the research paper
        self.model = build_baseline_model(self.vocab_size)

    def test_model_training(self):
        # Train for 2 epochs to verify the training pipeline (use reduced epochs for quick tests)
        history = self.model.fit(self.X_train, self.y_train, epochs=2, batch_size=32,
                                 validation_data=(self.X_val, self.y_val), verbose=0)
        self.assertIn('loss', history.history)

    def test_model_evaluation(self):
        y_pred = (self.model.predict(self.X_test) > 0.5).astype("int32")
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(prec, float)
        self.assertIsInstance(rec, float)
        self.assertIsInstance(f1, float)
        print("\nEvaluation Metrics on Test Data:")
        print("Accuracy:", acc, "Precision:", prec, "Recall:", rec, "F1 Score:", f1)

    def test_decision_strategies(self):
        sample_urls = ['http://example.com', 'http://suspicious.biz/phish']
        sample_weights = {'http://example.com': 0.45, 'http://suspicious.biz/phish': 0.30}
        s_result = s_phs(sample_urls, self.model, self.tokenizer, max_length)
        m_result = mss(sample_urls, self.model, self.tokenizer, max_length, threshold=0.6)
        w_result = weas(sample_weights, self.model, self.tokenizer, max_length, threshold=0.6)
        self.assertIn(s_result, [0, 1])
        self.assertIn(m_result, [0, 1])
        self.assertIn(w_result, [0, 1])
        print("\nDecision Strategies Results:")
        print("SPhS:", s_result, "MSS:", m_result, "WeAS:", w_result)


