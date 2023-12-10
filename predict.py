import os
import re
import numpy as np
import tensorflow as tf
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import unicodedata
import pickle
import warnings
warnings.filterwarnings('ignore')

var_dir = './var/'

with open(os.path.normpath(var_dir+'src_lang_tokenizer.pkl'), 'rb') as f:
    src_lang_tokenizer = pickle.load(f)

with open(os.path.normpath(var_dir+'tgt_lang_tokenizer.pkl'), 'rb') as f:
    tgt_lang_tokenizer = pickle.load(f)

max_length_src = 53
max_length_trg = 51
src_vocab_size = 24794
tgt_vocab_size = 12934

# Defining hyperparameters
buffer_size = 95171
val_buffer_size = 23793
BATCH_SIZE = 108
embedding_dim = 128
units = 1024
steps_per_epoch = buffer_size//BATCH_SIZE
val_steps_per_epoch = val_buffer_size//BATCH_SIZE


def unicode_to_ascii(s):
    normalized = unicodedata.normalize('NFD', s)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')


def preprocess_text(text):
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = text.rstrip().strip()
    text = '<sos> ' + text + ' <eos>'

    return text


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, emb_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_sz = batch_sz
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, emb_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

encoder_entry = np.load(os.path.normpath(var_dir+'encoder_entry.npy'),allow_pickle=True)
encoder = Encoder(src_vocab_size, embedding_dim, units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(encoder_entry, sample_hidden)
encoder.load_weights(os.path.normpath(var_dir+'encoder_weights.h5')) 


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)  # fully-connected dense layer-1
        self.W2 = tf.keras.layers.Dense(units)  # fully-connected dense layer-2
        self.V = tf.keras.layers.Dense(1)  # fully-connected dense layer-3

    def call(self, query, values):

        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        
        self.attention = BahdanauAttention(self.dec_units)

        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)

        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


decoder = Decoder(tgt_vocab_size, embedding_dim, units, BATCH_SIZE)
sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),sample_hidden, sample_output)
decoder.load_weights(os.path.normpath(var_dir+'decoder_weights.h5'))


def evaluate(sentence):
    sentence = preprocess_text(sentence)
    inputs = [src_lang_tokenizer.word_index.get(i, 0) for i in sentence.split(' ')]
    inputs = pad_sequences([inputs], maxlen=max_length_src, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tgt_lang_tokenizer.word_index['<sos>']], 0)

    for t in range(max_length_trg):
        predictions, dec_hidden, _ = decoder(dec_input,dec_hidden,enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += tgt_lang_tokenizer.index_word[predicted_id] + ' '

        if tgt_lang_tokenizer.index_word[predicted_id] == '<eos>':
            return result, sentence
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def translate(sentence):
    result, sentence = evaluate(sentence)
    print('Predicted Translation: ', result)


translate(input('\n\nSpanish: '))
