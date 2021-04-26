import os
import re
import time
import string
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import unicodedata
# from encoder import Encoder
# from decoder import Decoder

DATA_PATH = './mar.txt'
CHECKPOINT_DIR = './checkpoints'
EPOCHS = 40

def fetch_data():
    lines = pd.read_table(DATA_PATH, encoding='utf-8', names = ['eng', 'mar'])
    lines["english"] = lines.index
    lines["marathi"] = lines.eng

    lines.drop(["eng", "mar"], axis = 1, inplace = True)
    lines = lines.reset_index(drop = True)

    return lines

def unicode_to_ascii(sentence):
  return "".join(c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn')

def preprocessing(sentence, language = False):
  sentence = unicode_to_ascii(sentence.lower().strip())

  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+ ', " ", sentence )
  sentence = re.sub(r"'", r'', sentence)
  
  exclude = set(string.punctuation)
  remove_digits = str.maketrans('', '', string.digits)
  sentence = "".join(ch for ch in sentence if ch not in exclude)
  sentence = sentence.translate(remove_digits)

  sentence = re.sub(r"[२३०८१५७९४६]", "", sentence)
  sentence = re.sub(r" +", r" ", sentence)
  sentence = sentence.strip()
  sentence = "<start> " + sentence + " <end>"
  return sentence

def create_dataset(file_name):
  file_name.english = file_name.english.apply(lambda x : preprocessing(x, False))
  file_name.marathi = file_name.marathi.apply(lambda x : preprocessing(x, True))
  return file_name.english, file_name.marathi

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
  return tensor, lang_tokenizer

def load_dataset(lines):
  target_lang, input_lang = create_dataset(lines)
  input_tensor, inp_lang_tokenizer = tokenize(input_lang)
  target_tensor, targ_lang_tokenizer = tokenize(target_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def convert(lang, tensor):
  for t in tensor:
    if t != 0:
      print(f'{t} ------> {lang.index_word[t]}')

def get_model_metrics():
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    return optimizer, loss_function

def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocessing(sentence)

  inputs = [inp_lang_tokenizer.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, UNITS))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang_tokenizer.index_word[predicted_id] + ' '

    if targ_lang_tokenizer.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot   

def translate(sentence):
  result, sentence, _ = evaluate(sentence)

  print('Input:', sentence)
  print('Predicted translation:', result) 


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

if __name__ == "__main__":
    print("Training language translator model..( ENG --> MAR ) ")
    print("Checking if GPU device exist")
    try:
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError("GPU device not found")
        else:
            print("Executing model with GPU...{}".format(device_name))
    except Exception as ex:
        print("Exception occurred %s", ex)
        print("Continueing with CPU training....")

    # Fetch training data 
    data = fetch_data()

    # Generate training tensor and tokenizer
    input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset(data)
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

    # Split training and testing dataset for model training
    input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test = train_test_split(input_tensor, target_tensor, test_size = 0.2)

    # Check data format for further processing.
    print("Input language: Index to word mapping...")
    convert(inp_lang_tokenizer, input_tensor_train[0])
    print("Target Language; index to word mapping")
    convert(targ_lang_tokenizer, target_tensor_train[0])

    # set model parameters..
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    EPOCH_SIZE = len(input_tensor_train) // BATCH_SIZE
    EMBEDING_DIM = 256
    UNITS = 1024

    VOCAB_INPUT_SIZE = len(inp_lang_tokenizer.word_index) + 1
    VOCAB_TAR_SIZE = len(targ_lang_tokenizer.word_index) + 1

    # Create tensor dataset
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)

    # Generate NMT model
    encoder = Encoder(VOCAB_INPUT_SIZE, EMBEDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(VOCAB_TAR_SIZE, EMBEDING_DIM, UNITS, BATCH_SIZE)

    optimizer, loss_function = get_model_metrics()

    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(EPOCH_SIZE)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    # Prediction..
    translate("what is your name")
