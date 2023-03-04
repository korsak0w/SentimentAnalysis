import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from sklearn import metrics
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from collections import Counter



######################
# Upload Dataset
######################

def create_default_df():
    df1 = pd.read_csv('data/IMDB_Dataset_Part1.csv')
    df2 = pd.read_csv('data/IMDB_Dataset_Part2.csv')
    df3 = pd.read_csv('data/IMDB_Dataset_Part2.csv')
    df4 = pd.read_csv('data/IMDB_Dataset_Part4.csv')
    frames = [df1, df2, df3, df4]
    df = pd.concat(frames)
    df = df.iloc[:, -2:]
    return df

def create_pd_info(df):
    buffer = io.StringIO()
    st.session_state['df'].info(buf=buffer)
    s = buffer.getvalue()
    return s


######################
# Preprocessing
######################

def preprocess(X_batch, y_batch):
    # limit the input length (optional)
    X_batch = tf.strings.substr(X_batch, 0, 300)
    # replaces HTML tags and non-alphabetic or non-apostrophe characters 
    X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    # splits each input string into a sequence of tokens -> nested tensor of variable lengths (arXiv:1609.08144)
    X_batch = tf.strings.split(X_batch)
    # converts the nested tensor of tokens to a dense tensor of fixed shape (model requires inputs of fixed shape)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch

def create_vocabulary(dataset, batch_size, vocab_size):
    vocabulary = Counter()
    for X_batch, y_batch in dataset.batch(batch_size).map(preprocess):
        for review in X_batch:
            vocabulary.update(list(review.numpy()))
    truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]
    return truncated_vocabulary

def create_lookup_table(vocabulary, num_oov_buckets):
    words = tf.constant(vocabulary)
    word_ids = tf.range(len(vocabulary), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)
    return table

def encode_words(X_batch, y_batch, table):
    return table.lookup(X_batch), y_batch

def encode_labels(dataset, column_y):
    labels = dataset[column_y].value_counts()
    mapping = {labels.index[1]: 0, labels.index[0]: 1}
    dataset[column_y] = dataset[column_y].map(mapping)
    return dataset

def create_tensorflow_dataset(dataset, column_X, column_y):
    X_col = dataset[column_X].tolist()
    y_col = dataset[column_y].tolist()
    tf_dataset = tf.data.Dataset.from_tensor_slices((X_col, y_col))
    tf_dataset = tf_dataset.map(lambda x, y: (x, tf.cast(y, tf.int64))).prefetch(1)
    return tf_dataset

def split_dataset(df, test_train_split):
    train_len = round(len(df) * test_train_split)
    splitted = df[:train_len]
    test_set = df[train_len:]
    val_length = round(len(splitted) * 0.2)
    train_set = splitted[val_length:]
    val_set = splitted[:val_length]
    return train_set, val_set, test_set

def encode_datasets(datasets, batch_size, table):
    encoded_datasets = []
    for set in datasets:
        prep_set = set.batch(batch_size).map(preprocess)
        encoded_set = prep_set.map(lambda x, y: encode_words(x, y, table)).prefetch(1)
        encoded_datasets.append(encoded_set)
    train, val, test = encoded_datasets
    return train, val, test


######################
# Build Model
######################

activation_functions = [None, "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
weight_initializers = ['random_normal', 'random_uniform', 'truncated_normal', 'zeros', 'ones', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'identity', 'orthogonal']
weight_regularizers = [None, 'l1', 'l2', 'l1_l2']

def create_infos(layer_type, num_layer, input_dim, init):
    with st.expander('Hyperparameters'):
        if layer_type == 'Dense Layer':
            hyper_params = dense_params(num_layer, init)
            return hyper_params
        elif layer_type == 'Embedding Layer':
            hyper_params = embedding_params(num_layer, input_dim, init)
            return hyper_params
        elif layer_type == 'Simple Recurrent Neural Network Layer':
            hyper_params = simple_rnn_params(num_layer)
            return hyper_params
        elif layer_type == 'Long Short-Term Memory Layer':
            hyper_params = lstm_params(num_layer)
            return hyper_params
        else:
            hyper_params = gru_params(num_layer, init)
            return hyper_params


def dense_params(num_layer, init):
    col1, col2 = st.columns(2)
    if init:
        units =  col1.number_input('Number of Units', step=1, value=1, key=f'units_{num_layer}')
        activation = col2.selectbox('Activation Function', activation_functions, index=2, key=f'activation_{num_layer}')
    else:
        units =  col1.number_input('Number of Units', step=1, key=f'units_{num_layer}')
        activation = col2.selectbox('Activation Function', activation_functions, key=f'activation_{num_layer}')

    col3, col4 = st.columns(2)
    bias = col3.selectbox('Use Bias', (True, False), key=f'bias_{num_layer}')
    kernel_initializer = col4.selectbox('Kernel Initializer', weight_initializers, index=6, key=f'kernel_initializer_{num_layer}')

    if bias:
        col5, col6 = st.columns(2)
        bias_initializer = col5.selectbox('Bias Initializer', weight_initializers, index=3, key=f'bias_initializer_{num_layer}')
        bias_regularizer = col6.selectbox('Bias Regularizer', weight_regularizers, key=f'bias_regularizer_{num_layer}')
    else:
        bias_initializer = None
        bias_regularizer = None

    col7, col8 = st.columns(2)
    kernel_regularizer = col7.selectbox('Kernel Regularizer', weight_regularizers, key=f'kernel_regularizer_{num_layer}')
    activity_regularizer = col8.selectbox('Activity Regularizer', weight_regularizers, key=f'activity_regularizer_{num_layer}')

    return {
        'layer': 'Dense',
        'units': units,
        'activation': activation,
        'use_bias': bias,
        'kernel_initializer': kernel_initializer,
        'bias_initializer': bias_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'activity_regularizer': activity_regularizer,
        'kernel_constraint': None,
        'bias_constraint': None,
    }

def embedding_params(num_layer, input_dim, init):
    col1, col2 = st.columns(2)
    if init:
        output_dim = col1.number_input('Embedding Dimensions', step=1, value=128, key=f'output_dim_{num_layer}')
    else:
        output_dim = col1.number_input('Embedding Dimensions', step=1, key=f'output_dim_{num_layer}')
    mask_zero = col2.selectbox('Mask Zero', (False, True), key=f'mask_zero_{num_layer}')

    return {
        'layer': 'Embedding',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'embeddings_initializer': "uniform",
        'embeddings_regularizer': None,
        'activity_regularizer': None,
        'embeddings_constraint': None,
        'mask_zero': mask_zero,
        'input_length': None,
        'input_shape': [None],
    }

def simple_rnn_params(num_layer):
    col1, col2 = st.columns(2)
    units =  col1.number_input('Number of Units', step=1, key=f'units_{num_layer}')
    activation = col2.selectbox('Activation Function', activation_functions, index=6, key=f'activation_{num_layer}')

    col3, col4 = st.columns(2)
    bias = col3.selectbox('Use Bias', (True, False), key=f'bias_{num_layer}')
    kernel_initializer = col4.selectbox('Kernel Initializer', weight_initializers, index=6, key=f'kernel_initializer_{num_layer}')

    if bias:
        col5, col6 = st.columns(2)
        bias_initializer = col5.selectbox('Bias Initializer', weight_initializers, index=3, key=f'bias_initializer_{num_layer}')
        bias_regularizer = col6.selectbox('Bias Regularizer', weight_regularizers, key=f'bias_regularizer_{num_layer}')
    else:
        bias_initializer = None
        bias_regularizer = None

    col7, col8 = st.columns(2)
    recurrent_initializer = col7.selectbox('Recurrent Initializer', weight_initializers, index=10, key=f'recurrent_initializer_{num_layer}')
    recurrent_regularizer = col8.selectbox('Recurrent Regularizer', weight_regularizers, key=f'recurrent_regularizer_{num_layer}')

    col9, col10 = st.columns(2)
    kernel_regularizer = col9.selectbox('Kernel Regularizer', weight_regularizers, key=f'kernel_regularizer_{num_layer}')
    activity_regularizer = col10.selectbox('Activity Regularizer', weight_regularizers, key=f'activity_regularizer_{num_layer}')

    col11, col12 = st.columns(2)
    dropout = col11.slider('Dropout', 0.0, 1.0, step=0.05, key=f'dropout_{num_layer}')
    recurrent_dropout = col12.slider('Recurrent Dropout', 0.0, 1.0, step=0.05, key=f'recurrent_dropout_{num_layer}')

    col13, col14 = st.columns(2)
    return_sequences = col13.selectbox('Return Sequences', (False, True), key=f'return_sequences_{num_layer}')
    return_state = col14.selectbox('Return State', (False, True), key=f'return_state_{num_layer}')

    col15, col16 = st.columns(2)
    go_backwards = col15.selectbox('Go Backwards', (False, True), key=f'go_backwards_{num_layer}')
    stateful = col16.selectbox('Stateful', (False, True), key=f'stateful_{num_layer}')

    return {
        'layer': 'SimpleRNN',
        'units': units,
        'activation': activation,
        'use_bias': bias,
        'kernel_initializer': kernel_initializer,
        'recurrent_initializer': recurrent_initializer,
        'bias_initializer': bias_initializer,
        'kernel_regularizer': kernel_regularizer,
        'recurrent_regularizer': recurrent_regularizer,
        'bias_regularizer': bias_regularizer,
        'activity_regularizer': activity_regularizer,
        'kernel_constraint': None,
        'recurrent_constraint': None,
        'bias_constraint': None,
        'dropout': dropout,
        'recurrent_dropout': recurrent_dropout,
        'return_sequences': return_sequences,
        'return_state': return_state,
        'go_backwards': go_backwards,
        'stateful': stateful,
        'unroll': False,
    }

def lstm_params(num_layer):
    col1, col2 = st.columns(2)
    units =  col1.number_input('Number of Units', step=1, key=f'units_{num_layer}')
    activation = col2.selectbox('Activation Function', activation_functions, index=6, key=f'activation_{num_layer}')

    col3, col4 = st.columns(2)
    recurrent_activation = col3.selectbox('Recurrent Activation Function', activation_functions, index=2, key=f'recurrent_activation_{num_layer}')
    bias = col4.selectbox('Use Bias', (True, False), key=f'bias_{num_layer}')

    if bias:
        col5, col6 = st.columns(2)
        bias_initializer = col5.selectbox('Bias Initializer', weight_initializers, index=3, key=f'bias_initializer_{num_layer}')
        bias_regularizer = col6.selectbox('Bias Regularizer', weight_regularizers, key=f'bias_regularizer_{num_layer}')
    else:
        bias_initializer = None
        bias_regularizer = None
    
    col7, col8 = st.columns(2)
    kernel_initializer = col7.selectbox('Kernel Initializer', weight_initializers, index=6, key=f'kernel_initializer_{num_layer}')
    recurrent_initializer = col8.selectbox('Recurrent Initializer', weight_initializers, index=10, key=f'recurrent_initializer_{num_layer}')

    col9, col10 = st.columns(2)
    kernel_regularizer = col9.selectbox('Kernel Regularizer', weight_regularizers, key=f'kernel_regularizer_{num_layer}')
    recurrent_regularizer = col10.selectbox('Recurrent Regularizer', weight_regularizers, key=f'recurrent_regularizer_{num_layer}')

    col11, col12 = st.columns(2)
    activity_regularizer = col11.selectbox('Activity Regularizer', weight_regularizers, key=f'activity_regularizer_{num_layer}')
    unit_forget_bias = col12.selectbox('Unit Forget Bias', (True, False), key=f'unit_forget_bias_{num_layer}')

    col13, col14 = st.columns(2)
    dropout = col13.slider('Dropout', 0.0, 1.0, step=0.05, key=f'dropout_{num_layer}')
    recurrent_dropout = col14.slider('Recurrent Dropout', 0.0, 1.0, step=0.05, key=f'recurrent_dropout_{num_layer}')

    col15, col16 = st.columns(2)
    return_sequences = col15.selectbox('Return Sequences', (False, True), key=f'return_sequences_{num_layer}')
    return_state = col16.selectbox('Return State', (False, True), key=f'return_state_{num_layer}')

    col17, col18 = st.columns(2)
    go_backwards = col17.selectbox('Go Backwards', (False, True), key=f'go_backwards_{num_layer}')
    stateful = col18.selectbox('Stateful', (False, True), key=f'stateful_{num_layer}')

    return {
        'layer': 'LSTM',
        'units': units,
        'activation': activation,
        'recurrent_activation': recurrent_activation,
        'use_bias': bias,
        'kernel_initializer': kernel_initializer,
        'recurrent_initializer': recurrent_initializer,
        'bias_initializer': bias_initializer,
        'unit_forget_bias': unit_forget_bias,
        'kernel_regularizer': kernel_regularizer,
        'recurrent_regularizer': recurrent_regularizer,
        'bias_regularizer': bias_regularizer,
        'activity_regularizer': activity_regularizer,
        'kernel_constraint': None,
        'recurrent_constraint': None,
        'bias_constraint': None,
        'dropout': dropout,
        'recurrent_dropout': recurrent_dropout,
        'return_sequences': return_sequences,
        'return_state': return_state,
        'go_backwards': go_backwards,
        'stateful': stateful,
        'time_major': False,
        'unroll': False,
    }

def gru_params(num_layer, init):
    col1, col2 = st.columns(2)
    if init:
        units =  col1.number_input('Number of Units', step=1, value=128, key=f'units_{num_layer}')
    else:
        units =  col1.number_input('Number of Units', step=1, key=f'units_{num_layer}')
    activation = col2.selectbox('Activation Function', activation_functions, index=6, key=f'activation_{num_layer}')

    col3, col4 = st.columns(2)
    recurrent_activation = col3.selectbox('Recurrent Activation Function', activation_functions, index=2, key=f'recurrent_activation_{num_layer}')
    bias = col4.selectbox('Use Bias', (True, False), key=f'bias_{num_layer}')

    if bias:
        col5, col6 = st.columns(2)
        bias_initializer = col5.selectbox('Bias Initializer', weight_initializers, index=3, key=f'bias_initializer_{num_layer}')
        bias_regularizer = col6.selectbox('Bias Regularizer', weight_regularizers, key=f'bias_regularizer_{num_layer}')
    else:
        bias_initializer = None
        bias_regularizer = None

    col7, col8 = st.columns(2)
    kernel_initializer = col7.selectbox('Kernel Initializer', weight_initializers, index=6, key=f'kernel_initializer_{num_layer}')
    recurrent_initializer = col8.selectbox('Recurrent Initializer', weight_initializers, index=10, key=f'recurrent_initializer_{num_layer}')

    col9, col10 = st.columns(2)
    kernel_regularizer = col9.selectbox('Kernel Regularizer', weight_regularizers, key=f'kernel_regularizer_{num_layer}')
    recurrent_regularizer = col10.selectbox('Recurrent Regularizer', weight_regularizers, key=f'recurrent_regularizer_{num_layer}')

    col11, col12 = st.columns(2)
    activity_regularizer = col11.selectbox('Activity Regularizer', weight_regularizers, key=f'activity_regularizer_{num_layer}')
    reset_after = col12.selectbox('Reset After', (True, False), key=f'reset_after_{num_layer}')

    col13, col14 = st.columns(2)
    dropout = col13.slider('Dropout', 0.0, 1.0, step=0.05, key=f'dropout_{num_layer}')
    recurrent_dropout = col14.slider('Recurrent Dropout', 0.0, 1.0, step=0.05, key=f'recurrent_dropout_{num_layer}')

    col15, col16 = st.columns(2)
    if init and num_layer == 2:
        return_sequences = col15.selectbox('Return Sequences', (True, False), key=f'return_sequences_{num_layer}')
    else:
        return_sequences = col15.selectbox('Return Sequences', (False, True), key=f'return_sequences_{num_layer}')
    return_state = col16.selectbox('Return State', (False, True), key=f'return_state_{num_layer}')

    col17, col18 = st.columns(2)
    go_backwards = col17.selectbox('Go Backwards', (False, True), key=f'go_backwards_{num_layer}')
    stateful = col18.selectbox('Stateful', (False, True), key=f'stateful_{num_layer}')

    return {
        'layer': 'GRU',
        'units': units,
        'activation': activation,
        'recurrent_activation': recurrent_activation,
        'use_bias': bias,
        'kernel_initializer': kernel_initializer,
        'recurrent_initializer': recurrent_initializer,
        'bias_initializer': bias_initializer,
        'kernel_regularizer': kernel_regularizer,
        'recurrent_regularizer': recurrent_regularizer,
        'bias_regularizer': bias_regularizer,
        'activity_regularizer': activity_regularizer,
        'kernel_constraint': None,
        'recurrent_constraint': None,
        'bias_constraint': None,
        'dropout': dropout,
        'recurrent_dropout': recurrent_dropout,
        'return_sequences': return_sequences,
        'return_state': return_state,
        'go_backwards': go_backwards,
        'stateful': stateful,
        'unroll': False,
        'time_major': False,
        'reset_after': reset_after,
    }


layer_dict = {'Dense': Dense, 'Embedding': Embedding, 'SimpleRNN': SimpleRNN, 'LSTM': LSTM, 'GRU': GRU}

def build_model(info_dict):
    model = Sequential()
    for layer in info_dict:
        layer_type = info_dict[layer]['layer']
        layer_class = layer_dict[info_dict[layer]['layer']]
        hyper_params = {k: v for i, (k, v) in enumerate(info_dict[layer].items()) if i != 0}
        model.add(layer_class(**hyper_params))


######################
# Evaluate
######################

def acc_loss_over_time():
    history_dict = st.session_state['history'].history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # r is for "solid red line"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    return fig

def get_metrics(true_labels, predicted_labels):
    accuracy = np.round(metrics.accuracy_score(true_labels, predicted_labels), 4)
    precision = np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 4)
    recall = np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 4)
    f1 = np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 4)
    return accuracy, precision, recall, f1

def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]):
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]
    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, labels=classes)
    return cm

def plot_confusion_matrix(conf_mx):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.matshow(conf_mx, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(conf_mx)))
    ax.set_yticks(np.arange(len(conf_mx)))
    for i in range(len(conf_mx)):
        for j in range(len(conf_mx)):
            color = 'white' if conf_mx[i, j] > np.max(conf_mx) / 2 else 'black'
            ax.text(j, i, str(conf_mx[i, j]), ha='center', va='center', color=color)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.colorbar(im)
    return fig


def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random guess')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    return fig

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


######################
# Inference
######################

def inf_preprocessing(text):
    tf_text = tf.convert_to_tensor([text])
    tf_text = tf.strings.substr(tf_text, 0, 300)
    tf_text = tf.strings.regex_replace(tf_text, b"<br\\s*/?>", b" ")
    tf_text = tf.strings.regex_replace(tf_text, b"[^a-zA-Z']", b" ")
    tf_text = tf.strings.split(tf_text)
    tf_text = tf_text.to_tensor(default_value=b"<pad>")
    tf_text = st.session_state['table'].lookup(tf_text)
    return tf_text