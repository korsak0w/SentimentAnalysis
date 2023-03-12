######################
# Import libraries
######################
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import options
import messages

from sklearn import metrics
from collections import Counter
from tensorflow import keras


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

def preprocess(texts):
    texts_prepped = []
    for text in texts:
        text = tf.strings.regex_replace(text, b"<br\\s*/?>", b" ")
        text = tf.strings.regex_replace(text, b"[^a-zA-Z']", b" ")
        texts_prepped.append(text.numpy().decode('latin-1'))
    return texts_prepped

def encode_labels(dataset, column_y):
    labels = dataset[column_y].value_counts()
    mapping = {labels.index[1]: 0, labels.index[0]: 1}
    dataset[column_y] = dataset[column_y].map(mapping)
    return dataset

def split_dataset(df, test_train_split):
    train_len = round(len(df) * test_train_split)
    splitted = df[:train_len]
    test_set = df[train_len:]
    val_length = round(len(splitted) * 0.2)
    train_set = splitted[val_length:]
    val_set = splitted[:val_length]
    return train_set, val_set, test_set

def seperate_columns(df, col_X, col_y):
    X = np.asarray(df[col_X].reset_index(drop=True))
    y = np.asarray(df[col_y].reset_index(drop=True))
    return X, y

def transform_texts(tokenizer, maxlen, X_train, X_val, X_test):
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
    X_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=maxlen)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

    return X_train, X_val, X_test


######################
# Build Model
######################

# ! move these to options
activation_functions = [None, "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
weight_initializers = ['random_normal', 'random_uniform', 'truncated_normal', 'zeros', 'ones', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'identity', 'orthogonal']
weight_regularizers = [None, 'l1', 'l2', 'l1_l2']

# ! rewrite functions -> see training block 
def create_infos(layer_type, num_layer, vocab_size, maxlen, init):
    with st.expander('Hyperparameters'):
        if layer_type == 'Dense Layer':
            hyper_params = dense_params(num_layer, init)
            return hyper_params
        elif layer_type == 'Embedding Layer':
            hyper_params = embedding_params(num_layer, vocab_size, init)
            return hyper_params
        elif layer_type == 'Simple Recurrent Neural Network Layer':
            hyper_params = simple_rnn_params(num_layer)
            return hyper_params
        elif layer_type == 'Long Short-Term Memory Layer':
            hyper_params = lstm_params(num_layer)
            return hyper_params
        elif layer_type == 'Gated Recurrent Unit Layer':
            hyper_params = gru_params(num_layer, init)
            return hyper_params
        elif layer_type == 'Pretrained Embedding Layer':
            hyper_params = pretrained_embedding_params(num_layer)
            return hyper_params
        elif layer_type == 'Input Layer':
            return {'layer': 'Input', 'shape': (maxlen,)}
        elif layer_type == 'Dropout Layer':
            hyper_params = dropout_params(num_layer)
            return hyper_params
        elif layer_type == 'Global Average Pooling 1D Layer':
            return {
                'layer': 'GlobalAveragePooling1D',
                'data_format': 'channels_last',
            }
        elif layer_type == 'Token And Position Embedding Layer':
            hyper_params = tp_embedding_params(num_layer, maxlen, vocab_size)
            return hyper_params
        elif layer_type == 'Transformer Block':
            hyper_params = transformer_params(num_layer)
            return hyper_params
        else:
            pass

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
        'input_dim': int(input_dim),
        'output_dim': int(output_dim),
        'embeddings_initializer': "uniform",
        'embeddings_regularizer': None,
        'activity_regularizer': None,
        'embeddings_constraint': None,
        'mask_zero': mask_zero,
        'input_length': None,
    }

def simple_rnn_params(num_layer):
    bidirectional = st.checkbox('Add Bidirectional Wrapper', key=f'bidirectional_{num_layer}')
    
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
        'bidirectional': bidirectional,
        'units': int(units),
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
    bidirectional = st.checkbox('Add Bidirectional Wrapper', key=f'bidirectional_{num_layer}')
    
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
        'bidirectional': bidirectional,
        'units': int(units),
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
    bidirectional = st.checkbox('Add Bidirectional Wrapper', key=f'bidirectional_{num_layer}')
    
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
        'bidirectional': bidirectional,
        'units': int(units),
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

def pretrained_embedding_params(num_layer):
    handle = st.selectbox('Select Text Embedding', options.text_embeddings, key=f'url_{num_layer}')
    return {
        'layer': 'KerasLayer',
        'handle': handle,
        'dtype': tf.string,
        'input_shape': [],
        'output_shape': [50],
    }

def dropout_params(num_layer):
    rate = st.slider('Rate', min_value=0.0, max_value=1.0, step=0.1, value=0.2, key=f'rate_{num_layer}')
    return {
        'layer': 'Dropout',
        'rate': rate,
    }

def tp_embedding_params(num_layer, maxlen, vocab_size):
    embed_dim = st.number_input('Embedding Dimensions', step=1, value=32, key=f'embed_dim_{num_layer}')
    st.warning(messages.LAYER_WARNING)
    return {
        'layer': 'TokenAndPositionEmbedding',
        'maxlen': maxlen,
        'vocab_size': vocab_size,
        'embed_dim': int(embed_dim),
    }

def transformer_params(num_layer):
    col1, col2 = st.columns(2)
    embed_dim = col1.number_input('Embedding Dimensions', step=1, value=32, key=f'embed_dim_{num_layer}')
    num_heads = col2.number_input('Number Of Attention Heads', step=1, value=2, key=f'num_heads_{num_layer}')

    col3, col4 = st.columns(2)
    ff_dim = col3.number_input('Hidden Layer Size in Feed Forward Network', step=1, value=32, key=f'ff_dim_{num_layer}')
    rate = col4.slider('Dropout Rate', min_value=0.0, max_value=1.0, step=0.1, value=0.1, key=f'rate_{num_layer}')

    return {
        'layer': 'TransformerBlock',
        'embed_dim': int(embed_dim),
        'num_heads': int(num_heads),
        'ff_dim': int(ff_dim),
        'rate': rate,
    }

# ! Input layer warning
# ! Pooling layer warning
# ! pretrained embedding input shape warning 

######################
# Train Model
######################

def create_cb_options(name):
    callback_map = {
        'EarlyStopping': create_early_stopping,
        'ReduceLROnPlateau': create_reduce_lr_on_plateau
    }
    return callback_map[name]()

def create_early_stopping():
    with st.expander('EarlyStopping Options'):
        col1, col2 = st.columns(2)
        monitor = col1.selectbox('Monitor', ("loss", "val_loss"), index=1, key='es_monitor')
        min_delta = col2.number_input('Min Delta', min_value=0.00)

        col3, col4 = st.columns(2)
        patience = col3.number_input('Patience', min_value=0, key='es_patience')
        mode = col4.selectbox('Mode', ("auto", "min", "max"), key='es_mode')

        col5, col6 = st.columns(2)
        baseline = col5.number_input('Baseline', min_value=0.0)
        restore_best_weights = col6.selectbox('Restore Best Weights', (False, True))
        
    return {
        'monitor': monitor,
        'min_delta': min_delta,
        'patience': patience,
        'verbose': 0,
        'mode': mode,
        'baseline': baseline,
        'restore_best_weights': restore_best_weights,
    }

def create_reduce_lr_on_plateau():
    with st.expander('ReduceLROnPlateau Options'):
        col1, col2 = st.columns(2)
        monitor = col1.selectbox('Monitor', ("loss", "val_loss"), index=1, key='rlr_monitor')
        factor = col2.number_input('Factor', min_value=0.1, max_value=0.9, step=0.1)

        col3, col4 = st.columns(2)
        patience = col3.number_input('Patience', min_value=0, key='rlr_patience')
        mode = col4.selectbox('Mode', ("auto", "min", "max"), key='rlr_mode')

        col5, col6 = st.columns(2)
        cooldown = col5.number_input('Cooldown', min_value=0)
        min_lr = col6.number_input('Min LR', min_value=0.0000, value=0.0000, step=0.0001, format="%f",)
    
    return {
        'monitor': monitor,
        'factor': factor,
        'patience': patience,
        'verbose': 0,
        'mode': mode,
        'min_delta': 0.0001,
        'cooldown': cooldown,
        'min_lr': min_lr,
    }


######################
# Evaluate Model
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
    return metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, labels=classes)

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


######################
# Inference
######################

def inf_preprocessing(tokenizer, maxlen, text):
    text = preprocess([text])
    text = tokenizer.texts_to_sequences(text)
    text = keras.preprocessing.sequence.pad_sequences(text, maxlen=maxlen)
    return text