######################
# Import libraries
######################
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import re
import custom_layers

from sklearn import metrics
from collections import Counter
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
from tensorflow_hub import KerasLayer


######################
# General Functions
######################

def reset_session_states(stage):
    if stage == "prep":
        st.session_state.model = None
        st.session_state.model_built = False
        st.session_state.model_compiled = False
        st.session_state.history = None
        st.session_state.pred_test = None
    elif stage == "build":
        st.session_state.model_compiled = False
        st.session_state.history = None
        st.session_state.pred_test = None
    elif stage == "compile":
        st.session_state.history = None
        st.session_state.pred_test = None
    elif stage == "train":
        st.session_state.pred_test = None

######################
# Upload Dataset
######################

def display_upload_settings():
    separator_expander = st.expander('Upload settings')
    with separator_expander:
        a1, a2 = st.columns(2)
        with a1:
            col_sep = a1.selectbox("Column Separation", [',', ';', '|', '\\s+', '\\t', 'other'], key='col_sep')
            if col_sep == 'other':
                col_sep = st.text_input('Specify your column separator', key='col_sep_custom')
        with a2:
            encoding_val = a2.selectbox("Encoding", [None, 'utf8', 'utf_8', 'utf_8_sig', 'utf_16_le', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'other'], key='encoding_val')
            if encoding_val == 'other':
                encoding_val = st.text_input('Specify your encoding', key='encoding_val_custom')
    return col_sep, encoding_val

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

def display_df_checkboxes(df):
    TXT_RAW_DATA = "Display Raw Data"
    TXT_DATA_INFO = "Pandas DataFrame Info"

    col1, col2 = st.columns(2)
    if col1.checkbox(TXT_DATA_INFO, value = False, key=f"{st.session_state['key']}_info"):
        st.text(create_pd_info(df))
    if col2.checkbox(TXT_RAW_DATA, value=False, key=f"{st.session_state['key']}_data"):
        st.dataframe(df)

# ! remove ?
#def display_labels(df, column_y):
#    labels = df[column_y].unique()
#    col1, col2 = st.columns(2)
#    pos_label = col1.selectbox('Positive Label', labels, index=0)
#    neg_label = col2.selectbox('Negative Label', labels, index=1)
#    return pos_label, neg_label

def display_preprocess_options(df):
    COLUMNS_WARNING = "Please ensure that you select only one column for the input text and one column for the labels. Using multiple columns for either the input text or the labels may result in errors or unexpected behavior in your analysis or model."
    columns = list(df.columns.values)

    col1, col2 = st.columns(2)
    column_X = col1.selectbox('Input Column', columns, index=0)
    column_y = col2.selectbox('Label Column', columns, index=1)
    if column_X==column_y: st.error(COLUMNS_WARNING)

    labels = df[column_y].unique()
    col3, col4 = st.columns(2)
    pos_label = col3.selectbox('Positive Label', labels, index=0)
    neg_label = col4.selectbox('Negative Label', labels, index=1)
    
    test_train_split = st.slider('Test Train Split', 0.1, 0.9, step=0.1, value=(0.5))

    col5, col6 = st.columns(2)
    maxlen = col5.number_input('Max Sequence Length (Words)', min_value=1, value=100)
    vocab_size = col6.number_input('Vocabulary Size (Words)', step=1, value=10000)
    
    return column_X, column_y, pos_label, neg_label, test_train_split, maxlen, vocab_size

def encode_labels(dataset, column_y, pos_label, neg_label):
    mapping = {neg_label: 0, pos_label: 1}
    dataset[column_y] = dataset[column_y].map(mapping)
    return dataset

def preprocess(data, column_X):
    # regex
    data[column_X] = data[column_X].replace("<br\\s*/?>", " ", regex=True)
    data[column_X] = data[column_X].apply(lambda x: re.sub(r"[^a-zA-ZäöüÄÖÜß']", " ", x))
    # collapse multiple spaces to a single space
    data[column_X] = data[column_X].apply(lambda x: re.sub(r"\s+", " ", x))
    # lowercase
    data[column_X] = data[column_X].apply(lambda x: x.lower())
    # strip
    data[column_X] = data[column_X].apply(lambda x: x.strip())
    # remove null-values
    data.dropna(subset=[column_X], inplace=True)
    # shuffle
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    return data

def split_dataset(df, test_train_split):
    train_len = round(len(df) * test_train_split)
    splitted = df[:train_len]
    test_set = df[train_len:]
    val_length = round(len(splitted) * 0.2)
    train_set = splitted[val_length:]
    val_set = splitted[:val_length]
    return train_set, val_set, test_set

def separate_columns(df, col_X, col_y):
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

activation_functions = [
    None,
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "tanh",
    "selu",
    "elu",
    ]

weight_initializers = [
    'random_normal',
    'random_uniform',
    'truncated_normal',
    'zeros',
    'ones',
    'glorot_normal',
    'glorot_uniform',
    'he_normal',
    'he_uniform',
    'orthogonal'
    ]

weight_regularizers = [
    None,
    'l1',
    'l2',
    'l1_l2'
    ]

# indices for default layers
default_index_dict = {
    1: 7,
    2: 1,
    3: 4,
    4: 4,
    5: 0,
    }

layer_options = [
    'Dense Layer',
    'Embedding Layer',
    'Simple Recurrent Neural Network Layer',
    'Long Short-Term Memory Layer',
    'Gated Recurrent Unit Layer',
    'Pretrained Embedding Layer',
    'Dropout Layer',
    'Input Object',
    'Global Average Pooling 1D Layer',
    'Token And Position Embedding Layer',
    'Transformer Block',
    ]

layer_dict = {
    'Input': Input,
    'Dense': Dense,
    'Embedding': Embedding,
    'SimpleRNN': SimpleRNN,
    'LSTM': LSTM,
    'GRU': GRU,
    'KerasLayer': KerasLayer,
    'Dropout': Dropout,
    'GlobalAveragePooling1D': GlobalAveragePooling1D,
    'TokenAndPositionEmbedding': custom_layers.TokenAndPositionEmbedding,
    'TransformerBlock': custom_layers.TransformerBlock,
    }


def create_buttons(max_layers):
    col1, col2, col3 = st.columns([.055,.05,1])
    if col1.button('+') and len(st.session_state.model_layers) <= max_layers:
        st.session_state.model_layers.append("Layer")
    if col2.button('-') and len(st.session_state.model_layers) > 0:
        st.session_state.model_layers.pop()
        st.session_state.info_dict.popitem()
    if col3.button('Remove All') and len(st.session_state.model_layers) > 0:
        st.session_state.model_layers = []
        st.session_state.info_dict = {}

def add_layer():
    vocab_size = st.session_state.vocab_size
    maxlen = st.session_state.maxlen

    for i, layer in enumerate(st.session_state.model_layers):
        layer_number = i + 1
        # default layer adding
        if st.session_state.model_layers[i] == 'Default_Layer':
            index = default_index_dict.get(layer_number, None)
            model_layer = st.selectbox(
                f'Select Layer {layer_number}',
                layer_options,
                index=index,
                key=f'layer_{layer_number}'
                )
            infos = create_infos(model_layer, layer_number, vocab_size, maxlen, init=True)
        # normal layer adding
        else:
            model_layer = st.selectbox(
                f'Select Layer {layer_number}',
                layer_options,
                index=0,
                key=f'layer_{layer_number}'
                )
            infos = create_infos(model_layer, layer_number, vocab_size, maxlen, init=False)

        # Flag to use raw dataset instead of preprocessed dataset
        if infos['layer'] == 'KerasLayer':
            st.session_state.use_txt = True

        # Update session
        st.session_state.info_dict[layer_number] = infos

def build_model():
    
    # ! delete model_built
    # ! reset session_states: model_compiled, history, pred_test
    if len(st.session_state.info_dict):
        if st.button('Build Model'):
            reset_session_states('build')
            st.session_state.model = Sequential()
            info_dict = st.session_state.info_dict
            for layer in info_dict:
                layer_type = info_dict[layer]['layer']
                layer_class = layer_dict[layer_type]
                hyper_params = {
                    # extracts the hyperparams from the info_dict
                    k: v for i, (k, v) in enumerate(info_dict[layer].items())
                    if i != 0 and k != 'bidirectional'
                    }
                # Decides if bidirectional wrapper should be added
                if info_dict[layer].get('bidirectional', False):
                    st.session_state.model.add(Bidirectional(layer_class(**hyper_params)))
                else:
                    st.session_state.model.add(layer_class(**hyper_params))

            st.session_state.info_dict = info_dict
            st.session_state.model_built = True

    else:
        NUM_LAYER_WARNING = "You must add at least one layer to the model before you can build it!"
        st.warning(NUM_LAYER_WARNING)

def create_infos(layer_type, num_layer, vocab_size, maxlen, init):
    def dense_params(num_layer, vocab_size, maxlen, init):
        with st.expander('Hyperparameters'):
            col1, col2 = st.columns(2)
            if init:
                units = col1.number_input('Number of Units', step=1, value=1, key=f'units_{num_layer}')
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

    def embedding_params(num_layer, vocab_size, maxlen, init):
        with st.expander('Hyperparameters'):
            col1, col2 = st.columns(2)
            if init:
                output_dim = col1.number_input('Embedding Dimensions', step=1, value=128, key=f'output_dim_{num_layer}')
                mask_zero = col2.selectbox('Mask Zero', (False, True), key=f'mask_zero_{num_layer}')
            else:
                output_dim = col1.number_input('Embedding Dimensions', step=1, key=f'output_dim_{num_layer}')
                mask_zero = col2.selectbox('Mask Zero', (False, True), key=f'mask_zero_{num_layer}')

            return {
                'layer': 'Embedding',
                'input_dim': int(vocab_size),
                'output_dim': int(output_dim),
                'embeddings_initializer': "uniform",
                'embeddings_regularizer': None,
                'activity_regularizer': None,
                'embeddings_constraint': None,
                'mask_zero': mask_zero,
                'input_length': None,
            }

    def simple_rnn_params(num_layer, vocab_size, maxlen, init):
        with st.expander('Hyperparameters'):
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
            kernel_regularizer = col7.selectbox('Kernel Regularizer', weight_regularizers, key=f'kernel_regularizer_{num_layer}')
            activity_regularizer = col8.selectbox('Activity Regularizer', weight_regularizers, key=f'activity_regularizer_{num_layer}')

            col9, col10 = st.columns(2)
            recurrent_initializer = col9.selectbox('Recurrent Initializer', weight_initializers, index=9, key=f'recurrent_initializer_{num_layer}')
            recurrent_regularizer = col10.selectbox('Recurrent Regularizer', weight_regularizers, key=f'recurrent_regularizer_{num_layer}')
            
            col11, col12 = st.columns(2)
            dropout = col11.slider('Dropout', 0.0, 1.0, step=0.05, key=f'dropout_{num_layer}')
            recurrent_dropout = col12.slider('Recurrent Dropout', 0.0, 1.0, step=0.05, key=f'recurrent_dropout_{num_layer}')

            col13, col14 = st.columns(2)
            return_sequences = col13.selectbox('Return Sequences', (False, True), key=f'return_sequences_{num_layer}')
            return_state = col14.selectbox('Return State', (False, True), key=f'return_state_{num_layer}')

            go_backwards = st.selectbox('Go Backwards', (False, True), key=f'go_backwards_{num_layer}')

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
                'unroll': False,
            }

    def lstm_params(num_layer, vocab_size, maxlen, init):
        with st.expander('Hyperparameters'):
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
            recurrent_initializer = col8.selectbox('Recurrent Initializer', weight_initializers, index=9, key=f'recurrent_initializer_{num_layer}')

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

            go_backwards = st.selectbox('Go Backwards', (False, True), key=f'go_backwards_{num_layer}')

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
                'time_major': False,
                'unroll': False,
            }

    def gru_params(num_layer, vocab_size, maxlen, init):
        with st.expander('Hyperparameters'):
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
            recurrent_initializer = col8.selectbox('Recurrent Initializer', weight_initializers, index=9, key=f'recurrent_initializer_{num_layer}')

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
            if init and num_layer == 3:
                return_sequences = col15.selectbox('Return Sequences', (True, False), key=f'return_sequences_{num_layer}')
            else:
                return_sequences = col15.selectbox('Return Sequences', (False, True), key=f'return_sequences_{num_layer}')
            return_state = col16.selectbox('Return State', (False, True), key=f'return_state_{num_layer}')

            go_backwards = st.selectbox('Go Backwards', (False, True), key=f'go_backwards_{num_layer}')

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
                'unroll': False,
                'time_major': False,
                'reset_after': reset_after,
            }

    def pretrained_embedding_params(num_layer, vocab_size, maxlen, init):
        text_embeddings = [
            'https://tfhub.dev/google/nnlm-en-dim50/2',
            'https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2',
            'https://tfhub.dev/google/nnlm-en-dim128/2',
            'https://tfhub.dev/google/nnlm-de-dim50/2',
            'https://tfhub.dev/google/nnlm-de-dim128/2',
                ]
        with st.expander('Hyperparameters'):             
            handle = st.selectbox('Select Text Embedding', text_embeddings, key=f'url_{num_layer}')
            PRETRAINED_WARNING = "When using this layer, **avoid using the Input Object** to prevent a shape mismatch as this layer expects a tensor shape of **(None,)**."
            st.warning(PRETRAINED_WARNING)

            return {
                'layer': 'KerasLayer',
                'handle': handle,
                'dtype': tf.string,
                'input_shape': [],
                'output_shape': [50],
            }

    def dropout_params(num_layer, vocab_size, maxlen, init):
        with st.expander('Hyperparameters'):
            rate = st.slider('Rate', min_value=0.0, max_value=1.0, step=0.1, value=0.2, key=f'rate_{num_layer}')
            return {
                'layer': 'Dropout',
                'rate': rate,
            }

    def tp_embedding_params(num_layer, vocab_size, maxlen, init):
        with st.expander('Hyperparameters'):
            embed_dim = st.number_input('Embedding Dimensions', step=1, value=32, key=f'embed_dim_{num_layer}')
            LAYER_WARNING = "Please note that this layer was specifically implemented for use with transformer blocks. It may not be suitable for use in other types of neural networks."
            st.warning(LAYER_WARNING)
            return {
                'layer': 'TokenAndPositionEmbedding',
                'maxlen': maxlen,
                'vocab_size': vocab_size,
                'embed_dim': int(embed_dim),
            }

    def transformer_params(num_layer, vocab_size, maxlen, init):
        with st.expander('Hyperparameters'):
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

    def input_params(num_layer, vocab_size, maxlen, init):
        INPUT_OBJECT_INFO = "Defines an input placeholder with a shape of **(maxlen,)**, where maxlen represents the maximum length of the input sequence that will be fed into the network during training or inference."
        st.info(INPUT_OBJECT_INFO)
        return {'layer': 'Input', 'shape': (maxlen,)}
    
    def global_avg_pooling_1d_params(num_layer, vocab_size, maxlen, init):
        POOLING_INFO = "Computes the average of the feature maps in the time dimension (i.e., along the length of each sequence) of a 1D input tensor, which results in a **single output value per feature map**."
        st.info(POOLING_INFO)
        return {'layer': 'GlobalAveragePooling1D', 'data_format': 'channels_last'}
    
    layer_param_funcs = {
        'Dense Layer': dense_params,
        'Embedding Layer': embedding_params,
        'Simple Recurrent Neural Network Layer': simple_rnn_params,
        'Long Short-Term Memory Layer': lstm_params,
        'Gated Recurrent Unit Layer': gru_params,
        'Pretrained Embedding Layer': pretrained_embedding_params,
        'Input Object': input_params,
        'Dropout Layer': dropout_params,
        'Global Average Pooling 1D Layer': global_avg_pooling_1d_params,
        'Token And Position Embedding Layer': tp_embedding_params,
        'Transformer Block': transformer_params,
    }

    if layer_type in layer_param_funcs:
        return layer_param_funcs[layer_type](num_layer, vocab_size, maxlen, init)

######################
# Compile Model
######################

optimizers = [
    'SGD',
    'RMSprop',
    'Adagrad',
    'Adadelta',
    'Adam',
    'Adamax',
    'Nadam',
    ]

optimizer_dict = {
    'SGD': tf.keras.optimizers.SGD,
    'RMSprop': tf.keras.optimizers.RMSprop,
    'Adagrad': tf.keras.optimizers.Adagrad,
    'Adadelta': tf.keras.optimizers.Adadelta,
    'Adam': tf.keras.optimizers.Adam,
    'Adamax': tf.keras.optimizers.Adamax,
    'Nadam': tf.keras.optimizers.Nadam,
}

loss_functions = [
    'mean_squared_error',
    'mean_absolute_error',
    'mean_squared_logarithmic_error',
    'binary_crossentropy',
    'hinge',
    'squared_hinge',
    ]


def display_optimizer():
    col1, col2 = st.columns(2)
    select_optimizer = col1.selectbox('Optimizer', optimizers, index=4)
    default_lr = 0.01 if select_optimizer == 'SGD' else 0.001
    learning_rate = col2.number_input(
        label='Learning Rate',
        min_value=1e-7,
        step=0.001,
        max_value=1.0,
        value=default_lr,
        format="%f",
        )
    optimizer = optimizer_dict[select_optimizer](lr=learning_rate)
    return optimizer

def display_loss_function():
    loss_function = st.selectbox('Loss Function', loss_functions, index=3)
    return loss_function

def print_model_summary():
    if st.session_state.model_compiled:
        with st.expander('Summary'):
            st.session_state.model.summary(
                line_length=79,
                print_fn=lambda x: st.text(x)
                )
        SUCCESS_COMPILE = "The model was compiled successfully!"    
        st.success(SUCCESS_COMPILE)


######################
# Train Model
######################

callback_options = [
    'EarlyStopping',
    'ReduceLROnPlateau',
    ]

cb_dict = {
    'EarlyStopping': tf.keras.callbacks.EarlyStopping,
    'ReduceLROnPlateau': tf.keras.callbacks.ReduceLROnPlateau,
    }


def get_datasets():
    # Choose the correct train and val set
    X_train = np.array(st.session_state.X_train_txt) \
        if st.session_state.use_txt else st.session_state.X_train_int
    X_val = np.array(st.session_state.X_val_txt) \
        if st.session_state.use_txt else st.session_state.X_val_int
    return X_train, X_val

def create_cb_options(name):
    callback_map = {
        'EarlyStopping': create_early_stopping,
        'ReduceLROnPlateau': create_reduce_lr_on_plateau
    }
    return callback_map[name]()

def configure_callbacks():
    cb_select = st.multiselect('Select Callbacks', callback_options)
    cb_options = {callback: create_cb_options(callback) for callback in cb_select}
    return [cb_dict[cb](**cb_options[cb]) for cb in cb_options]

def create_early_stopping():
    with st.expander('EarlyStopping Options'):
        col1, col2 = st.columns(2)
        monitor = col1.selectbox('Monitor', ("loss", "val_loss"), index=1, key='es_monitor')
        min_delta = col2.number_input('Min Delta', min_value=0.000, format="%.3f")

        col3, col4 = st.columns(2)
        patience = col3.number_input('Patience', min_value=0, key='es_patience')
        mode = col4.selectbox('Mode', ("auto", "min", "max"), key='es_mode')

        restore_best_weights = st.selectbox('Restore Best Weights', (False, True))
    
    return {
        'monitor': monitor,
        'min_delta': min_delta,
        'patience': patience,
        'verbose': 0,
        'mode': mode,
        'baseline': None,
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

def get_test_set():
    if st.session_state.use_txt:
        X_test = np.array(st.session_state.X_test_txt)
    else:
        X_test = st.session_state.X_test_int
    return X_test

def get_predictions(model, X_test):
    predictions = []
    progress_bar = st.progress(0)
    batch_size = 100

    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size]
        batch_preds = model.predict(batch)
        batch_preds = (batch_preds > 0.5).astype("int32")
        predictions.extend(batch_preds)

        # update progress bar
        progress = (i + len(batch)) / len(X_test)
        progress_bar.progress(progress)
    
    # flatten predictions into a single array
    return np.concatenate(predictions)


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

def display_figures():
    SUCCESS_EVALUATE = "The model was evaluated successfully!"
    PERFORMANCE_METRICS = "**Model Performance Metrics on the Test Dataset**"
    st.success(SUCCESS_EVALUATE)
    st.write(PERFORMANCE_METRICS)
    st.write(st.session_state.scores_df)
    with st.expander('Accuracy and Loss over time'):
        st.pyplot(st.session_state.fig_acc_loss)
    with st.expander('Confusion Matrix'):
        st.pyplot(st.session_state.fig_cm)
    with st.expander('Receiver Operating Characteristic (ROC)'):
        st.pyplot(st.session_state.fig_roc)

######################
# Inference
######################

def inf_preprocessing(tokenizer, maxlen, text):
    text_series = pd.Series([text])
    text_series = text_series.replace("<br\\s*/?>", " ", regex=True)
    text_series = text_series.apply(lambda x: re.sub(r"[^a-zA-ZäöüÄÖÜß']", " ", x))
    text_series = text_series.apply(lambda x: re.sub(r"\s+", " ", x))
    text_series = text_series.apply(lambda x: x.lower())
    text_series = text_series.apply(lambda x: x.strip())
    text_series = tokenizer.texts_to_sequences(text_series)
    text_series = keras.preprocessing.sequence.pad_sequences(text_series, maxlen=maxlen)
    return text_series