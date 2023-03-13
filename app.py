######################
# Import libraries
######################
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

import functions as fnc
import custom_layers
import messages
import callbacks
import options

from sklearn.metrics import roc_curve, auc
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
# Page Title
######################
st.write(messages.APP_INFO)


######################
# Session States
######################

states = options.states

for state in states:
    st.session_state.setdefault(state, None)


######################
# Upload Dataset
######################
st.header('Upload Your Dataset')

# Upload file section
file_upload_section = st.empty()
uploaded_file = file_upload_section.file_uploader(messages.SELECT_INFO)

# Disable default dataset
if uploaded_file:
    use_default = st.checkbox('Use Default Dataset', disabled=True)
else:
    use_default = st.checkbox('Use Default Dataset', disabled=False)

# Display default dataset information and create default dataframe
if use_default:
    st.session_state['use_default'] = True
    st.write(messages.DEFAULT_DATASET_INFO)
    file_upload_section.empty()
    default_df = fnc.create_default_df()
    st.session_state.df = default_df
    st.text(fnc.create_pd_info(default_df))

# Load uploaded file and display warning if necessary
else:
    st.session_state.use_default = False
    st.session_state.df = None

    if uploaded_file:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.text(fnc.create_pd_info(st.session_state.df))
            st.warning(messages.DATASET_PREPARATION_WARNING)
        except Exception as e:
            st.warning(messages.FILE_LOADING_ERROR)
            st.exception(e)

st.write(messages.BREAK)


######################
# Preprocessing
######################

if st.session_state.df is not None:
    st.header('Preprocess Your Dataset')
    st.write(messages.PREPROCESSING_INFO)
    df = st.session_state.df
    
    # Creates two selectboxes for column selection
    col1, col2 = st.columns(2)
    columns = list(df.columns.values)
    column_X = col1.selectbox('Select Input Column', columns, index=0)
    column_y = col2.selectbox('Select Label Column', columns, index=1)
    
    # Create slider for split and batch size
    col3, col4 = st.columns(2)
    batch_options = options.batch_options
    test_train_split = col3.slider('Test Train Split', 0.1, 0.9, step=0.1, value=(0.5))
    batch_size = col4.select_slider('Batch Size', options=batch_options, value=32)

    # Create inputs for the size of the vocab and buckets
    col5, col6 = st.columns(2)
    maxlen = col5.number_input('Max Sequence Length (Words)', min_value=1, value=100)
    vocab_size = col6.number_input('Vocabulary Size (Words)', step=1, value=10000)
    
    # Preprocess the datasets
    if st.button('Start Preprocessing'):
        # encode labels and split df
        df = fnc.encode_labels(df, column_y)
        train_set, val_set, test_set = fnc.split_dataset(df, test_train_split)
        
        # seperate text and labels
        X_train_txt, y_train = fnc.seperate_columns(train_set, column_X, column_y)
        X_val_txt, y_val = fnc.seperate_columns(val_set, column_X, column_y)
        X_test_txt, y_test = fnc.seperate_columns(test_set, column_X, column_y)

        X_train_txt = fnc.preprocess(X_train_txt)
        X_val_txt = fnc.preprocess(X_val_txt)
        X_test_txt = fnc.preprocess(X_test_txt)

        # Tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(X_train_txt)

        # Transform texts to int and pad to maxlen
        X_train_int, X_val_int, X_test_int = fnc.transform_texts(
            tokenizer, maxlen, X_train_txt, X_val_txt, X_test_txt
            )

        # Update sessions
        data_dict = {
        'X_train_int': X_train_int,
        'X_train_txt': X_train_txt,
        'y_train': y_train,
        'X_val_int': X_val_int,
        'X_val_txt': X_val_txt,
        'y_val': y_val,
        'X_test_int': X_test_int,
        'X_test_txt': X_test_txt,
        'y_test': y_test,
        'vocab_size': vocab_size,
        'batch_size': batch_size,
        'maxlen': maxlen,
        'tokenizer': tokenizer,
        }
        st.session_state.update(data_dict)

    if st.session_state.X_train_int is not None:
        st.session_state.prep_completed = True
        st.success(messages.SUCCESS_PREP)
    
    st.write(messages.BREAK)


######################
# Build Model
######################

if st.session_state.X_train_int is not None:
    st.header('Build Your Model')
    st.write(messages.BUILD_INFO)

    # Set session states
    st.session_state.setdefault("model_layers", ["Default_Layer"]*5)
    st.session_state.setdefault("info_dict", {})

    # Arrays and dictionaries
    layer_options = options.layer_options
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
    default_index_dict = {
        1: 7,
        2: 1,
        3: 4,
        4: 4,
        5: 0,
        }

    col1, col2, col3 = st.columns([.055,.05,1])
    MAX_LAYERS = 20

    # Create buttons to add and remove layers
    if not st.session_state.model_built:
        if col1.button('+') and len(st.session_state.model_layers) <= MAX_LAYERS:
            st.session_state.model_layers.append("Layer")
        if col2.button('-') and len(st.session_state.model_layers) > 0:
            st.session_state.model_layers.pop()
            st.session_state.info_dict.popitem()
        if col3.button('Remove All') and len(st.session_state.model_layers) > 0:
            st.session_state.model_layers = []
            st.session_state.info_dict = {}

    # Add layers and hyperparameters
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
            infos = fnc.create_infos(model_layer, layer_number, vocab_size, maxlen, init=True)
        # normal layer adding
        else:
            model_layer = st.selectbox(
                f'Select Layer {layer_number}',
                layer_options,
                index=0,
                key=f'layer_{layer_number}'
                )
            infos = fnc.create_infos(model_layer, layer_number, vocab_size, maxlen, init=False)
        
        # Flag to use raw dataset instead of preprocessed dataset
        if infos['layer'] == 'KerasLayer':
            st.session_state.use_txt = True

        # Update session
        st.session_state.info_dict[layer_number] = infos

    # Build sequential model
    if len(st.session_state.info_dict) and not st.session_state.model_built:
        if st.button('Build Model'):
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
    elif len(st.session_state.info_dict) and st.session_state.model_built:
        pass
    else:
        st.warning(messages.NUM_LAYER_WARNING)
     
    if st.session_state.model_built:
        st.success(messages.SUCCESS_BUILD)
    
    st.write(messages.BREAK)


######################
# Compile Model
######################

if st.session_state.model_built:
    st.header('Compile Your Model')
    st.write(messages.COMPILE_INFO)
    col1, col2 = st.columns(2)
    
    # Optimizer and lr
    optimizers = options.optimizers
    optimizer_dict = options.optimizer_dict
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

    # Loss function
    loss_functions = options.loss_functions
    loss_function = st.selectbox('Loss Function', loss_functions, index=6)

    # Compile the model
    if st.button('Compile Model'):
        st.session_state.model.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=['accuracy']
            )
        st.session_state.model_compiled = True

    # Print model summary
    if st.session_state.model_compiled:
        with st.expander('Summary'):
            st.session_state.model.summary(
                line_length=79,
                print_fn=lambda x: st.text(x)
                )
        st.success(messages.SUCCESS_COMPILE)

    st.write(messages.BREAK)


######################
# Train Model
######################

if st.session_state.model_compiled:
    st.header('Train Your Model')
    st.write(messages.TRAINING_INFO)

    # Choose the correct train and val set
    X_train = np.array(st.session_state.X_train_txt) \
        if st.session_state.use_txt else st.session_state.X_train_int
    X_val = np.array(st.session_state.X_val_txt) \
        if st.session_state.use_txt else st.session_state.X_val_int

    num_epochs = st.number_input('Epochs', min_value=1, step=1)
    
    num_steps = round(len(X_train) / st.session_state.batch_size)

    # Select and configure callbacks
    cb_select = st.multiselect('Select Callbacks', options.callback_options)
    cb_options = {callback: fnc.create_cb_options(callback) for callback in cb_select}
    cb_dict = options.cb_dict
    my_callbacks = [cb_dict[cb](**cb_options[cb]) for cb in cb_options]

    # Train the model
    if st.button('Train Model'):
        my_callbacks.append(callbacks.PrintCallback(num_epochs))
        my_callbacks.append(callbacks.ProgressCallback(num_steps))
        
        st.session_state.history = st.session_state.model.fit(
            X_train,
            st.session_state.y_train,
            epochs=num_epochs,
            validation_data=(X_val, st.session_state.y_val),
            callbacks=[my_callbacks]
            )

    if st.session_state.history:
        st.success(messages.SUCCESS_TRAIN)

    st.write(messages.BREAK)


######################
# Evaluate Model
######################

if st.session_state.history:
    st.header('Evaluate Your Model')
    st.write(messages.EVALUATE_INFO)
    
    if st.button('Evaluate Model'):
        model = st.session_state.model
        y_test = st.session_state.y_test

        # Choose the correct test set
        if st.session_state.use_txt:
            X_test = np.array(st.session_state.X_test_txt)
        else:
            X_test = st.session_state.X_test_int
        
        # Predict ŷ on test set and evaluate with y
        pred_test = (model.predict(X_test) > 0.5).astype("int32")
        accuracy, precision, recall, f1 = fnc.get_metrics(
            y_test,
            pred_test.flatten()
            )
        scores = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
            }
        
        # Create dataframe
        scores_df = pd.DataFrame(scores, index=[0]).rename(index={0: 'Score'})

        # Create figures
        fig_acc_loss = fnc.acc_loss_over_time()
        cm = fnc.display_confusion_matrix(y_test, pred_test.flatten())
        fig_cm = fnc.plot_confusion_matrix(cm)
        fpr, tpr, thresholds = roc_curve(y_test, pred_test.flatten())
        roc_auc = auc(fpr, tpr)
        fig_roc = fnc.plot_roc_curve(fpr, tpr, roc_auc)

        # Update sessions
        data_dict = {
        'fig_acc_loss': fig_acc_loss,
        'scores_df': scores_df,
        'fig_cm': fig_cm,
        'fig_roc': fig_roc,
        }
        st.session_state.update(data_dict)
    
    # Display dataframe and figures
    if st.session_state.fig_acc_loss:
        st.success(messages.SUCCESS_EVALUATE)
        st.write(messages.PERFORMANCE_METRICS)
        st.write(st.session_state.scores_df)
        with st.expander('Accuracy and Loss over time'):
            st.pyplot(st.session_state.fig_acc_loss)
        with st.expander('Confusion Matrix'):
            st.pyplot(st.session_state.fig_cm)
        with st.expander('Receiver Operating Characteristic (ROC)'):
            st.pyplot(st.session_state.fig_roc)
    
    st.write(messages.BREAK)


######################
# Inference
######################

if st.session_state.history:
    st.header('Inference')
    st.write(messages.INFERENCE_INFO)

    MAX_INPUT_LENGTH = 300
    tokenizer = st.session_state.tokenizer
    maxlen = st.session_state.maxlen
    text = st.text_area(messages.INFERENCE_TEXT)
    
    if st.button('Predict Sentiment') and len(text) <= MAX_INPUT_LENGTH:
        if not st.session_state.use_txt:
            text = fnc.inf_preprocessing(tokenizer, maxlen, text)
        else:
            text = tf.expand_dims(text, 0)
            text = tf.data.Dataset.from_tensor_slices(text).batch(1)
        result = st.session_state.model.predict(text)
        percentage = round(result[0][0] * 100)
        result_str = f"""
        There is a **{percentage}%** chance that your text has a positive sentiment.
        """
        st.info(result_str)
    
    # Warning for large text inputs
    elif len(text) > MAX_INPUT_LENGTH:
        st.warning(messages.TEXT_LEN_WARNING)

