######################
# Import libraries
######################
import streamlit as st
import pandas as pd
import tensorflow as tf
import functions as fnc
import messages
import callbacks
import options

from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
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
    st.session_state['df'] = default_df
    st.text(fnc.create_pd_info(default_df))

# Load uploaded file and display warning if necessary
else:
    st.session_state['use_default'] = False
    st.session_state['df'] = None

    if uploaded_file:
        try:
            st.session_state['df'] = pd.read_csv(uploaded_file)
            st.text(fnc.create_pd_info(st.session_state['df']))
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
    vocab_size = col5.number_input('Vocabulary Size', step=1, value=10000)
    num_oov_buckets = col6.number_input('Number of OOV Buckets', step=1, value=1000)
    
    # Preprocess the datasets
    if not st.session_state.train_set:
        if st.button('Start Preprocessing'):
            df = fnc.encode_labels(df, column_y)
            train_set, val_set, test_set = fnc.split_dataset(df, test_train_split)
            test_labels = test_set.iloc[:, 1]

            # Transform dataframes into tf datasets
            train_set = fnc.create_tensorflow_dataset(train_set, column_X, column_y)
            val_set = fnc.create_tensorflow_dataset(val_set, column_X, column_y)
            test_set = fnc.create_tensorflow_dataset(test_set, column_X, column_y)

            # Create the lookup table
            vocabulary = fnc.create_vocabulary(train_set, batch_size, vocab_size)
            table = fnc.create_lookup_table(vocabulary, num_oov_buckets)

            # Encode the datasets
            datasets = [train_set, val_set, test_set]
            train_set, val_set, test_set = fnc.encode_datasets(datasets, batch_size, table)
            
            # Extract raw sets for pretrained models
            raw_train_set, raw_val_set, raw_test_set = fnc.extract_raw_datasets(
                df,
                test_train_split,
                column_X,
                column_y,
                batch_size
                )

            # Update sessions
            data_dict = {
            'train_set': train_set,
            'val_set': val_set,
            'test_set': test_set,
            'vocab_size': vocab_size,
            'num_oov_buckets': num_oov_buckets,
            'table': table,
            'test_labels': test_labels,
            'batch_size': batch_size,

            'raw_train_set': raw_train_set,
            'raw_val_set': raw_val_set,
            'raw_test_set': raw_test_set,
            }
            st.session_state.update(data_dict)

    if st.session_state.train_set:
        st.success(messages.SUCCESS_PREP)
    
    st.write(messages.BREAK)


######################
# Build Model
######################

if st.session_state.train_set:
    st.header('Build Your Model')
    st.write(messages.BUILD_INFO)

    # Set session states
    st.session_state.setdefault("model_layers", ["Default_Layer"]*4)
    st.session_state.setdefault("info_dict", {})

    # Arrays and dictionaries
    layer_options = options.layer_options
    layer_dict = {
        'Dense': Dense,
        'Embedding': Embedding,
        'SimpleRNN': SimpleRNN,
        'LSTM': LSTM,
        'GRU': GRU,
        'KerasLayer': KerasLayer,
        }
    default_index_dict = {
        1: 1,
        2: 4,
        3: 4,
        4: 0,
        }

    col1, col2 = st.columns([.05,1])
    MAX_LAYERS = 20

    # Create buttons to add and remove layers
    if not st.session_state.model_built:
        with col1:
            if col1.button('+') and len(st.session_state.model_layers) <= MAX_LAYERS:
                st.session_state.model_layers.append("Layer")
        with col2:
            if col2.button('-') and len(st.session_state.model_layers) > 0:
                st.session_state.model_layers.pop()
                st.session_state.info_dict.popitem()

    # Add layers and hyperparameters
    input_dim = st.session_state['vocab_size'] + st.session_state['num_oov_buckets']
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
            infos = fnc.create_infos(model_layer, layer_number, input_dim, init=True)
        # normal layer adding
        else:
            model_layer = st.selectbox(
                f'Select Layer {layer_number}',
                layer_options,
                key=f'layer_{layer_number}'
                )
            infos = fnc.create_infos(model_layer, layer_number, input_dim, init=False)
        
        # Flag to use raw dataset instead of preprocessed dataset
        if infos['layer'] == 'KerasLayer':
            st.session_state.use_raw_ds = True

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
                # Decide if bidirectional wrapper should be added
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
    optimizer_dict = {
        'SGD': tf.keras.optimizers.SGD,
        'RMSprop': tf.keras.optimizers.RMSprop,
        'Adagrad': tf.keras.optimizers.Adagrad,
        'Adadelta': tf.keras.optimizers.Adadelta,
        'Adam': tf.keras.optimizers.Adam,
        'Adamax': tf.keras.optimizers.Adamax,
        'Nadam': tf.keras.optimizers.Nadam,
        'Ftrl': tf.keras.optimizers.Ftrl,
    }
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
    if not st.session_state.model_compiled:
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
    if st.session_state.use_raw_ds:
        train_set = st.session_state.raw_train_set
        val_set = st.session_state.raw_val_set
    else:
        train_set = st.session_state.train_set
        val_set = st.session_state.val_set

    num_epochs = st.number_input('Epochs', step=1)
    num_steps = len(train_set)

    # Train the model
    if not st.session_state.history:
        if st.button('Train Model'):
            st.session_state.history = st.session_state.model.fit(
                train_set,
                epochs=num_epochs,
                validation_data=val_set,
                callbacks=[
                callbacks.PrintCallback(num_epochs),
                callbacks.ProgressCallback(num_steps),
                ])

    if st.session_state.history:
        st.success(messages.SUCCESS_TRAIN)

    st.write(messages.BREAK)


######################
# Evaluate Model
######################

if st.session_state.history:
    st.header('Evaluate Your Model')
    st.write(messages.EVALUATE_INFO)
    
    if not st.session_state.fig_acc_loss:
        if st.button('Evaluate Model'):
            model = st.session_state.model
            test_labels = st.session_state.test_labels

            # Choose the correct test set
            if st.session_state.use_raw_ds:
                test_set = st.session_state.raw_test_set
            else:
                test_set = st.session_state.test_set
            
            # Predict ŷ on test set and evaluate with y
            pred_test = (model.predict(test_set) > 0.5).astype("int32")
            accuracy, precision, recall, f1 = fnc.get_metrics(
                test_labels,
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
            cm = fnc.display_confusion_matrix(test_labels, pred_test.flatten())
            fig_cm = fnc.plot_confusion_matrix(cm)
            fpr, tpr, thresholds = roc_curve(test_labels, pred_test.flatten())
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
    text = st.text_area(messages.INFERENCE_TEXT)

    if st.button('Predict Sentiment') and len(text) <= MAX_INPUT_LENGTH:
        if not st.session_state.use_raw_ds:
            tf_text = fnc.inf_preprocessing(text)
        else:
            text = tf.expand_dims(text, 0)
            tf_text = tf.data.Dataset.from_tensor_slices(text).batch(1)
        result = st.session_state.model.predict(tf_text)
        percentage = round(result[0][0] * 100)
        result_str = f"""
        There is a **{percentage}%** chance that your text has a positive sentiment.
        """
        st.info(result_str)
    
    # Warning for large text inputs
    elif len(text) > MAX_INPUT_LENGTH:
        st.warning(messages.TEXT_LEN_WARNING)

