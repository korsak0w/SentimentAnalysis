######################
# Import libraries
######################
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

import functions as fnc
import custom_layers
import callbacks

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
APP_INFO = """
# Sentiment Analysis with Keras
**Build** and **train** your own model for sentiment analysis using Keras! You can **customize the network architecture**, **manage datasets**,  **train** and **evaluate** your model.
***
"""
st.write(APP_INFO)


######################
# Session States
######################
states = [
    # Upload Dataset
    'df',
    'use_default',

    # Preprocessing
    'X_train_int',
    'X_train_txt',
    'y_train',
    'X_val_int',
    'X_val_txt',
    'y_val',
    'X_test_int',
    'X_test_txt',
    'y_test',
    'vocab_size',
    'batch_size',
    'maxlen',
    'tokenizer',
    'prep_completed'
    
    # Build Model
    'model',
    'model_built',
    'use_txt',

    # Compile
    'model_compiled',

    # Train
    'history',

    # Evaluate
    'fig_acc_loss',
    'scores_df',
    'fig_cm',
    'fig_roc',

    # Pretrained
    'raw_train_set',
    'raw_val_set',
    'raw_test_set',
]

if 'key' not in st.session_state:
    st.session_state['key'] = 0   

for state in states:
    st.session_state.setdefault(state, None)

def in_wid_change():        
    for state in states:
        st.session_state[state]= None 

######################
# Data upload
######################
st.header('Data upload')

# Upload file section
file_upload_section = st.empty()
data_source=st.radio('Select data source',['Use example dataset','Upload data'],index=0, key=st.session_state['key'],on_change=in_wid_change)  
uploaded_file=None

if data_source=='Upload data':    
    st.session_state.use_default = False
    st.session_state.df = None
    st.write("The expected input format for the training data is a list of reviews: each row should consist an individual review and the corresponding sentiment labelled as either **positive** or **negative** (i.e., review, sentiment). The separation symbol between the review and the sentiment can be specified in the upload settings. ")
    separator_expander=st.expander('Upload settings')
    with separator_expander:                    
        a1,a2=st.columns(2)
        with a1:            
            col_sep=a1.selectbox("Column sep.",[',',  ';'  , '|', '\s+', '\t','other'], key = st.session_state['key'])
            if col_sep=='other':
                col_sep=st.text_input('Specify your column separator', key = st.session_state['key'])     
        with a2:  
            encoding_val=a2.selectbox("Encoding",[None,'utf8','utf_8','utf_8_sig','utf_16_le','cp1140','cp1250','cp1251','cp1252','cp1253','cp1254','other'], key = st.session_state['key'])
            if encoding_val=='other':
                encoding_val=st.text_input('Specify your encoding', key = st.session_state['key'])  
        
    uploaded_file = st.file_uploader("Default column separator ','", type=["csv", "txt"],on_change=in_wid_change)
    
    if uploaded_file:        
        try:
            st.session_state.df = pd.read_csv(uploaded_file, sep = col_sep,encoding=encoding_val, engine='python')
            DATASET_PREPARATION_WARNING = """
                    When preparing a dataset for binary sentiment analysis, please keep in mind that your dataset must contain a column with text and a column with labels that can be binary encoded (e.g. positive and negative). 
                    Additionally, your text and labels should not contain null-values. 
                    Please ensure that your dataset meets these requirements before using it to train a binary sentiment analysis model.
                """
            st.warning(DATASET_PREPARATION_WARNING)
        except Exception as e:
            FILE_LOADING_ERROR = "Failed to load the file. Please check that it is a valid CSV file and try again."
            st.warning(FILE_LOADING_ERROR)
            st.exception(e)

    
else: # use default data
    # Create default dataframe   
    st.session_state['use_default'] = True    
    #default_df = fnc.create_default_df() 
    default_df = pd.read_csv('data/IMDB_Dataset_mini.csv') 
    st.session_state.df = default_df        
    
    
st.write("""***""")


######################
# Preprocessing
######################

if st.session_state.df is not None:
    st.header('Data screening and processing')
    PREPROCESSING_INFO = """        
    During preprocessing, all **punctuation** is removed, words are converted to **lowercase**, and **split by spaces**. The words are then **indexed by frequency**, with low integers representing frequent words. Additionally, there are three special tokens: **0** represents padding, **1** represents the start-of-sequence (SOS) token, and **2** represents unknown words.
    """
    st.write(PREPROCESSING_INFO)
    df = st.session_state.df
    
    if st.session_state['use_default'] :        
        if st.checkbox("Show data description", value = False, key = st.session_state['key']):          
            DEFAULT_DATASET_INFO = "The **Large Movie Review Dataset** is a comprehensive collection of movie reviews that is commonly used as a **benchmark dataset** for binary sentiment classification tasks. It consists of 50,000 movie reviews, with 25,000 reviews for training and another 25,000 for testing. For more information, please read the paper ***Learning Word Vectors for Sentiment Analysis*** https://aclanthology.org/P11-1015/."
            st.write(DEFAULT_DATASET_INFO)     
    
        if st.checkbox("Show pandas DataFrame info", value = False, key = st.session_state['key']):      
            st.text(fnc.create_pd_info(default_df)) 
    else:
        if st.checkbox("Show pandas DataFrame info", value = False, key = st.session_state['key']):      
            st.text(fnc.create_pd_info(st.session_state.df))        
    if st.checkbox("Show raw data ", value = False, key = st.session_state['key']):      
        st.write(df)

    # Creates two selectboxes for column selection
    col1, col2 = st.columns(2)
    columns = list(df.columns.values)
    column_X = col1.selectbox('Select Input Column', columns, index=0)
    column_y = col2.selectbox('Select Label Column', columns, index=1)

    COLUMNS_WARNING = "Please ensure that you select only one column for the input text and one column for the labels. Using multiple columns for either the input text or the labels may result in errors or unexpected behavior in your analysis or model."
    if column_X==column_y:
        st.error(COLUMNS_WARNING)
    # Create slider for split and batch size
    col3, col4 = st.columns(2)
    batch_options = [16, 32, 64, 128, 256, 512, 1024, 2048]
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
        SUCCESS_PREP = "The dataset was successfully preprocessed!"
        st.success(SUCCESS_PREP)
    
    st.write("""***""")


######################
# Build Model
######################

if st.session_state.X_train_int is not None:
    st.header('Build Your Model')
    BUILD_INFO = """
    You can add or remove layers by clicking on the (**+**) and (**-**) buttons, and each layer is displayed with a dropdown menu of **available layer types** (e.g., Dense, LSTM) and **input parameters** specific to that type.
    
    Once you have selected and configured your desired layers, you can click on the "Build Model" button to generate a **Sequential model** using the **Keras API**.
    """
    st.write(BUILD_INFO)

    # Set session states
    st.session_state.setdefault("model_layers", ["Default_Layer"]*5)
    st.session_state.setdefault("info_dict", {})

    # Arrays and dictionaries
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
        NUM_LAYER_WARNING = "You must add at least one layer to the model before you can build it!"
        st.warning(NUM_LAYER_WARNING)
     
    if st.session_state.model_built:
        SUCCESS_BUILD = "The model was built successfully!"
        st.success(SUCCESS_BUILD)
    
    st.write("""***""")


######################
# Compile Model
######################

if st.session_state.model_built:
    st.header('Compile Your Model')
    COMPILE_INFO = """
    The process of compiling the model involves specifying the **loss function**, **optimizer**, and **evaluation metrics** that will be utilized to train and evaluate the model throughout the training process.
    
    Different combinations of optimizers, loss functions, and metrics can have a significant impact on the performance of the model, so it's important to **choose these hyperparameters carefully** and optimize them for the specific task at hand.
    """
    st.write(COMPILE_INFO)
    col1, col2 = st.columns(2)
    
    # Optimizer and lr
    optimizers = [
        'SGD',
        'RMSprop',
        'Adagrad',
        'Adadelta',
        'Adam',
        'Adamax',
        'Nadam',
        'Ftrl',
    ]
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
    loss_functions = [
        'mean_squared_error',
        'mean_absolute_error',
        'mean_absolute_percentage_error',
        'mean_squared_logarithmic_error',
        'categorical_crossentropy',
        'sparse_categorical_crossentropy',
        'binary_crossentropy',
        'hinge',
        'squared_hinge',
        'cosine_similarity',
        'poisson',
        'kullback_leibler_divergence',
    ]
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
        SUCCESS_COMPILE = "The model was compiled successfully!"    
        st.success(SUCCESS_COMPILE)

    st.write("""***""")


######################
# Train Model
######################

if st.session_state.model_compiled:
    st.header('Train Your Model')
    TRAINING_INFO = """
    After compiling the model, you can **specifying the number of epochs** and then train your model on the labeled training dataset.

    During each epoch, the model is fed the entire training dataset in small batches, and the weights and biases of the model are adjusted to **minimize the loss function**. Typically, **multiple epochs are needed** to achieve good performance on the training dataset, but too many epochs can lead to overfitting, where the model performs well on the training data but poorly on new, unseen data.
    """

    st.write(TRAINING_INFO)

    # Choose the correct train and val set
    X_train = np.array(st.session_state.X_train_txt) \
        if st.session_state.use_txt else st.session_state.X_train_int
    X_val = np.array(st.session_state.X_val_txt) \
        if st.session_state.use_txt else st.session_state.X_val_int

    num_epochs = st.number_input('Epochs', min_value=1, step=1)
    
    num_steps = round(len(X_train) / st.session_state.batch_size)

    # Select and configure callbacks
    callback_options = [
        'EarlyStopping',
        'ReduceLROnPlateau',
        ]
    cb_select = st.multiselect('Select Callbacks', callback_options)
    cb_options = {callback: fnc.create_cb_options(callback) for callback in cb_select}
    cb_dict = {
        'EarlyStopping': tf.keras.callbacks.EarlyStopping,
        'ReduceLROnPlateau': tf.keras.callbacks.ReduceLROnPlateau,
    }
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
        SUCCESS_TRAIN = "The model was trained successfully!"
        st.success(SUCCESS_TRAIN)

    st.write("""***""")


######################
# Evaluate Model
######################

if st.session_state.history:
    st.header('Evaluate Your Model')
    EVALUATE_INFO = """
    When working with binary classifiers in Keras, it is important to **evaluate the performance** of the model using appropriate metrics. Some of the commonly used metrics for evaluating binary classifiers include **accuracy and loss over time**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.
    Evaluating the model using these metrics helps to identify areas where the model can be improved and to assess the overall performance of the classifier.
    """
    st.write(EVALUATE_INFO)
    
    #if st.button('Evaluate Model'):
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
    
    st.write("""***""")


######################
# Inference
######################

if st.session_state.history:
    st.header('Inference')
    INFERENCE_INFO = """
    Machine learning inference is the stage in the development process where the knowledge acquired by the neural network during training is applied. The trained model is utilized to **make predictions** or inferences on **new** and **previously unseen data**. 
    """
    st.write(INFERENCE_INFO)

    MAX_INPUT_LENGTH = 300
    tokenizer = st.session_state.tokenizer
    maxlen = st.session_state.maxlen
    INFERENCE_TEXT = "Write something and let your model predict the sentiment."
    text = st.text_area(INFERENCE_TEXT)
    
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
        TEXT_LEN_WARNING = "The input text exceeds the maximum length of 300 characters allowed."
        st.warning(TEXT_LEN_WARNING)

