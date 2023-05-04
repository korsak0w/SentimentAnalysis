######################
# Import libraries
######################
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

import functions as fnc
import callbacks

from sklearn.metrics import roc_curve, auc




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

file_upload_section = st.empty()
data_source=st.radio(
    'Select data source',
    ['Use example dataset','Upload data'],
    index=0, key=st.session_state['key'],
    on_change=in_wid_change
    )  
uploaded_file=None

# upload data
if data_source=='Upload data':
    st.session_state.use_default = False
    st.session_state.df = None
    
    col_sep, encoding_val = fnc.display_upload_settings()
    uploaded_file = st.file_uploader("Default column separator ','", type=["csv", "txt"], on_change=in_wid_change)
    
    if uploaded_file:
        try:
            st.session_state.df = pd.read_csv(uploaded_file, sep=col_sep, encoding=encoding_val, engine='python')
            DATASET_PREPARATION_WARNING = """
                    When preparing a dataset for binary sentiment analysis, please keep in mind that your dataset must contain a column with text and a column with labels that can be binary encoded (e.g. positive and negative). 
                    Additionally, your text and labels should not contain null-values. 
                    Please ensure that your dataset meets these requirements before using it to train a binary sentiment analysis model.
                """
            st.warning(DATASET_PREPARATION_WARNING)
        except Exception as e:
            FILE_LOADING_ERROR = "Failed to load the file. Please check that it is a valid CSV file and try again."
            st.error(FILE_LOADING_ERROR)
            st.exception(e)

# use default data
else:
    st.session_state['use_default'] = True
    default_df = fnc.create_default_df()
    st.session_state.df = default_df
    DEFAULT_DATASET_INFO = "The **Large Movie Review Dataset** is a comprehensive collection of movie reviews that is commonly used as a **benchmark dataset** for binary sentiment classification tasks. It consists of 50,000 movie reviews, with 25,000 reviews for training and another 25,000 for testing. For more information, please read the paper ***Learning Word Vectors for Sentiment Analysis*** https://aclanthology.org/P11-1015/."
    st.write(DEFAULT_DATASET_INFO)


st.write("""***""")


######################
# Preprocessing
######################

if st.session_state.df is not None:
    st.header('Data Screening and Processing')
    PREPROCESSING_INFO = """
    During preprocessing, all **punctuation** is removed, words are converted to **lowercase**, and **split by spaces**. The words are then **indexed by frequency**, with low integers representing frequent words. Additionally, there are three special tokens: **0** represents padding, **1** represents the start-of-sequence (SOS) token, and **2** represents unknown words.
    """
    st.write(PREPROCESSING_INFO)
    
    df = st.session_state.df
    fnc.display_df_checkboxes(df)
    column_X, column_y, pos_label, neg_label, test_train_split, maxlen, vocab_size = fnc.display_preprocess_options(df)

    # Preprocess the datasets
    if st.button('Start Preprocessing'):
        # encode labels and split df
        df = fnc.encode_labels(df, column_y, pos_label, neg_label)
        # preoprocess data
        df = fnc.preprocess(df, column_X)
        # split data
        train_set, val_set, test_set = fnc.split_dataset(df, test_train_split)
        # seperate labes from text
        X_train_txt, y_train = fnc.seperate_columns(train_set, column_X, column_y)
        X_val_txt, y_val = fnc.seperate_columns(val_set, column_X, column_y)
        X_test_txt, y_test = fnc.seperate_columns(test_set, column_X, column_y)

        # Tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
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

    MAX_LAYERS = 20
    fnc.create_buttons(MAX_LAYERS)
    fnc.add_layer()
    fnc.build_model()

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
    
    optimizer = fnc.display_optimizer()
    loss_function = fnc.display_loss_function()
    
    # Compile the model
    if st.button('Compile Model'):
        st.session_state.model.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=['accuracy']
            )
        st.session_state.model_compiled = True

    fnc.print_model_summary()

    st.write("""***""")


######################
# Train Model
######################

if st.session_state.model_compiled:
    st.header('Train Your Model')
    TRAINING_INFO = """
    After compiling the model, you can **specifying the number of epochs** and then train your model on the labeled training dataset.

    During each epoch, the model is fed the entire training dataset, and the weights and biases of the model are adjusted to **minimize the loss function**. Typically, **multiple epochs are needed** to achieve good performance on the training dataset, but too many epochs can lead to overfitting, where the model performs well on the training data but poorly on new, unseen data.
    """
    st.write(TRAINING_INFO)

    X_train, X_val = fnc.get_datasets()

    num_epochs = st.number_input('Epochs', min_value=1, step=1)
    num_steps = round(len(X_train) / 32)

    my_callbacks = fnc.configure_callbacks()

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
    
    if st.button('Evaluate Model'):
        model = st.session_state.model
        y_test = st.session_state.y_test
        X_test = fnc.get_test_set()
        
        # Predict ŷ on test set and evaluate with y
        pred_test = (model.predict(X_test) > 0.5).astype("int32")
        accuracy, precision, recall, f1 = fnc.get_metrics(y_test, pred_test.flatten())
        scores = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        
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
        data_dict = {'fig_acc_loss': fig_acc_loss, 'scores_df': scores_df, 'fig_cm': fig_cm, 'fig_roc': fig_roc}
        st.session_state.update(data_dict)
        
        # Display dataframe and figures
        if st.session_state.fig_acc_loss:
            fnc.display_figures()
        
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

    tokenizer = st.session_state.tokenizer
    maxlen = st.session_state.maxlen
    INFERENCE_TEXT = "Write something and let your model predict the sentiment."
    text = st.text_area(INFERENCE_TEXT)
    
    if st.button('Predict Sentiment'):
        if not st.session_state.use_txt:
            text = fnc.inf_preprocessing(tokenizer, maxlen, text)
        else:
            text = tf.expand_dims(text, 0)
            text = tf.data.Dataset.from_tensor_slices(text).batch(1)
        result = st.session_state.model.predict(text) > 0.5
        
        sentiment_string = "The sentiment of the sentence is positive." if result[0][0] else "The sentiment of the sentence is negative."
        
        st.info(sentiment_string)


