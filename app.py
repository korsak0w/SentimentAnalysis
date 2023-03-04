######################
# Import libraries
######################
import streamlit as st
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import functions as fnc

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU

import callbacks

######################
# Page Title
######################

st.write("""
# Sentiment Analysis with Keras
This app allows users to **build** and **train** their own model for sentiment analysis using Keras. Users can **customize the network architecture**, **manage datasets**, and **train** and **evaluate** their model.
***
""")

######################
# Session States
######################

states = [
    'file',
    'df',
    'use_default',
    'column_X',
    'column_y',
    'test_labels',
    'table',
    'test_train_split',
    'batch_size',
    'vocab_size',
    'num_oov_buckets',
    'progress_prep',
    'encoded_train_set',
    'encoded_val_set',
    'encoded_test_set',
    'model',
    'model_built',
    'model_compiled',
    'history',
    'loss',
    'accuracy',
    'tf_text',
    'fig_acc_loss',
    'scores_df',
    'fig_cm',
    'fig_roc',
]

for state in states:
    if state not in st.session_state:
        st.session_state[state] = None

######################
# Upload Dataset
######################

st.header('Upload Your Dataset')

holder = st.empty()
st.session_state['upload_file'] = holder.file_uploader("Select a file from you hard drive")

if st.session_state['upload_file']:
    use_default = st.checkbox('Use Default Dataset', disabled=True)
else:
    use_default = st.checkbox('Use Default Dataset', disabled=False)

if use_default:
    st.session_state['use_default'] = True
    dataset_info = st.write('The **Large Movie Review Dataset** is a comprehensive collection of movie reviews that is commonly used as a **benchmark dataset** for binary sentiment classification tasks. It consists of 50,000 movie reviews, with 25,000 reviews for training and another 25,000 for testing. For more information, please read the paper ***Learning Word Vectors for Sentiment Analysis*** https://aclanthology.org/P11-1015/.')
    holder.empty()
    st.session_state['df'] = fnc.create_default_df()
    st.text(fnc.create_pd_info(st.session_state['df']))
else:
    st.session_state['use_default'] = False
    st.session_state['df'] = None

if st.session_state['upload_file']:
    st.session_state['df'] = pd.read_csv(st.session_state['upload_file'])
    st.text(fnc.create_pd_info(st.session_state['df']))
    st.warning("""
    When preparing a dataset for binary sentiment analysis, please keep in mind that your dataset must contain a column with text and a column with labels that can be binary encoded (e.g. positive and negative). 
    Additionally, your text and labels should not contain null-values. 
    Please ensure that your dataset meets these requirements before using it to train a binary sentiment analysis model.
    """)
    st.write("""
    ***
    """)


######################
# Preprocessing
######################

if st.session_state['df'] is not None:
    st.header('Preprocess Your Dataset')
    st.write("""
    The expected input format for the training data is a list of reviews, where each review is represented as an **array of integers**. 
    
    During preprocessing, all **punctuation** is removed, words are converted to **lowercase**, and **split by spaces**. The words are then **indexed by frequency**, with low integers representing frequent words. Additionally, there are three special tokens: **0** represents padding, **1** represents the start-of-sequence (SOS) token, and **2** represents unknown words.
    """)
    
    col1, col2 = st.columns(2)
    columns = list(st.session_state['df'].columns.values)
    if st.session_state['use_default']:
        st.session_state['column_X'] = col1.multiselect('Select Input Column', columns, ('review'))
        st.session_state['column_y'] = col2.multiselect('Select Label Column', columns, ('sentiment'))
    else:
        st.session_state['column_X'] = col1.multiselect('Select Input Column', columns)
        st.session_state['column_y'] = col2.multiselect('Select Label Column', columns)
    if len(st.session_state['column_X']) > 1 or len(st.session_state['column_y']) > 1:
        st.warning('Please ensure that you select only one column for the input text and one column for the labels. Using multiple columns for either the input text or the labels may result in errors or unexpected behavior in your analysis or model.')
    
    col3, col4 = st.columns(2)
    st.session_state['test_train_split'] = col3.slider('Test Train Split', 0.1, 0.9, step=0.1, value=(0.5))
    st.session_state['batch_size'] = col4.select_slider('Batch Size', options=[16, 32, 64, 128, 256, 512, 1024, 2048], value=(32))

    col5, col6 = st.columns(2)
    st.session_state['vocab_size'] = col5.number_input('Vocabulary Size', step=1, value=10000)
    st.session_state['num_oov_buckets'] = col6.number_input('Number of OOV Buckets', step=1, value=1000)
    
    if st.button('Start Preprocessing'):
        # sessions
        df = st.session_state['df']
        column_X = st.session_state['column_X'][0]
        column_y = st.session_state['column_y'][0]
        test_train_split = st.session_state['test_train_split']
        batch_size = int(st.session_state['batch_size'])
        vocab_size = int(st.session_state['vocab_size'])
        num_oov_buckets = int(st.session_state['num_oov_buckets'])
        
        prep_bar = st.progress(0)
        df = fnc.encode_labels(df, column_y)
        train, val, test = fnc.split_dataset(df, test_train_split)
        test_labels = test.iloc[:, 1]
        train_set = fnc.create_tensorflow_dataset(train, column_X, column_y)
        val_set = fnc.create_tensorflow_dataset(val, column_X, column_y)
        test_set = fnc.create_tensorflow_dataset(test, column_X, column_y)

        # lookup table
        prep_bar.progress(20)
        vocabulary = fnc.create_vocabulary(train_set, batch_size, vocab_size)
        prep_bar.progress(60)
        st.session_state['table'] = fnc.create_lookup_table(vocabulary, num_oov_buckets)
        prep_bar.progress(80)
        table = st.session_state['table']

        # encode datasets
        encoded_train_set, encoded_val_set, encoded_test_set = fnc.encode_datasets([train_set, val_set, test_set], batch_size, table)
        prep_bar.progress(100)
    
        st.session_state['test_labels'] = test_labels
        st.session_state['encoded_train_set'] = encoded_train_set
        st.session_state['encoded_val_set'] = encoded_val_set
        st.session_state['encoded_test_set'] = encoded_test_set
    #if st.session_state['encoded_train_set']:
        st.write('Train Set: ', encoded_train_set)
        st.write('Validation Set: ', encoded_val_set)
        st.write('Test Set: ', encoded_test_set)
        
    st.write("""
    ***
    """)

######################
# Build Model
######################

# ! Add Bidirectional layer (for LSTM or GRU)
# ! Add 1D Convolutional layers

if st.session_state['encoded_train_set']:
    st.header('Build Your Model')
    st.write("""
    You can add or remove layers by clicking on the (+) and (-) buttons, and each layer is displayed with a dropdown menu of **available layer types** (e.g., Dense, LSTM) and **input parameters** specific to that type.
    
    Once you have selected and configured your desired layers, you can click on the "Build Model" button to generate a **Sequential model** using the **Keras API**.
    """)

    if "model_layers" not in st.session_state:
        st.session_state["model_layers"] = ["Default_Layer", "Default_Layer", "Default_Layer", "Default_Layer"]
    if "info_dict" not in st.session_state:
        st.session_state["info_dict"] = {}

    col1, col2 = st.columns([.05,1])
    with col1:
        if st.button('+'):
            st.session_state.model_layers.append("Layer")
    with col2:
        if st.button('-') and len(st.session_state.model_layers) > 0:
            st.session_state.model_layers.pop()
            st.session_state["info_dict"].popitem()
    
    layer_options = [
        'Dense Layer',
        'Embedding Layer',
        'Simple Recurrent Neural Network Layer',
        'Long Short-Term Memory Layer',
        'Gated Recurrent Unit Layer'
        ]
    input_dim = st.session_state['vocab_size'] + st.session_state['num_oov_buckets']
    
    for i, layer in enumerate(st.session_state.model_layers):
        layer_number = i + 1
        if st.session_state.model_layers[i] == 'Default_Layer':
            # ! default layer adding
            index_dict = {1: 1, 2: 4, 3: 4, 4: 0}
            index = index_dict.get(layer_number, None)
            model_layer = st.selectbox(f'Select Layer {layer_number}', layer_options, index=index, key=f'select_layer_{layer_number}')
            infos = fnc.create_infos(model_layer, layer_number, input_dim, init=True)
            st.session_state["info_dict"][layer_number] = infos
        else:
            # ! normal layer adding
            model_layer = st.selectbox(f'Select Layer {layer_number}', layer_options, key=f'select_layer_{layer_number}')
            infos = fnc.create_infos(model_layer, layer_number, input_dim, init=False)
            st.session_state["info_dict"][layer_number] = infos
    
    if len(st.session_state["info_dict"]) >= 1:
        if st.button('Build Model'):
            layer_dict = {'Dense': Dense, 'Embedding': Embedding, 'SimpleRNN': SimpleRNN, 'LSTM': LSTM, 'GRU': GRU}
            st.session_state["model"] = Sequential()
            for layer in st.session_state["info_dict"]:
                layer_type = st.session_state["info_dict"][layer]['layer']
                layer_class = layer_dict[layer_type]
                hyper_params = {k: v for i, (k, v) in enumerate(st.session_state["info_dict"][layer].items()) if i != 0}
                st.session_state["model"].add(layer_class(**hyper_params))
            st.session_state['model_built'] = True
    else:
        st.warning('You must add at least one layer to the model before you can build it!')
    
    if st.session_state['model_built']:
        st.success('The model was built successfully!')
    st.write("""
    ***
    """)


######################
# Compile Model
######################

if st.session_state['model_built']:
    st.header('Compile Your Model')
    st.write("""
    The process of compiling the model involves specifying the **loss function**, **optimizer**, and **evaluation metrics** that will be utilized to train and evaluate the model throughout the training process.
    
    Different combinations of optimizers, loss functions, and metrics can have a significant impact on the performance of the model, so it's important to **choose these hyperparameters carefully** and optimize them for the specific task at hand.
    """)

    optimizers = [
        'sgd',
        'rmsprop',
        'adagrad',
        'adadelta',
        'adam',
        'adamax',
        'nadam',
        'ftrl'
    ]
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

    col1, col2 = st.columns(2)
    optimizer = col1.selectbox('Optimizer', optimizers, index=4)
    loss_function = col2.selectbox('Loss Function', loss_functions, index=6)

    if st.button('Compile Model'):
        st.session_state["model"].compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        st.session_state["model_compiled"] = True

    if st.session_state["model_compiled"]:
        with st.expander('Summary'):
            st.session_state["model"].summary(line_length=79, print_fn=lambda x: st.text(x))
        st.success('The model was compiled successfully!')

    st.write("""
    ***
    """)


######################
# Train Model
######################

if st.session_state["model_compiled"]:
    st.header('Train Your Model')
    st.write("""
    After compiling the model, you can **specifying the number of epochs** and then train your model on the labeled training dataset.

    During each epoch, the model is fed the entire training dataset in small batches, and the weights and biases of the model are adjusted to **minimize the loss function**. Typically, **multiple epochs are needed** to achieve good performance on the training dataset, but too many epochs can lead to overfitting, where the model performs well on the training data but poorly on new, unseen data.
    """)

    train_set = st.session_state['encoded_train_set']
    num_epochs = st.number_input('Epochs', step=1)

    if st.button('Train Model'):
        st.session_state['history'] = st.session_state["model"].fit(train_set, epochs=num_epochs, validation_data=st.session_state['encoded_val_set'], callbacks=[callbacks.StreamlitCallback(num_epochs)])

    if st.session_state['history']:
        st.success('The model was trained successfully!')

    st.write("""
    ***
    """)


######################
# Evaluate Model
######################

if st.session_state['history']:
    st.header('Evaluate Your Model')
    st.write("""
    When working with binary classifiers in Keras, it is important to **evaluate the performance** of the model using appropriate metrics. Some of the commonly used metrics for evaluating binary classifiers include **accuracy and loss over time**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.
    Evaluating the model using these metrics helps to identify areas where the model can be improved and to assess the overall performance of the classifier.
    """)
    
    if st.button('Evaluate Model'):
        pred_test = (st.session_state["model"].predict(st.session_state['encoded_test_set']) > 0.5).astype("int32")
        accuracy, precision, recall, f1 = fnc.get_metrics(st.session_state['test_labels'], pred_test.flatten())
        st.session_state['fig_acc_loss'] = fnc.acc_loss_over_time()
        scores = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        st.session_state['scores_df'] = pd.DataFrame(scores, index=[0]).rename(index={0: 'Score'})
        cm = fnc.display_confusion_matrix(st.session_state['test_labels'], pred_test.flatten())
        st.session_state['fig_cm'] = fnc.plot_confusion_matrix(cm)
        fpr, tpr, thresholds = roc_curve(st.session_state['test_labels'], pred_test.flatten())
        roc_auc = auc(fpr, tpr)
        st.session_state['fig_roc'] = fnc.plot_roc_curve(fpr, tpr, roc_auc)
        
    if st.session_state['fig_acc_loss']:
        st.success('The model was evaluated successfully!')
        st.write('**Model Performance Metrics on the Test Dataset**')
        st.write(st.session_state['scores_df'])
        with st.expander('Accuracy and Loss over time'):
            st.pyplot(st.session_state['fig_acc_loss'])

        with st.expander('Confusion Matrix'):
            st.pyplot(st.session_state['fig_cm'])

        with st.expander('Receiver Operating Characteristic (ROC)'):
            st.pyplot(st.session_state['fig_roc'])
    
    st.write("""
    ***
    """)


######################
# Inference
######################

if st.session_state['history']:
    st.header('Inference')
    st.write("""
    Machine learning inference is the stage in the development process where the knowledge acquired by the neural network during training is applied. The trained model is utilized to **make predictions** or inferences on **new** and **previously unseen data**. 
    """)

    text = st.text_area('Write something and let your model predict the sentiment.')
    if text:
        st.session_state['tf_text'] = fnc.inf_preprocessing(text)

    if st.button('Predict Sentiment'):
        result = st.session_state['model'].predict(st.session_state['tf_text'])
        percentage = round(result[0][0] * 100)
        result_str = f'There is a {percentage}% chance that your text has a positive sentiment.'
        st.info(result_str)
