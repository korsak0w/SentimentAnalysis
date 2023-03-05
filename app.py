######################
# Import libraries
######################
import streamlit as st
import pandas as pd
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


######################
# Page Title
######################

st.write(messages.APP_INFO)

######################
# Session States
######################

states = [
    # Upload Dataset
    'df',
    'use_default',

    # Preprocessing
    'train_set',
    'val_set',
    'test_set',
    'vocab_size',
    'num_oov_buckets',
    'table',
    
    # Build Model
    'model',
    'model_built',


    'test_labels',
    'batch_size',
    
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
    st.session_state.setdefault(state, None)



######################
# Upload Dataset
######################
st.header('Upload Your Dataset')

# Upload file section
file_upload_section = st.empty()
uploaded_file = file_upload_section.file_uploader("Select a file from your hard drive")

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

if st.session_state['df'] is not None:
    st.header('Preprocess Your Dataset')
    st.write(messages.PREPROCESSING_INFO)
    
    # Creates two selectboxes for column selection
    col1, col2 = st.columns(2)
    columns = list(st.session_state['df'].columns.values)
    column_X = col1.selectbox('Select Input Column', columns, index=0)
    column_y = col2.selectbox('Select Label Column', columns, index=1)
    
    # Create slider for split and batch size
    col3, col4 = st.columns(2)
    test_train_split = col3.slider('Test Train Split', 0.1, 0.9, step=0.1, value=(0.5))
    batch_options = [16, 32, 64, 128, 256, 512, 1024, 2048]
    batch_size = col4.select_slider('Batch Size', options=batch_options, value=32)

    # Create inputs for the size of the vocab and buckets
    col5, col6 = st.columns(2)
    vocab_size = col5.number_input('Vocabulary Size', step=1, value=10000)
    num_oov_buckets = col6.number_input('Number of OOV Buckets', step=1, value=1000)
    
    # Preprocess the datasets
    if st.button('Start Preprocessing'):
        df = st.session_state['df']
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
        
        # Update sessions
        data_dict = {
        'train_set': train_set,
        'val_set': val_set,
        'test_set': test_set,
        'vocab_size': vocab_size,
        'num_oov_buckets': num_oov_buckets,
        'table': table
        }
        st.session_state.update(data_dict)

    if st.session_state['train_set']:
        st.success(messages.SUCCESS_PREP)
    
    st.write(messages.BREAK)


######################
# Build Model
######################

# ! Add Bidirectional layer (for LSTM or GRU)
# ! Add 1D Convolutional layers

if st.session_state['train_set']:
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
        'GRU': GRU
        }
    default_index_dict = {
        1: 1,
        2: 4,
        3: 4,
        4: 0,
        }

    # Create buttons to add and remove layers
    col1, col2 = st.columns([.05,1])
    MAX_LAYERS = 20

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
        st.session_state.info_dict[layer_number] = infos
    
    # Build sequential model
    if len(st.session_state.info_dict):
        if st.button('Build Model'):
            st.session_state.model = Sequential()
            info_dict = st.session_state.info_dict
            for layer in info_dict:
                layer_type = info_dict[layer]['layer']
                layer_class = layer_dict[layer_type]
                hyper_params = {
                    k: v for i, (k, v) in enumerate(info_dict[layer].items()) if i != 0
                    }
                st.session_state.model.add(layer_class(**hyper_params))
            st.session_state.info_dict = info_dict
            st.session_state.model_built = True
    else:
        st.warning(messages.NUM_LAYER_WARNING)
     
    if st.session_state.model_built:
        st.success(messages.SUCCESS_BUILD)
    
    st.write(messages.BREAK)


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

    train_set = st.session_state['train_set']
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
