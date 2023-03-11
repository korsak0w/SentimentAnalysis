APP_INFO = """
# Sentiment Analysis with Keras
This app allows users to **build** and **train** their own model for sentiment analysis using Keras. Users can **customize the network architecture**, **manage datasets**, and **train** and **evaluate** their model.
***
"""

BREAK = """
***
"""

######################
# Upload Dataset
######################

SELECT_INFO = "Select a file from your hard drive"

DEFAULT_DATASET_INFO = "The **Large Movie Review Dataset** is a comprehensive collection of movie reviews that is commonly used as a **benchmark dataset** for binary sentiment classification tasks. It consists of 50,000 movie reviews, with 25,000 reviews for training and another 25,000 for testing. For more information, please read the paper ***Learning Word Vectors for Sentiment Analysis*** https://aclanthology.org/P11-1015/."

DATASET_PREPARATION_WARNING = """
    When preparing a dataset for binary sentiment analysis, please keep in mind that your dataset must contain a column with text and a column with labels that can be binary encoded (e.g. positive and negative). 
    Additionally, your text and labels should not contain null-values. 
    Please ensure that your dataset meets these requirements before using it to train a binary sentiment analysis model.
"""

FILE_LOADING_ERROR = "Failed to load the file. Please check that it is a valid CSV file and try again."


######################
# Preprocessing
######################

PREPROCESSING_INFO = """
    The expected input format for the training data is a list of reviews, where each review is represented as an **array of integers**. 
    
    During preprocessing, all **punctuation** is removed, words are converted to **lowercase**, and **split by spaces**. The words are then **indexed by frequency**, with low integers representing frequent words. Additionally, there are three special tokens: **0** represents padding, **1** represents the start-of-sequence (SOS) token, and **2** represents unknown words.
    """

COLUMNS_WARNING = "Please ensure that you select only one column for the input text and one column for the labels. Using multiple columns for either the input text or the labels may result in errors or unexpected behavior in your analysis or model."

SUCCESS_PREP = "The dataset was successfully preprocessed!"


######################
# Build Model
######################

BUILD_INFO = """
    You can add or remove layers by clicking on the (**+**) and (**-**) buttons, and each layer is displayed with a dropdown menu of **available layer types** (e.g., Dense, LSTM) and **input parameters** specific to that type.
    
    Once you have selected and configured your desired layers, you can click on the "Build Model" button to generate a **Sequential model** using the **Keras API**.
    """

NUM_LAYER_WARNING = "You must add at least one layer to the model before you can build it!"

SUCCESS_BUILD = "The model was built successfully!"

######################
# Compile Model
######################

COMPILE_INFO = """
    The process of compiling the model involves specifying the **loss function**, **optimizer**, and **evaluation metrics** that will be utilized to train and evaluate the model throughout the training process.
    
    Different combinations of optimizers, loss functions, and metrics can have a significant impact on the performance of the model, so it's important to **choose these hyperparameters carefully** and optimize them for the specific task at hand.
    """

SUCCESS_COMPILE = "The model was compiled successfully!"


######################
# Train Model
######################

TRAINING_INFO = """
    After compiling the model, you can **specifying the number of epochs** and then train your model on the labeled training dataset.

    During each epoch, the model is fed the entire training dataset in small batches, and the weights and biases of the model are adjusted to **minimize the loss function**. Typically, **multiple epochs are needed** to achieve good performance on the training dataset, but too many epochs can lead to overfitting, where the model performs well on the training data but poorly on new, unseen data.
    """

SUCCESS_TRAIN = "The model was trained successfully!"

######################
# Evaluate Model
######################

EVALUATE_INFO = """
    When working with binary classifiers in Keras, it is important to **evaluate the performance** of the model using appropriate metrics. Some of the commonly used metrics for evaluating binary classifiers include **accuracy and loss over time**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.
    Evaluating the model using these metrics helps to identify areas where the model can be improved and to assess the overall performance of the classifier.
    """

SUCCESS_EVALUATE = "The model was evaluated successfully!"

PERFORMANCE_METRICS = "**Model Performance Metrics on the Test Dataset**"


######################
# Inference
######################

INFERENCE_INFO = """
    Machine learning inference is the stage in the development process where the knowledge acquired by the neural network during training is applied. The trained model is utilized to **make predictions** or inferences on **new** and **previously unseen data**. 
    """

INFERENCE_TEXT = "Write something and let your model predict the sentiment."

TEXT_LEN_WARNING = "The input text exceeds the maximum length of 300 characters allowed."
