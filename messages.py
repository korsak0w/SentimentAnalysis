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
    You can add or remove layers by clicking on the (+) and (-) buttons, and each layer is displayed with a dropdown menu of **available layer types** (e.g., Dense, LSTM) and **input parameters** specific to that type.
    
    Once you have selected and configured your desired layers, you can click on the "Build Model" button to generate a **Sequential model** using the **Keras API**.
    """

NUM_LAYER_WARNING = "You must add at least one layer to the model before you can build it!"

SUCCESS_BUILD = "The model was built successfully!"