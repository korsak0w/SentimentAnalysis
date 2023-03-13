import tensorflow as tf

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

batch_options = [16, 32, 64, 128, 256, 512, 1024, 2048]

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

text_embeddings = [
    'https://tfhub.dev/google/nnlm-en-dim50/2',
    'https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2',
    'https://tfhub.dev/google/nnlm-en-dim128/2',
    'https://tfhub.dev/google/nnlm-de-dim50/2',
    'https://tfhub.dev/google/nnlm-de-dim128/2',
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

callback_options = [
    'EarlyStopping',
    'ReduceLROnPlateau',
]

cb_dict = {
        'EarlyStopping': tf.keras.callbacks.EarlyStopping,
        'ReduceLROnPlateau': tf.keras.callbacks.ReduceLROnPlateau,
    }

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
    "exponential"
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
    'identity',
    'orthogonal'
    ]
weight_regularizers = [
    None,
    'l1',
    'l2',
    'l1_l2'
    ]