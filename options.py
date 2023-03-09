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
    'test_labels',
    'batch_size',
    
    # Build Model
    'model',
    'model_built',
    'use_raw_ds',

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
        ]

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

text_embeddings = [
    'https://tfhub.dev/google/nnlm-en-dim50/2',
    'https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2',
    'https://tfhub.dev/google/nnlm-en-dim128/2',
    'https://tfhub.dev/google/nnlm-de-dim50/2',
    'https://tfhub.dev/google/nnlm-de-dim128/2',
]