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