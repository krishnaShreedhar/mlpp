"""
Build a custom model architecture using custom json reader and output a tensorflow keras model object
"""
import tensorflow as tf


def tf_layers(str_layer):
    """
    Translate string to actual tf.keras.layers object.
    We can define a custom_layer and add it to the dictionary.
    :param str_layer:
    :return:
    """
    dict_layers = {
        'Dense': tf.keras.layers.Dense,
        'Conv1D': tf.keras.layers.Conv1D,
        'Conv2D': tf.keras.layers.Conv2D,
        'Conv3D': tf.keras.layers.Conv3D,
        'RNN': tf.keras.layers.RNN,
        'GRU': tf.keras.layers.GRU,
        'LSTM': tf.keras.layers.LSTM,
        'Embedding': tf.keras.layers.Embedding
    }
    return dict_layers[str_layer]


def tf_activations(str_active):
    """
    Translate string to actual tf.keras.activation function.
    We can define a custom_activation and add it to the dictionary.
    :param str_active:
    :return:
    """
    dict_actives = {
        'relu': tf.keras.activations.relu,
        'elu': tf.keras.activations.elu,
        'selu': tf.keras.activations.selu,
        'tanh': tf.keras.activations.tanh,
        'sigmoid': tf.keras.activations.sigmoid,
        'hard_sigmoid': tf.keras.activations.hard_sigmoid,
        'linear': tf.keras.activations.linear,
        'softmax': tf.keras.activations.softmax,
        'softplus': tf.keras.activations.softplus,

    }
    return dict_actives[str_active]


def tf_optimizers(str_optimizer):
    """
    Translate string to actual tf.keras.optimizer object.
    We can define a custom_optimizer and add it to the dictionary.
    :param str_optimizer:
    :return:
    """
    dict_optimizers = {
        'adam': tf.keras.optimizers.Adam,
        'adadelta': tf.keras.optimizers.Adadelta,
        'adagrad': tf.keras.optimizers.Adagrad,
        'nadam': tf.keras.optimizers.Nadam,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'sgd': tf.keras.optimizers.SGD,
        'adamax': tf.keras.optimizers.Adamax

    }
    return dict_optimizers[str_optimizer]


def tf_losses(str_loss):
    """
    Translate string to actual tf.keras.losses function.
    We can define a custom_loss and add it to the dictionary.
    :param str_loss:
    :return:
    """
    dict_losses = {
        'mae': tf.keras.losses.mae,
        'mse': tf.keras.losses.mse,
        'mape': tf.keras.losses.mape,
        'msle': tf.keras.losses.msle,
        'kld': tf.keras.losses.kld,
        'poisson': tf.keras.losses.poisson,
        'logcosh': tf.keras.losses.logcosh,
        'cos_similarity': tf.keras.losses.cosine_similarity,
        'sq_hinge': tf.keras.losses.squared_hinge,
        'hinge': tf.keras.losses.hinge,
        'cat_hinge': tf.keras.losses.categorical_hinge

    }
    return dict_losses[str_loss]


def json_to_tf_objects():
    pass


def build_neural_arch(dict_params):
    model = None
    return model


def infer_io_shape(data, list_inp_cols, list_out_cols):
    """
    Infer input and output shape for the architecture.
    :param data:
    :param list_inp_cols:
    :param list_out_cols:
    :return:
    """
    input_shape = (None, len(list_inp_cols))
    output_shape = (None, len(list_out_cols))
    return input_shape, output_shape
