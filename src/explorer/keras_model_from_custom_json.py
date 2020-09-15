"""
Build a custom model architecture using custom json reader and output a tensorflow keras model object
"""
import tensorflow as tf
import json


def layers(str_layer):
    """
    Translate string to actual tf.keras.layers object.
    We can define a custom_layer and add it to the dictionary.
    :param str_layer:
    :return:
    """
    dict_layers = {
        'Input': tf.keras.layers.Input,
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


def activations(str_active):
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


def optimizers(str_optimizer):
    """
    Translate string to actual tf.keras.optimizer object.
    We can define a custom_optimizer and add it to the dictionary.
    :param str_optimizer:
    :return:
    """
    dict_optimizers = {
        'Adam': tf.keras.optimizers.Adam,
        'Adadelta': tf.keras.optimizers.Adadelta,
        'Adagrad': tf.keras.optimizers.Adagrad,
        'Nadam': tf.keras.optimizers.Nadam,
        'RMSprop': tf.keras.optimizers.RMSprop,
        'SGD': tf.keras.optimizers.SGD,
        'Adamax': tf.keras.optimizers.Adamax

    }
    return dict_optimizers[str_optimizer]


def losses(str_loss):
    """
    Translate string to actual tf.keras.losses function.
    We can define a custom_loss and add it to the dictionary.
    :param str_loss:
    :return:
    """
    dict_losses = {
        'binary_crossentropy': tf.keras.losses.binary_crossentropy,
        'mae': tf.keras.losses.mae,
        'mse': tf.keras.losses.mse,
        'mape': tf.keras.losses.mape,
        'msle': tf.keras.losses.msle,
        'kld': tf.keras.losses.kld,
        'poisson': tf.keras.losses.poisson,
        'logcosh': tf.keras.losses.logcosh,
        'cosine_similarity': tf.keras.losses.cosine_similarity,
        'squared_hinge': tf.keras.losses.squared_hinge,
        'hinge': tf.keras.losses.hinge,
        'categorical_hinge': tf.keras.losses.categorical_hinge

    }
    return dict_losses[str_loss]


def metrics(str_metric):
    """
    Translate string to actual tf.keras.metrics function.
    We can define a custom_loss and add it to the dictionary.
    :param str_metric:
    :return:
    """
    dict_metrics = {
        'binary_accuracy': tf.keras.metrics.binary_accuracy,
        'categorical_accuracy': tf.keras.metrics.categorical_accuracy,
        'sparse_categorical_accuracy': tf.keras.metrics.sparse_categorical_accuracy,
        'sparse_top_k_categorical_accuracy': tf.keras.metrics.sparse_top_k_categorical_accuracy

    }
    return dict_metrics[str_metric]


def str_to_layer(dict_layer):
    """

    :param dict_layer:
    :return:
    """
    dict_layer['layer'] = layers(dict_layer['layer'])
    dict_layer['activation'] = activations(dict_layer['activation'])
    return dict_layer


def str_to_layers(dict_layers):
    """

    :param dict_layers:
    :return:
    """
    for key, list_layers in dict_layers.items():
        print(f"key:{key}, list_layers: {list_layers}")
        dict_layers[key] = [str_to_layer(dict_layer) for dict_layer in list_layers]
    return dict_layers


def str_to_optimizer(dict_optimizer):
    """

    :param dict_optimizer:
    :return:
    """
    dict_optimizer['opt_type'] = optimizers(dict_optimizer['opt_type'])
    return dict_optimizer


def str_to_loss(dict_loss):
    """

    :param dict_loss:
    :return:
    """
    dict_loss['loss'] = losses(dict_loss['loss'])
    return dict_loss


def str_to_metrics(dict_metrics):
    """

    :param dict_metrics:
    :return:
    """
    dict_metrics['metrics'] = [metrics(str_metric) for str_metric in dict_metrics['metrics']]
    return dict_metrics


def json_to_tf_objects(str_json):
    """

    :param str_json:
    :return:
    """
    dict_model_arch = json.loads(str_json)
    print(f"dict_model_arch: {dict_model_arch}")

    dict_model_arch['layers'] = str_to_layers(dict_model_arch['layers'])
    dict_model_arch['optimizer'] = str_to_optimizer(dict_model_arch['optimizer'])
    dict_model_arch['loss'] = str_to_loss(dict_model_arch['loss'])
    dict_model_arch['metrics'] = str_to_metrics(dict_model_arch['metrics'])

    return dict_model_arch


def build_neural_arch(dict_params):
    """

    :param dict_params:
    :return:
    """
    dict_model_arch = json_to_tf_objects(dict_params['json_model_arch'])
    print(dict_model_arch)
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
