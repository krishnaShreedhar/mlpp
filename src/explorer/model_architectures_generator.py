"""
This will generate different architecture settings depending on user input via json file
"""
import json


def model_arch_json():
    """
    Returns a json object which can be translated to a TensorFlow Keras model
    :return:
    """
    dict_layers = {
        'layer_0': [{
            'layer': 'Dense',
            'n_units': 100,
            'activation': 'relu',
            'dropout': 0.0,
            'batch_norm': True
        }],
        'layer_1': [{
            'layer': 'Dense',
            'n_units': 10,
            'activation': 'relu',
            'dropout': 0.0,
            'batch_norm': True
        }],
        'output': [{
            'layer': 'Dense',
            'n_units': 1,
            'activation': 'softplus',
            'dropout': 0.0,
            'batch_norm': True
        }]
    }

    dict_optimizer = {
        'opt_type': 'Adam',
        'opt_params': {
            'learning_rate': .001,
            'beta_1': .9,
            'beta_2': .999,
            'amsgrad': False
        }
    }

    dict_loss = {
        'loss': 'mse'
    }

    dict_metrics = {
        'metrics': ['binary_accuracy']
    }

    dict_model_arch = {
        'layers': dict_layers,
        'optimizer': dict_optimizer,
        'loss': dict_loss,
        'metrics': dict_metrics
    }

    return json.dumps(dict_model_arch, indent=2)
