"""
Download Mitra safetensors from HuggingFace and export to ONNX.

Produces:
  - mitra-classifier.onnx
  - mitra-regressor.onnx

Usage:
  python convert.py
  python convert.py --variant classifier
  python convert.py --variant regressor
"""

import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model import Tab2D

VARIANTS = {
    'classifier': {
        'repo_id': 'autogluon/mitra-classifier',
        'output': 'mitra-classifier.onnx',
    },
    'regressor': {
        'repo_id': 'autogluon/mitra-regressor',
        'output': 'mitra-regressor.onnx',
    },
}

# Dummy input shapes for tracing
B = 1
N_SUPPORT = 50
N_QUERY = 10
N_FEATURES = 5


def download_and_load(repo_id: str, device: str = 'cpu'):
    config_path = hf_hub_download(repo_id=repo_id, filename='config.json')
    weights_path = hf_hub_download(repo_id=repo_id, filename='model.safetensors')

    with open(config_path) as f:
        config = json.load(f)

    model = Tab2D(
        dim=config['dim'],
        dim_output=config['dim_output'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        task=config['task'],
    )

    state_dict = load_file(weights_path, device=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def make_dummy_inputs(config, device='cpu'):
    task = config['task'].upper()
    dim_output = config['dim_output']

    x_support = torch.randn(B, N_SUPPORT, N_FEATURES, device=device)
    x_query = torch.randn(B, N_QUERY, N_FEATURES, device=device)
    padding_features = torch.zeros(B, N_FEATURES, device=device, dtype=torch.bool)
    padding_obs_support = torch.zeros(B, N_SUPPORT, device=device, dtype=torch.bool)
    padding_obs_query = torch.zeros(B, N_QUERY, device=device, dtype=torch.bool)

    if task == 'CLASSIFICATION':
        y_support = torch.randint(0, dim_output, (B, N_SUPPORT), device=device)
    else:
        y_support = torch.randn(B, N_SUPPORT, device=device)

    return (x_support, y_support, x_query, padding_features,
            padding_obs_support, padding_obs_query)


def export_onnx(model, config, output_path):
    dummy_inputs = make_dummy_inputs(config)
    input_names = [
        'x_support', 'y_support', 'x_query',
        'padding_features', 'padding_obs_support', 'padding_obs_query',
    ]
    output_names = ['output']

    dynamic_axes = {
        'x_support': {0: 'batch', 1: 'n_support', 2: 'n_features'},
        'y_support': {0: 'batch', 1: 'n_support'},
        'x_query': {0: 'batch', 1: 'n_query', 2: 'n_features'},
        'padding_features': {0: 'batch', 1: 'n_features'},
        'padding_obs_support': {0: 'batch', 1: 'n_support'},
        'padding_obs_query': {0: 'batch', 1: 'n_query'},
        'output': {0: 'batch', 1: 'n_query'},
    }

    # Add n_classes dimension for classification output
    task = config['task'].upper()
    if task == 'CLASSIFICATION' or (task == 'REGRESSION' and config['dim_output'] > 1):
        dynamic_axes['output'][2] = 'n_classes'

    print(f'Exporting {output_path}...')
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    print(f'Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Convert Mitra safetensors to ONNX')
    parser.add_argument(
        '--variant', choices=['classifier', 'regressor', 'all'],
        default='all', help='Which variant to convert (default: all)',
    )
    args = parser.parse_args()

    variants = list(VARIANTS.keys()) if args.variant == 'all' else [args.variant]

    for variant in variants:
        info = VARIANTS[variant]
        print(f'\n=== Converting {variant} ===')
        model, config = download_and_load(info['repo_id'])
        export_onnx(model, config, info['output'])


if __name__ == '__main__':
    main()
