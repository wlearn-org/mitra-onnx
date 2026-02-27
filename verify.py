"""
Verify numerical equivalence between PyTorch and ONNX Runtime outputs.

Usage:
  python verify.py
  python verify.py --variant classifier
  python verify.py --variant regressor
  python verify.py --atol 1e-4
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model import Tab2D

VARIANTS = {
    'classifier': {
        'repo_id': 'autogluon/mitra-classifier',
        'onnx_path': 'mitra-classifier.onnx',
    },
    'regressor': {
        'repo_id': 'autogluon/mitra-regressor',
        'onnx_path': 'mitra-regressor.onnx',
    },
}


def load_pytorch_model(repo_id):
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
    state_dict = load_file(weights_path, device='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def make_test_inputs(config, seed=42):
    torch.manual_seed(seed)
    task = config['task'].upper()
    dim_output = config['dim_output']

    B, N_S, N_Q, N_F = 1, 30, 8, 4

    x_support = torch.randn(B, N_S, N_F)
    x_query = torch.randn(B, N_Q, N_F)
    padding_features = torch.zeros(B, N_F, dtype=torch.bool)
    padding_obs_support = torch.zeros(B, N_S, dtype=torch.bool)
    padding_obs_query = torch.zeros(B, N_Q, dtype=torch.bool)

    if task == 'CLASSIFICATION':
        y_support = torch.randint(0, dim_output, (B, N_S))
    else:
        y_support = torch.randn(B, N_S)

    return (x_support, y_support, x_query, padding_features,
            padding_obs_support, padding_obs_query)


def verify_variant(variant, atol=1e-5):
    info = VARIANTS[variant]
    onnx_path = info['onnx_path']

    if not os.path.exists(onnx_path):
        print(f'SKIP {variant}: {onnx_path} not found (run convert.py first)')
        return None

    print(f'\n=== Verifying {variant} ===')

    # Load PyTorch model
    model, config = load_pytorch_model(info['repo_id'])

    # Load ONNX model
    session = ort.InferenceSession(onnx_path)

    # Generate test inputs
    inputs = make_test_inputs(config)
    x_support, y_support, x_query, pad_feat, pad_obs_s, pad_obs_q = inputs

    # PyTorch inference
    with torch.no_grad():
        pt_output = model(x_support, y_support, x_query, pad_feat, pad_obs_s, pad_obs_q)
    pt_np = pt_output.numpy()

    # ONNX Runtime inference
    # Note: padding_features and padding_obs_query are unused in the CPU code path
    # (same as upstream -- they only matter for flash attention). The TorchScript
    # tracer correctly removes them from the ONNX graph.
    ort_input_names = {inp.name for inp in session.get_inputs()}
    ort_inputs = {
        'x_support': x_support.numpy(),
        'y_support': y_support.numpy().astype(
            np.int64 if config['task'].upper() == 'CLASSIFICATION' else np.float32
        ),
        'x_query': x_query.numpy(),
    }
    if 'padding_features' in ort_input_names:
        ort_inputs['padding_features'] = pad_feat.numpy()
    if 'padding_obs_support' in ort_input_names:
        ort_inputs['padding_obs_support'] = pad_obs_s.numpy()
    if 'padding_obs_query' in ort_input_names:
        ort_inputs['padding_obs_query'] = pad_obs_q.numpy()
    ort_output = session.run(None, ort_inputs)[0]

    # Compare
    max_diff = np.max(np.abs(pt_np - ort_output))
    mean_diff = np.mean(np.abs(pt_np - ort_output))
    match = np.allclose(pt_np, ort_output, atol=atol)

    print(f'  PyTorch output shape: {pt_np.shape}')
    print(f'  ONNX output shape:    {ort_output.shape}')
    print(f'  Max abs diff:         {max_diff:.2e}')
    print(f'  Mean abs diff:        {mean_diff:.2e}')
    print(f'  Match (atol={atol}):  {"PASS" if match else "FAIL"}')

    return match


def main():
    parser = argparse.ArgumentParser(description='Verify ONNX vs PyTorch equivalence')
    parser.add_argument(
        '--variant', choices=['classifier', 'regressor', 'all'],
        default='all', help='Which variant to verify (default: all)',
    )
    parser.add_argument('--atol', type=float, default=1e-4, help='Absolute tolerance')
    args = parser.parse_args()

    variants = list(VARIANTS.keys()) if args.variant == 'all' else [args.variant]
    results = {}

    for variant in variants:
        results[variant] = verify_variant(variant, args.atol)

    print('\n=== Summary ===')
    all_pass = True
    for variant, passed in results.items():
        if passed is None:
            status = 'SKIP'
        elif passed:
            status = 'PASS'
        else:
            status = 'FAIL'
            all_pass = False
        print(f'  {variant}: {status}')

    if not all_pass:
        exit(1)


if __name__ == '__main__':
    main()
