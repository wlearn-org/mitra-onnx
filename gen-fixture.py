"""
Generate a fixed test fixture and compute expected outputs from both
PyTorch (safetensors) and ONNX Runtime, then save as JSON for the browser test.
"""

import json

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model import Tab2D

B, N_S, N_Q, N_F = 1, 20, 5, 3
SEED = 123


def make_inputs():
    torch.manual_seed(SEED)
    x_support = torch.randn(B, N_S, N_F)
    y_support = torch.randint(0, 10, (B, N_S))
    x_query = torch.randn(B, N_Q, N_F)
    padding_obs_support = torch.zeros(B, N_S, dtype=torch.bool)
    padding_features = torch.zeros(B, N_F, dtype=torch.bool)
    padding_obs_query = torch.zeros(B, N_Q, dtype=torch.bool)
    return x_support, y_support, x_query, padding_features, padding_obs_support, padding_obs_query


def main():
    config_path = hf_hub_download(repo_id='autogluon/mitra-classifier', filename='config.json')
    weights_path = hf_hub_download(repo_id='autogluon/mitra-classifier', filename='model.safetensors')

    with open(config_path) as f:
        config = json.load(f)

    model = Tab2D(
        dim=config['dim'], dim_output=config['dim_output'],
        n_layers=config['n_layers'], n_heads=config['n_heads'],
        task=config['task'],
    )
    state_dict = load_file(weights_path, device='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    x_s, y_s, x_q, pad_f, pad_os, pad_oq = make_inputs()

    # PyTorch output
    with torch.no_grad():
        pt_out = model(x_s, y_s, x_q, pad_f, pad_os, pad_oq).numpy()

    # ONNX Runtime output
    sess = ort.InferenceSession('mitra-classifier.onnx')
    ort_out = sess.run(None, {
        'x_support': x_s.numpy(),
        'y_support': y_s.numpy().astype(np.int64),
        'x_query': x_q.numpy(),
        'padding_obs_support': pad_os.numpy(),
    })[0]

    diff = np.max(np.abs(pt_out - ort_out))
    print(f'PyTorch vs ORT max diff: {diff:.2e}')

    # Save fixture
    fixture = {
        'x_support': x_s.numpy().flatten().tolist(),
        'y_support': y_s.numpy().flatten().tolist(),
        'x_query': x_q.numpy().flatten().tolist(),
        'padding_obs_support': pad_os.numpy().flatten().tolist(),
        'shapes': {
            'x_support': [B, N_S, N_F],
            'y_support': [B, N_S],
            'x_query': [B, N_Q, N_F],
            'padding_obs_support': [B, N_S],
        },
        'expected_output': pt_out.flatten().tolist(),
        'expected_shape': list(pt_out.shape),
    }

    with open('fixture.json', 'w') as f:
        json.dump(fixture, f)

    print(f'Output shape: {pt_out.shape}')
    print(f'Output (flat, first 10): {pt_out.flatten()[:10].tolist()}')
    print('Saved fixture.json')


if __name__ == '__main__':
    main()
