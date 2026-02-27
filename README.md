# @wlearn/mitra

Mitra Tab2D in-context learning model as a [wlearn](https://github.com/wlearn-org/wlearn)
Estimator. Wraps the 72M-parameter tabular foundation model from Amazon/AutoGluon in the
standard wlearn `fit()` / `predict()` / `save()` / `load()` API.

Mitra uses in-context learning: instead of gradient-based training, `fit()` selects a
support set from your data, and `predict()` passes that support set alongside your query
data through the ONNX model in a single forward pass.

## Data storage warning

**`fit()` stores a subset of your training data** (up to `maxSupport` rows, default 512)
inside the model instance. This support set is:

- Held in memory as Float32Array for the lifetime of the instance
- Serialized into the `.wlrn` bundle when you call `save()`
- Required for every `predict()` call (it is the model's "context")

If your training data is sensitive, be aware that saved bundles contain real data points.
The `maxSupport` parameter controls how many rows are stored. Call `dispose()` to release
the support set from memory.

## Install

```bash
npm install @wlearn/mitra onnxruntime-node   # Node.js
npm install @wlearn/mitra onnxruntime-web    # Browser
```

You also need the ONNX model files (see "ONNX conversion" below).

## Usage

```js
import { MitraClassifier } from '@wlearn/mitra'
import * as ort from 'onnxruntime-node'
import { readFileSync } from 'fs'

// Load ONNX model (290 MB)
const onnxBytes = readFileSync('mitra-classifier.onnx')
const session = await ort.InferenceSession.create(onnxBytes.buffer)

// Create estimator
const model = await MitraClassifier.create(session, {
  maxSupport: 512,  // max support set size (default)
  seed: 42          // RNG seed for support set sampling
}, { ort })

// Fit: selects and stores a support set from training data
model.fit(X_train, y_train)

// Predict: runs ONNX inference with stored support set as context
const predictions = await model.predict(X_test)
const probabilities = await model.predictProba(X_test)
const acc = await model.score(X_test, y_test)

// Save/load (requires the same ONNX model to load)
const bundle = model.save()  // Uint8Array (.wlrn format)
const loaded = await MitraClassifier.load(bundle, session, { ort })

model.dispose()
```

### Regressor

```js
import { MitraRegressor } from '@wlearn/mitra'

const session = await ort.InferenceSession.create(regressorOnnxBytes.buffer)
const model = await MitraRegressor.create(session, { maxSupport: 512 }, { ort })

model.fit(X_train, y_train)
const predictions = await model.predict(X_test)
const r2 = await model.score(X_test, y_test)
```

### Registry integration

```js
import { registerLoaders } from '@wlearn/mitra'
import { load } from '@wlearn/core'

// Register loaders so core.load() can dispatch .wlrn bundles
registerLoaders(classifierSession, regressorSession, { ort })
const model = await load(bundleBytes)
```

## API

### MitraClassifier

| Method | Returns | Description |
|--------|---------|-------------|
| `static create(onnxSource, params?, opts?)` | `Promise<MitraClassifier>` | Factory. onnxSource is an InferenceSession or Uint8Array of ONNX bytes |
| `fit(X, y)` | `this` | Select support set from training data (sync) |
| `predict(X)` | `Promise<Float64Array>` | Class label predictions |
| `predictProba(X)` | `Promise<Float64Array>` | Class probabilities (rows * nClasses) |
| `score(X, y)` | `Promise<number>` | Accuracy |
| `save()` | `Uint8Array` | Serialize to .wlrn bundle |
| `static load(bytes, onnxSource, opts?)` | `Promise<MitraClassifier>` | Deserialize |
| `dispose()` | `void` | Release session and support set |
| `getParams()` | `object` | `{ maxSupport, seed }` |
| `setParams(p)` | `this` | Update params |
| `capabilities` | `object` | `{ classifier: true, predictProba: true, ... }` |
| `classes` | `Int32Array` | Sorted unique class labels |
| `nrClass` | `number` | Number of classes |
| `nrFeature` | `number` | Number of features |

### MitraRegressor

Same API minus `predictProba`, `classes`, `nrClass`. `score()` returns R2.

### Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `maxSupport` | 512 | Maximum support set size. If training data exceeds this, a subset is sampled |
| `seed` | 42 | RNG seed for deterministic support set sampling |

## Bundle format

The `.wlrn` bundle stores the support set, not the ONNX model (which is 290 MB). Loading
requires the ONNX model to be provided separately.

| Artifact | Format | Contents |
|----------|--------|----------|
| `meta` | JSON | `{ nFeatures, nSupport, classes, onnxSha256, seed }` |
| `support_x` | raw float32 | Support set features, row-major |
| `support_y` | raw int32/float32 | Support set labels (int32 for classifier, float32 for regressor) |

TypeIds: `wlearn.mitra_onnx.classifier@1`, `wlearn.mitra_onnx.regressor@1`

## ONNX model variants

| Model | HuggingFace | Output |
|-------|-------------|--------|
| Classifier | [autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier) | `(B, N_query, 10)` logits |
| Regressor | [autogluon/mitra-regressor](https://huggingface.co/autogluon/mitra-regressor) | `(B, N_query)` values |

## ONNX conversion

```bash
pip install -r requirements.txt
```

### Convert

```bash
python convert.py              # both variants
python convert.py --variant classifier
python convert.py --variant regressor
```

Produces `mitra-classifier.onnx` and/or `mitra-regressor.onnx`.

### Verify

Compare PyTorch and ONNX Runtime outputs:

```bash
python verify.py
python verify.py --atol 1e-4
```

### ONNX model inputs

All dimensions are dynamic (variable batch, support/query/feature counts).

| Input | Shape | Type |
|-------|-------|------|
| `x_support` | `(B, N_support, N_features)` | float32 |
| `y_support` | `(B, N_support)` | int64 (classifier) / float32 (regressor) |
| `x_query` | `(B, N_query, N_features)` | float32 |
| `padding_obs_support` | `(B, N_support)` | bool |

Note: `padding_features` and `padding_obs_query` are accepted by the PyTorch model for
API compatibility but are unused in the CPU code path (they only matter for flash
attention). The ONNX tracer correctly eliminates them from the graph.

## Testing

Requires ONNX models in the repo root (run `python convert.py` first).

```bash
npm install
npm test
```

26 tests covering: create, fit, predict, predictProba, score, save/load round-trip,
dispose, error handling, support set selection, determinism.

## What was changed for ONNX export

The upstream AutoGluon implementation
([`_internal/models/tab2d.py`](https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/mitra/_internal/models/tab2d.py),
[`_internal/models/embedding.py`](https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/mitra/_internal/models/embedding.py))
uses several ops that the ONNX exporter cannot trace or has no opset mapping for. Each
replacement below preserves numerical equivalence while using only standard ONNX ops.

### Quantile computation (`Tab2DQuantileEmbeddingX`)

The upstream computes 999 quantiles of `x_support` along the observation axis with
`torch.quantile`, which has no ONNX opset mapping.

Replacement: `torch.sort` along dim 1, then `torch.gather` at fractional index positions
with linear interpolation between the floor and ceil indices. Produces identical quantile
boundaries.

### Bucketize / searchsorted (`Tab2DQuantileEmbeddingX`)

The upstream maps each value to its quantile bin using
`torch.vmap(torch.bucketize, in_dims=(0,0))`. Both `vmap` (not traceable) and
`bucketize` / `searchsorted` (no ONNX op) are unavailable.

Replacement: broadcasting comparison. For values `(b, f, s)` and boundaries
`(b, f, 999)`, compute `(values.unsqueeze(-1) >= boundaries.unsqueeze(-2)).sum(-1)`.
This counts how many boundaries each value exceeds, which is exactly the bucket index.
O(n * 999) instead of O(n * log 999), but 999 is small and the operation is pure
element-wise ONNX ops (GreaterOrEqual, Cast, ReduceSum).

### In-place masked assignment (`Tab2DQuantileEmbeddingX`, `Tab2DEmbeddingY*`)

The upstream uses `x_support[padding_mask] = 9999` and `y_support[padding_obs] = 0` to
set padded positions. In-place mutation through boolean indexing is not traceable.

Replacement: `torch.where(mask, fill_value, x)`. Functionally identical, produces a new
tensor instead of mutating.

### einops / einx (`Tab2D`, `Layer`, `MultiheadAttention`, embeddings)

The upstream uses `einops.rearrange`, `einops.pack`/`unpack`, `einx.rearrange`, and
`einx.sum` throughout. These are external libraries the ONNX tracer cannot see through.

Replacements:
- `einx.rearrange("b s f -> b s f 1", x)` -- `x.unsqueeze(-1)`
- `einops.rearrange("b n -> b n 1", y)` -- `y.unsqueeze(-1)`
- `einops.rearrange("b s f d -> (b f) s d", x)` -- `x.permute(0,2,1,3).reshape(b*f, s, d)`
- `einops.rearrange("(b f) s d -> b s f d", x, b=b)` -- `x.reshape(b, f, s, d).permute(0,2,1,3)`
- `einops.rearrange("b s f d -> (b s) f d", x)` -- `x.reshape(b*s, f, d)`
- `einops.rearrange("b t (h d) -> b h t d", q, h=h)` -- `q.reshape(b, t, h, d).permute(0,2,1,3)`
- `einops.pack((y, x), "b s * d")` -- `torch.cat([y, x], dim=2)`
- `einops.unpack(q, pack_info, "b s * c")` -- `q[:, :, 0, :]` (index the y slot)
- `einx.sum("b [s] f", x)` -- `x.sum(dim=1, keepdim=True)`

### Gradient checkpointing (`Tab2D`)

The upstream wraps each layer call in `torch.utils.checkpoint.checkpoint(layer, ...)`
which is a training-only optimization not compatible with tracing.

Replacement: direct call `layer(support, query)`. The model is exported in eval mode so
checkpointing has no effect anyway.

### Flash attention path (`Tab2D`, `Layer`, `Padder`)

The upstream has two code paths: a flash attention path (CUDA with `flash_attn` library)
and a CPU path using `F.scaled_dot_product_attention`. The flash attention path uses
`flash_attn_varlen_func`, `unpad_input`/`pad_input`, and a `Padder` class -- none of
which are ONNX-exportable.

Replacement: only the CPU path is reimplemented. `F.scaled_dot_product_attention` maps
cleanly to standard ONNX attention ops. The `Padder` class and all flash attention imports
are removed entirely.

### Summary table

| Original | Replacement | Location |
|----------|-------------|----------|
| `torch.quantile` | `sort` + `gather` + lerp | `Tab2DQuantileEmbeddingX` |
| `torch.vmap(torch.bucketize)` | broadcast compare + sum | `Tab2DQuantileEmbeddingX` |
| `x[mask] = val` (in-place) | `torch.where` | embeddings |
| `einops.rearrange` / `einx.rearrange` | `reshape` / `permute` / `unsqueeze` | everywhere |
| `einops.pack` / `unpack` | `torch.cat` / indexing | `Tab2D.forward` |
| `einx.sum` | `torch.sum` | `Tab2DQuantileEmbeddingX` |
| `checkpoint(layer, ...)` | `layer(...)` | `Tab2D.forward` |
| Flash attention path + `Padder` | Removed (CPU path only) | `Tab2D`, `Layer` |

State dict keys match upstream exactly -- safetensors load without any key renaming.

## License

The Mitra model weights are Apache-2.0 licensed by Amazon/AutoGluon.
This conversion code is Apache-2.0 licensed.
