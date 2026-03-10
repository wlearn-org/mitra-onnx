# Changelog

## 0.2.0

- Add `MitraModel` unified class via `createModelClass`
- Unified class accepts `task` parameter and auto-detect from labels
- Original split classes (`MitraClassifier`, `MitraRegressor`) still exported for backward compatibility

## 0.1.0

Initial release.

- Mitra Tab2D in-context learning model as a wlearn estimator
- MitraClassifier and MitraRegressor wrapping ONNX Runtime inference
- In-context learning: `fit()` stores a support set, `predict()` runs a single ONNX forward pass
- Save/load via WLRN bundle format (support set serialized, ONNX model provided separately)
- Registry loaders for `@wlearn/core` global `load()` dispatch
- 26 tests covering create, fit, predict, predictProba, score, save/load, dispose, determinism
- Apache-2.0 license
