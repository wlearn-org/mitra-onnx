import { MitraClassifier } from './classifier.js'
import { MitraRegressor } from './regressor.js'
import { register } from '@wlearn/core'

export { MitraClassifier, MitraRegressor }

/**
 * Register bundle loaders for both Mitra model types.
 * Requires ONNX sources since the ONNX model is not embedded in the .wlrn bundle.
 *
 * Usage:
 *   import { registerLoaders } from '@wlearn/mitra'
 *   import { load } from '@wlearn/core'
 *   registerLoaders(onnxClassifierBytes, onnxRegressorBytes)
 *   const model = await load(bundleBytes)
 */
export function registerLoaders(classifierOnnx, regressorOnnx, opts = {}) {
  if (classifierOnnx) {
    register('wlearn.mitra_onnx.classifier@1', (m, t, b) =>
      MitraClassifier._fromBundle(m, t, b, classifierOnnx, opts)
    )
  }
  if (regressorOnnx) {
    register('wlearn.mitra_onnx.regressor@1', (m, t, b) =>
      MitraRegressor._fromBundle(m, t, b, regressorOnnx, opts)
    )
  }
}
