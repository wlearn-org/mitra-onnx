const { MitraClassifier } = require('./classifier.js')
const { MitraRegressor } = require('./regressor.js')
const { register, createModelClass } = require('@wlearn/core')

const MitraModel = createModelClass(MitraClassifier, MitraRegressor, { name: 'MitraModel' })

/**
 * Register bundle loaders for both Mitra model types.
 * Requires ONNX sources since the ONNX model is not embedded in the .wlrn bundle.
 *
 * Usage:
 *   const { registerLoaders } = require('@wlearn/mitra')
 *   const { load } = require('@wlearn/core')
 *   registerLoaders(onnxClassifierBytes, onnxRegressorBytes)
 *   const model = await load(bundleBytes)
 */
function registerLoaders(classifierOnnx, regressorOnnx, opts = {}) {
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

module.exports = { MitraModel, MitraClassifier, MitraRegressor, registerLoaders }
