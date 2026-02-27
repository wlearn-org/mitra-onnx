import { makeLCG, shuffle } from '@wlearn/core'

/**
 * Select a support set from the training data.
 * For classification: stratified sampling (equal per class).
 * For regression: random sampling.
 *
 * Returns { xSupport: Float32Array, ySupport: Float32Array|Int32Array, indices: Int32Array }
 */
export function selectSupport(X, y, nFeatures, maxSupport, seed, stratified) {
  const rows = y.length
  const rng = makeLCG(seed)

  if (rows <= maxSupport) {
    // Use all data
    const xSupport = new Float32Array(rows * nFeatures)
    for (let i = 0; i < rows * nFeatures; i++) xSupport[i] = X[i]
    const ySupport = stratified
      ? new Int32Array(y)
      : new Float32Array(y)
    return { xSupport, ySupport, nSupport: rows }
  }

  let selectedIndices

  if (stratified) {
    // Group indices by class
    const classMap = new Map()
    for (let i = 0; i < rows; i++) {
      const label = y[i]
      if (!classMap.has(label)) classMap.set(label, [])
      classMap.get(label).push(i)
    }

    const classes = [...classMap.keys()].sort((a, b) => a - b)
    const nClasses = classes.length
    const perClass = Math.floor(maxSupport / nClasses)
    const remainder = maxSupport - perClass * nClasses

    selectedIndices = []
    for (let c = 0; c < nClasses; c++) {
      const classIndices = classMap.get(classes[c])
      shuffle(classIndices, rng)
      const take = perClass + (c < remainder ? 1 : 0)
      for (let j = 0; j < Math.min(take, classIndices.length); j++) {
        selectedIndices.push(classIndices[j])
      }
    }
  } else {
    // Random sample
    const indices = Array.from({ length: rows }, (_, i) => i)
    shuffle(indices, rng)
    selectedIndices = indices.slice(0, maxSupport)
  }

  selectedIndices.sort((a, b) => a - b)
  const nSupport = selectedIndices.length

  const xSupport = new Float32Array(nSupport * nFeatures)
  for (let i = 0; i < nSupport; i++) {
    const srcOffset = selectedIndices[i] * nFeatures
    for (let j = 0; j < nFeatures; j++) {
      xSupport[i * nFeatures + j] = X[srcOffset + j]
    }
  }

  const ySupport = stratified
    ? new Int32Array(nSupport)
    : new Float32Array(nSupport)
  for (let i = 0; i < nSupport; i++) {
    ySupport[i] = y[selectedIndices[i]]
  }

  return { xSupport, ySupport, nSupport }
}

/**
 * Run Mitra ONNX inference.
 *
 * @param {object} ort - The onnxruntime module (onnxruntime-node or onnxruntime-web)
 * @param {object} session - An InferenceSession
 * @param {Float32Array} xSupport - (nSupport, nFeatures) row-major
 * @param {Int32Array|Float32Array} ySupport - (nSupport,)
 * @param {Float32Array} xQuery - (nQuery, nFeatures) row-major
 * @param {number} nFeatures
 * @param {number} nSupport
 * @param {number} nQuery
 * @param {boolean} isClassifier
 * @returns {Promise<Float32Array>} raw output from ONNX model
 */
export async function runInference(ort, session, xSupport, ySupport, xQuery, nFeatures, nSupport, nQuery, isClassifier) {
  const B = 1

  // Build tensors -- Mitra expects batch dim
  const xSupportTensor = new ort.Tensor('float32', xSupport, [B, nSupport, nFeatures])

  const yType = isClassifier ? 'int64' : 'float32'
  let ySupportData
  if (isClassifier) {
    ySupportData = new BigInt64Array(nSupport)
    for (let i = 0; i < nSupport; i++) ySupportData[i] = BigInt(ySupport[i])
  } else {
    ySupportData = new Float32Array(ySupport)
  }
  const ySupportTensor = new ort.Tensor(yType, ySupportData, [B, nSupport])

  const xQueryTensor = new ort.Tensor('float32', xQuery, [B, nQuery, nFeatures])

  // padding_obs_support: all zeros (no padding)
  const paddingObsSupport = new Uint8Array(nSupport)
  const paddingTensor = new ort.Tensor('bool', paddingObsSupport, [B, nSupport])

  const feeds = {
    x_support: xSupportTensor,
    y_support: ySupportTensor,
    x_query: xQueryTensor,
    padding_obs_support: paddingTensor
  }

  const results = await session.run(feeds)
  return results.output
}

/**
 * Softmax over logits for a batch of queries.
 * logits: Float32Array of shape (nQuery, nClasses), row-major
 * Returns Float64Array of same shape with probabilities.
 */
export function softmax(logits, nQuery, nClasses) {
  const proba = new Float64Array(nQuery * nClasses)
  for (let i = 0; i < nQuery; i++) {
    const offset = i * nClasses
    let max = -Infinity
    for (let j = 0; j < nClasses; j++) {
      if (logits[offset + j] > max) max = logits[offset + j]
    }
    let sum = 0
    for (let j = 0; j < nClasses; j++) {
      proba[offset + j] = Math.exp(logits[offset + j] - max)
      sum += proba[offset + j]
    }
    for (let j = 0; j < nClasses; j++) {
      proba[offset + j] /= sum
    }
  }
  return proba
}

/**
 * Compute SHA-256 of a Uint8Array using the pure-JS implementation from @wlearn/core.
 */
export { sha256Sync } from '@wlearn/core'
