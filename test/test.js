import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const ROOT = join(__dirname, '..')

const hasModels =
  existsSync(join(ROOT, 'mitra-classifier.onnx')) &&
  existsSync(join(ROOT, 'mitra-regressor.onnx'))

const SKIP = !hasModels && 'ONNX model files not found (run convert.py or scripts/download-models.sh)'

// --- Lazy imports (only when ONNX models are available) ---
let ort, MitraClassifier, MitraRegressor, decodeBundle, validateBundle
let classifierSession, regressorSession

if (hasModels) {
  ort = await import('onnxruntime-node')
  const src = await import('../src/index.js')
  MitraClassifier = src.MitraClassifier
  MitraRegressor = src.MitraRegressor
  const core = await import('@wlearn/core')
  decodeBundle = core.decodeBundle
  validateBundle = core.validateBundle

  classifierSession = await ort.InferenceSession.create(join(ROOT, 'mitra-classifier.onnx'))
  regressorSession = await ort.InferenceSession.create(join(ROOT, 'mitra-regressor.onnx'))
}

// --- Helpers ---

function makeLCG(seed = 42) {
  let s = seed | 0
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff
    return s / 0x7fffffff
  }
}

function makeClassificationData(rng, nSamples, nFeatures, nClasses = 3) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const label = i % nClasses
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push(label * 2 + (rng() - 0.5) * 0.5)
    }
    X.push(row)
    y.push(label)
  }
  return { X, y }
}

function makeRegressionData(rng, nSamples, nFeatures) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let target = 0
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 4 - 2
      row.push(v)
      target += v * (j + 1)
    }
    X.push(row)
    y.push(target + (rng() - 0.5) * 0.1)
  }
  return { X, y }
}

// ============================================================
// MitraClassifier
// ============================================================

test('classifier: create() returns unfitted model', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  assert.equal(model.isFitted, false)
  assert.equal(model.capabilities.classifier, true)
  assert.equal(model.capabilities.regressor, false)
  assert.equal(model.capabilities.predictProba, true)
  model.dispose()
})

test('classifier: fit() stores support set', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 30, seed: 7 }, { ort })
  const rng = makeLCG(99)
  const { X, y } = makeClassificationData(rng, 50, 3, 4)

  model.fit(X, y)
  assert.equal(model.isFitted, true)
  assert.equal(model.nrFeature, 3)
  assert.equal(model.nrClass, 4)
  assert.equal(model.classes.length, 4)
  assert.equal(model.classes[0], 0)
  assert.equal(model.classes[3], 3)
  model.dispose()
})

test('classifier: fit() rejects double fit', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10 }, { ort })
  const rng = makeLCG(1)
  const { X, y } = makeClassificationData(rng, 20, 3)
  model.fit(X, y)
  assert.throws(() => model.fit(X, y))
  model.dispose()
})

test('classifier: predict() returns valid classes', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 40, 3, 3)
  model.fit(X, y)

  const testRng = makeLCG(20)
  const { X: Xt } = makeClassificationData(testRng, 5, 3, 3)
  const preds = await model.predict(Xt)

  assert.ok(preds instanceof Float64Array)
  assert.equal(preds.length, 5)
  const validClasses = new Set([0, 1, 2])
  for (let i = 0; i < preds.length; i++) {
    assert.ok(validClasses.has(preds[i]), `invalid prediction: ${preds[i]}`)
  }
  model.dispose()
})

test('classifier: predictProba() returns valid probabilities', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 40, 3, 3)
  model.fit(X, y)

  const testRng = makeLCG(20)
  const { X: Xt } = makeClassificationData(testRng, 5, 3, 3)
  const proba = await model.predictProba(Xt)

  assert.ok(proba instanceof Float64Array)
  assert.equal(proba.length, 15)
  for (let i = 0; i < 5; i++) {
    let sum = 0
    for (let j = 0; j < 3; j++) {
      const p = proba[i * 3 + j]
      assert.ok(p >= 0 && p <= 1, `proba out of range: ${p}`)
      sum += p
    }
    assert.ok(Math.abs(sum - 1.0) < 1e-6, `row ${i} sum=${sum}`)
  }
  model.dispose()
})

test('classifier: score() returns accuracy', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 40, 3, 3)
  model.fit(X, y)
  const s = await model.score(X, y)
  assert.equal(typeof s, 'number')
  assert.ok(s >= 0 && s <= 1, `score out of range: ${s}`)
  model.dispose()
})

test('classifier: getParams() / setParams()', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 100 }, { ort })
  assert.equal(model.getParams().maxSupport, 100)
  assert.equal(model.getParams().seed, 42)
  model.setParams({ seed: 7 })
  assert.equal(model.getParams().seed, 7)
  model.dispose()
})

test('classifier: predict before fit throws NotFittedError', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  await assert.rejects(() => model.predict([[1, 2, 3]]), { name: 'NotFittedError' })
  model.dispose()
})

// ============================================================
// Classifier Save/Load
// ============================================================

test('classifier: save() produces valid WLRN bundle', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 20, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 30, 3, 3)
  model.fit(X, y)

  const bundle = model.save()
  assert.ok(bundle instanceof Uint8Array)
  assert.ok(bundle.length > 16)
  const magic = String.fromCharCode(bundle[0], bundle[1], bundle[2], bundle[3])
  assert.equal(magic, 'WLRN')

  const { manifest, toc } = validateBundle(bundle)
  assert.equal(manifest.typeId, 'wlearn.mitra_onnx.classifier@1')
  assert.equal(toc.length, 3)
  const ids = toc.map(e => e.id).sort()
  assert.deepEqual(ids, ['meta', 'support_x', 'support_y'])
  model.dispose()
})

test('classifier: save/load round-trip', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 20, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 30, 3, 3)
  model.fit(X, y)

  const testRng = makeLCG(20)
  const { X: Xt } = makeClassificationData(testRng, 5, 3, 3)
  const predsBefore = await model.predict(Xt)
  const bundle = model.save()
  model.dispose()

  const loaded = await MitraClassifier.load(bundle, classifierSession, { ort })
  assert.equal(loaded.isFitted, true)
  assert.equal(loaded.nrClass, 3)
  assert.equal(loaded.nrFeature, 3)
  const predsAfter = await loaded.predict(Xt)
  assert.equal(predsAfter.length, predsBefore.length)
  for (let i = 0; i < predsBefore.length; i++) {
    assert.equal(predsBefore[i], predsAfter[i])
  }
  loaded.dispose()
})

test('classifier: load without onnxSource throws', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10 }, { ort })
  const rng = makeLCG(1)
  const { X, y } = makeClassificationData(rng, 20, 3)
  model.fit(X, y)
  const bundle = model.save()
  model.dispose()
  await assert.rejects(() => MitraClassifier.load(bundle))
})

// ============================================================
// MitraRegressor
// ============================================================

test('regressor: create() returns unfitted model', { skip: SKIP }, async () => {
  const model = await MitraRegressor.create(regressorSession, {}, { ort })
  assert.equal(model.isFitted, false)
  assert.equal(model.capabilities.classifier, false)
  assert.equal(model.capabilities.regressor, true)
  assert.equal(model.capabilities.predictProba, false)
  model.dispose()
})

test('regressor: fit() and predict()', { skip: SKIP }, async () => {
  const model = await MitraRegressor.create(regressorSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeRegressionData(rng, 40, 3)
  model.fit(X, y)
  assert.equal(model.isFitted, true)
  assert.equal(model.nrFeature, 3)

  const testRng = makeLCG(20)
  const { X: Xt } = makeRegressionData(testRng, 5, 3)
  const preds = await model.predict(Xt)
  assert.ok(preds instanceof Float64Array)
  assert.equal(preds.length, 5)
  for (let i = 0; i < preds.length; i++) {
    assert.ok(isFinite(preds[i]), `non-finite prediction at ${i}`)
  }
  model.dispose()
})

test('regressor: score() returns R2', { skip: SKIP }, async () => {
  const model = await MitraRegressor.create(regressorSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeRegressionData(rng, 40, 3)
  model.fit(X, y)
  const s = await model.score(X, y)
  assert.equal(typeof s, 'number')
  assert.ok(isFinite(s))
  model.dispose()
})

test('regressor: save/load round-trip', { skip: SKIP }, async () => {
  const model = await MitraRegressor.create(regressorSession, { maxSupport: 20, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeRegressionData(rng, 30, 3)
  model.fit(X, y)

  const testRng = makeLCG(20)
  const { X: Xt } = makeRegressionData(testRng, 5, 3)
  const predsBefore = await model.predict(Xt)
  const bundle = model.save()
  const { manifest } = decodeBundle(bundle)
  assert.equal(manifest.typeId, 'wlearn.mitra_onnx.regressor@1')
  model.dispose()

  const loaded = await MitraRegressor.load(bundle, regressorSession, { ort })
  assert.equal(loaded.isFitted, true)
  assert.equal(loaded.nrFeature, 3)
  const predsAfter = await loaded.predict(Xt)
  assert.equal(predsAfter.length, predsBefore.length)
  for (let i = 0; i < predsBefore.length; i++) {
    assert.ok(Math.abs(predsBefore[i] - predsAfter[i]) < 1e-6)
  }
  loaded.dispose()
})

// ============================================================
// Dispose & Error Handling
// ============================================================

test('classifier: dispose() is idempotent', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  model.dispose()
  model.dispose()
})

test('classifier: predict after dispose throws DisposedError', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10 }, { ort })
  const rng = makeLCG(1)
  const { X, y } = makeClassificationData(rng, 20, 3)
  model.fit(X, y)
  model.dispose()
  await assert.rejects(() => model.predict([[1, 2, 3]]), { name: 'DisposedError' })
})

test('classifier: fit after dispose throws DisposedError', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  model.dispose()
  assert.throws(() => model.fit([[1, 2, 3]], [0]), { name: 'DisposedError' })
})

test('classifier: save before fit throws NotFittedError', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  assert.throws(() => model.save(), { name: 'NotFittedError' })
  model.dispose()
})

test('classifier: feature dimension mismatch throws ValidationError', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10 }, { ort })
  model.fit([[1, 2, 3], [4, 5, 6]], [0, 1])
  await assert.rejects(() => model.predict([[1, 2]]), { name: 'ValidationError' })
  model.dispose()
})

// ============================================================
// Support Set Selection
// ============================================================

test('classifier: maxSupport limits support set size', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 100, 3, 3)
  model.fit(X, y)
  const bundle = model.save()
  const { toc } = decodeBundle(bundle)
  const xEntry = toc.find(e => e.id === 'support_x')
  const nSupport = xEntry.length / (3 * 4)
  assert.equal(nSupport, 10)
  model.dispose()
})

test('classifier: small dataset uses all data', { skip: SKIP }, async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 512, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 8, 3, 2)
  model.fit(X, y)
  const bundle = model.save()
  const { toc } = decodeBundle(bundle)
  const xEntry = toc.find(e => e.id === 'support_x')
  const nSupport = xEntry.length / (3 * 4)
  assert.equal(nSupport, 8)
  model.dispose()
})

test('classifier: deterministic with same seed', { skip: SKIP }, async () => {
  async function fitAndPredict() {
    const model = await MitraClassifier.create(classifierSession, { maxSupport: 15, seed: 42 }, { ort })
    const rng = makeLCG(10)
    const { X, y } = makeClassificationData(rng, 30, 3, 3)
    model.fit(X, y)
    const testRng = makeLCG(20)
    const { X: Xt } = makeClassificationData(testRng, 5, 3, 3)
    const preds = await model.predict(Xt)
    model.dispose()
    return preds
  }
  const preds1 = await fitAndPredict()
  const preds2 = await fitAndPredict()
  assert.equal(preds1.length, preds2.length)
  for (let i = 0; i < preds1.length; i++) {
    assert.equal(preds1[i], preds2[i])
  }
})
