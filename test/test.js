import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import * as ort from 'onnxruntime-node'
import { decodeBundle, validateBundle } from '@wlearn/core'
import { MitraClassifier } from '../src/classifier.js'
import { MitraRegressor } from '../src/regressor.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const ROOT = join(__dirname, '..')

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    if (err.stack) {
      const lines = err.stack.split('\n').slice(1, 4)
      for (const l of lines) console.log(`        ${l.trim()}`)
    }
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// --- Deterministic LCG PRNG ---
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
// Load ONNX sessions (shared across tests)
// ============================================================

console.log('\n=== Loading ONNX models ===')

let classifierSession, regressorSession

await test('Load classifier ONNX model', async () => {
  const modelPath = join(ROOT, 'mitra-classifier.onnx')
  classifierSession = await ort.InferenceSession.create(modelPath)
  assert(classifierSession, 'session is null')
})

await test('Load regressor ONNX model', async () => {
  const modelPath = join(ROOT, 'mitra-regressor.onnx')
  regressorSession = await ort.InferenceSession.create(modelPath)
  assert(regressorSession, 'session is null')
})

// ============================================================
// MitraClassifier
// ============================================================

console.log('\n=== MitraClassifier ===')

await test('create() returns unfitted classifier', async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  assert(!model.isFitted, 'should not be fitted')
  assert(model.capabilities.classifier === true, 'should be classifier')
  assert(model.capabilities.regressor === false, 'should not be regressor')
  assert(model.capabilities.predictProba === true, 'should support predictProba')
  model.dispose()
})

await test('constructor throws without create()', () => {
  let threw = false
  try { new MitraClassifier() } catch { threw = true }
  assert(threw, 'should throw')
})

await test('fit() stores support set', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 30, seed: 7 }, { ort })
  const rng = makeLCG(99)
  const { X, y } = makeClassificationData(rng, 50, 3, 4)

  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  assert(model.nrFeature === 3, `nrFeature=${model.nrFeature}`)
  assert(model.nrClass === 4, `nrClass=${model.nrClass}`)

  const classes = model.classes
  assert(classes.length === 4, `classes.length=${classes.length}`)
  assert(classes[0] === 0, 'class 0')
  assert(classes[3] === 3, 'class 3')
  model.dispose()
})

await test('fit() rejects double fit', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10 }, { ort })
  const rng = makeLCG(1)
  const { X, y } = makeClassificationData(rng, 20, 3)
  model.fit(X, y)

  let threw = false
  try { model.fit(X, y) } catch { threw = true }
  assert(threw, 'should throw on double fit')
  model.dispose()
})

await test('predict() returns correct shape and valid classes', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 40, 3, 3)

  model.fit(X, y)

  const testRng = makeLCG(20)
  const { X: Xt } = makeClassificationData(testRng, 5, 3, 3)
  const preds = await model.predict(Xt)

  assert(preds instanceof Float64Array, 'should be Float64Array')
  assert(preds.length === 5, `preds.length=${preds.length}`)

  // All predictions should be valid class labels
  const validClasses = new Set([0, 1, 2])
  for (let i = 0; i < preds.length; i++) {
    assert(validClasses.has(preds[i]), `invalid prediction: ${preds[i]}`)
  }
  model.dispose()
})

await test('predictProba() returns valid probabilities', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 40, 3, 3)

  model.fit(X, y)

  const testRng = makeLCG(20)
  const { X: Xt } = makeClassificationData(testRng, 5, 3, 3)
  const proba = await model.predictProba(Xt)

  const nClasses = 3
  assert(proba instanceof Float64Array, 'should be Float64Array')
  assert(proba.length === 5 * nClasses, `proba.length=${proba.length}`)

  // Each row should sum to ~1 and all values in [0, 1]
  for (let i = 0; i < 5; i++) {
    let sum = 0
    for (let j = 0; j < nClasses; j++) {
      const p = proba[i * nClasses + j]
      assert(p >= 0 && p <= 1, `proba out of range: ${p}`)
      sum += p
    }
    assertClose(sum, 1.0, 1e-6, `row ${i} sum=${sum}`)
  }
  model.dispose()
})

await test('score() returns accuracy', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 40, 3, 3)

  model.fit(X, y)

  const s = await model.score(X, y)
  assert(typeof s === 'number', 'score should be a number')
  assert(s >= 0 && s <= 1, `score out of range: ${s}`)
  model.dispose()
})

await test('getParams() / setParams()', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 100 }, { ort })
  const params = model.getParams()
  assert(params.maxSupport === 100, `maxSupport=${params.maxSupport}`)
  assert(params.seed === 42, `seed=${params.seed}`)

  model.setParams({ seed: 7 })
  assert(model.getParams().seed === 7, 'seed should be 7')
  model.dispose()
})

await test('predict before fit throws NotFittedError', async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  let threw = false
  try { await model.predict([[1, 2, 3]]) } catch (e) { threw = e.name === 'NotFittedError' }
  assert(threw, 'should throw NotFittedError')
  model.dispose()
})

// ============================================================
// MitraClassifier: save / load round-trip
// ============================================================

console.log('\n=== Classifier Save/Load ===')

await test('save() produces valid WLRN bundle', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 20, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 30, 3, 3)
  model.fit(X, y)

  const bundle = model.save()
  assert(bundle instanceof Uint8Array, 'should be Uint8Array')
  assert(bundle.length > 16, 'bundle too small')

  // Validate magic
  const magic = String.fromCharCode(bundle[0], bundle[1], bundle[2], bundle[3])
  assert(magic === 'WLRN', `bad magic: ${magic}`)

  // Full validation (checks SHA-256)
  const { manifest, toc } = validateBundle(bundle)
  assert(manifest.typeId === 'wlearn.mitra_onnx.classifier@1', `typeId=${manifest.typeId}`)
  assert(toc.length === 3, `toc.length=${toc.length}`)

  const ids = toc.map(e => e.id).sort()
  assert(ids[0] === 'meta', `first artifact: ${ids[0]}`)
  assert(ids[1] === 'support_x', `second artifact: ${ids[1]}`)
  assert(ids[2] === 'support_y', `third artifact: ${ids[2]}`)
  model.dispose()
})

await test('save/load round-trip produces same predictions', async () => {
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
  assert(loaded.isFitted, 'loaded should be fitted')
  assert(loaded.nrClass === 3, `nrClass=${loaded.nrClass}`)
  assert(loaded.nrFeature === 3, `nrFeature=${loaded.nrFeature}`)

  const predsAfter = await loaded.predict(Xt)
  assert(predsAfter.length === predsBefore.length, 'length mismatch')
  for (let i = 0; i < predsBefore.length; i++) {
    assert(predsBefore[i] === predsAfter[i], `pred mismatch at ${i}: ${predsBefore[i]} vs ${predsAfter[i]}`)
  }
  loaded.dispose()
})

await test('load without onnxSource throws', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10 }, { ort })
  const rng = makeLCG(1)
  const { X, y } = makeClassificationData(rng, 20, 3)
  model.fit(X, y)
  const bundle = model.save()
  model.dispose()

  let threw = false
  try { await MitraClassifier.load(bundle) } catch { threw = true }
  assert(threw, 'should throw without onnxSource')
})

// ============================================================
// MitraRegressor
// ============================================================

console.log('\n=== MitraRegressor ===')

await test('create() returns unfitted regressor', async () => {
  const model = await MitraRegressor.create(regressorSession, {}, { ort })
  assert(!model.isFitted, 'should not be fitted')
  assert(model.capabilities.classifier === false, 'should not be classifier')
  assert(model.capabilities.regressor === true, 'should be regressor')
  assert(model.capabilities.predictProba === false, 'should not support predictProba')
  model.dispose()
})

await test('fit() and predict() on regression data', async () => {
  const model = await MitraRegressor.create(regressorSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeRegressionData(rng, 40, 3)

  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  assert(model.nrFeature === 3, `nrFeature=${model.nrFeature}`)

  const testRng = makeLCG(20)
  const { X: Xt } = makeRegressionData(testRng, 5, 3)
  const preds = await model.predict(Xt)

  assert(preds instanceof Float64Array, 'should be Float64Array')
  assert(preds.length === 5, `preds.length=${preds.length}`)

  // Predictions should be finite numbers
  for (let i = 0; i < preds.length; i++) {
    assert(isFinite(preds[i]), `non-finite prediction at ${i}: ${preds[i]}`)
  }
  model.dispose()
})

await test('regressor score() returns R2', async () => {
  const model = await MitraRegressor.create(regressorSession, { maxSupport: 30, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeRegressionData(rng, 40, 3)
  model.fit(X, y)

  const s = await model.score(X, y)
  assert(typeof s === 'number', 'score should be a number')
  assert(isFinite(s), `score should be finite: ${s}`)
  model.dispose()
})

// ============================================================
// Regressor Save/Load
// ============================================================

console.log('\n=== Regressor Save/Load ===')

await test('regressor save/load round-trip', async () => {
  const model = await MitraRegressor.create(regressorSession, { maxSupport: 20, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeRegressionData(rng, 30, 3)
  model.fit(X, y)

  const testRng = makeLCG(20)
  const { X: Xt } = makeRegressionData(testRng, 5, 3)
  const predsBefore = await model.predict(Xt)

  const bundle = model.save()
  const { manifest } = decodeBundle(bundle)
  assert(manifest.typeId === 'wlearn.mitra_onnx.regressor@1', `typeId=${manifest.typeId}`)

  model.dispose()

  const loaded = await MitraRegressor.load(bundle, regressorSession, { ort })
  assert(loaded.isFitted, 'loaded should be fitted')
  assert(loaded.nrFeature === 3, `nrFeature=${loaded.nrFeature}`)

  const predsAfter = await loaded.predict(Xt)
  assert(predsAfter.length === predsBefore.length, 'length mismatch')
  for (let i = 0; i < predsBefore.length; i++) {
    assertClose(predsBefore[i], predsAfter[i], 1e-6, `pred mismatch at ${i}`)
  }
  loaded.dispose()
})

// ============================================================
// Dispose / error handling
// ============================================================

console.log('\n=== Dispose & Error Handling ===')

await test('dispose() is idempotent', async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  model.dispose()
  model.dispose() // should not throw
})

await test('predict after dispose throws DisposedError', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10 }, { ort })
  const rng = makeLCG(1)
  const { X, y } = makeClassificationData(rng, 20, 3)
  model.fit(X, y)
  model.dispose()

  let threw = false
  try { await model.predict([[1, 2, 3]]) } catch (e) { threw = e.name === 'DisposedError' }
  assert(threw, 'should throw DisposedError')
})

await test('fit after dispose throws DisposedError', async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  model.dispose()

  let threw = false
  try { model.fit([[1, 2, 3]], [0]) } catch (e) { threw = e.name === 'DisposedError' }
  assert(threw, 'should throw DisposedError')
})

await test('save before fit throws NotFittedError', async () => {
  const model = await MitraClassifier.create(classifierSession, {}, { ort })
  let threw = false
  try { model.save() } catch (e) { threw = e.name === 'NotFittedError' }
  assert(threw, 'should throw NotFittedError')
  model.dispose()
})

await test('feature dimension mismatch throws ValidationError', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10 }, { ort })
  model.fit([[1, 2, 3], [4, 5, 6]], [0, 1])

  let threw = false
  try { await model.predict([[1, 2]]) } catch (e) { threw = e.name === 'ValidationError' }
  assert(threw, 'should throw ValidationError for wrong feature count')
  model.dispose()
})

// ============================================================
// Support set selection
// ============================================================

console.log('\n=== Support Set Selection ===')

await test('maxSupport limits support set size', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 10, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 100, 3, 3)
  model.fit(X, y)

  // Check bundle to verify support set size
  const bundle = model.save()
  const { toc, blobs } = decodeBundle(bundle)
  const xEntry = toc.find(e => e.id === 'support_x')
  const nSupport = xEntry.length / (3 * 4) // 3 features, float32
  assert(nSupport === 10, `nSupport=${nSupport}, expected 10`)
  model.dispose()
})

await test('small dataset uses all data', async () => {
  const model = await MitraClassifier.create(classifierSession, { maxSupport: 512, seed: 42 }, { ort })
  const rng = makeLCG(10)
  const { X, y } = makeClassificationData(rng, 8, 3, 2)
  model.fit(X, y)

  const bundle = model.save()
  const { toc, blobs } = decodeBundle(bundle)
  const xEntry = toc.find(e => e.id === 'support_x')
  const nSupport = xEntry.length / (3 * 4)
  assert(nSupport === 8, `nSupport=${nSupport}, expected 8`)
  model.dispose()
})

await test('deterministic: same seed produces same predictions', async () => {
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

  assert(preds1.length === preds2.length, 'length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `pred mismatch at ${i}: ${preds1[i]} vs ${preds2[i]}`)
  }
})

// ============================================================
// Summary
// ============================================================

console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed\n`)
if (failed > 0) process.exit(1)
