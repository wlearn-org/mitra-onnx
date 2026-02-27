import { test } from 'node:test'
import assert from 'node:assert/strict'
import { selectSupport, softmax } from '../src/shared.js'
import { MitraClassifier } from '../src/classifier.js'

// ============================================================
// selectSupport() -- stratified (classification)
// ============================================================

test('selectSupport: stratified sampling with maxSupport', () => {
  const nFeatures = 3
  const rows = 30
  const X = new Float32Array(rows * nFeatures)
  const y = new Int32Array(rows)
  // 3 classes, 10 samples each
  for (let i = 0; i < rows; i++) {
    y[i] = i % 3
    for (let j = 0; j < nFeatures; j++) {
      X[i * nFeatures + j] = i * 0.1 + j
    }
  }

  const { xSupport, ySupport, nSupport } = selectSupport(X, y, nFeatures, 9, 42, true)
  assert.equal(nSupport, 9)
  assert.equal(xSupport.length, 9 * nFeatures)
  assert.equal(ySupport.length, 9)
  assert.ok(ySupport instanceof Int32Array)

  // Check stratification: 3 per class
  const counts = [0, 0, 0]
  for (let i = 0; i < nSupport; i++) counts[ySupport[i]]++
  assert.equal(counts[0], 3)
  assert.equal(counts[1], 3)
  assert.equal(counts[2], 3)
})

test('selectSupport: stratified with uneven remainder', () => {
  const nFeatures = 2
  const rows = 20
  const X = new Float32Array(rows * nFeatures)
  const y = new Int32Array(rows)
  for (let i = 0; i < rows; i++) {
    y[i] = i % 3
    for (let j = 0; j < nFeatures; j++) X[i * nFeatures + j] = i
  }

  // maxSupport=7, 3 classes: 3+2+2 or 2+3+2 etc
  const { nSupport, ySupport } = selectSupport(X, y, nFeatures, 7, 42, true)
  assert.equal(nSupport, 7)
  const counts = [0, 0, 0]
  for (let i = 0; i < nSupport; i++) counts[ySupport[i]]++
  // Total should be 7
  assert.equal(counts[0] + counts[1] + counts[2], 7)
  // Each class gets at least floor(7/3)=2
  assert.ok(counts[0] >= 2)
  assert.ok(counts[1] >= 2)
  assert.ok(counts[2] >= 2)
})

// ============================================================
// selectSupport() -- random (regression)
// ============================================================

test('selectSupport: random sampling for regression', () => {
  const nFeatures = 2
  const rows = 20
  const X = new Float32Array(rows * nFeatures)
  const y = new Float32Array(rows)
  for (let i = 0; i < rows; i++) {
    y[i] = i * 1.5
    for (let j = 0; j < nFeatures; j++) X[i * nFeatures + j] = i + j
  }

  const { xSupport, ySupport, nSupport } = selectSupport(X, y, nFeatures, 5, 42, false)
  assert.equal(nSupport, 5)
  assert.equal(xSupport.length, 5 * nFeatures)
  assert.ok(ySupport instanceof Float32Array)
  assert.equal(ySupport.length, 5)
})

// ============================================================
// selectSupport() -- small dataset (all data used)
// ============================================================

test('selectSupport: small dataset uses all data', () => {
  const nFeatures = 3
  const rows = 5
  const X = new Float32Array(rows * nFeatures)
  const y = new Int32Array(rows)
  for (let i = 0; i < rows; i++) {
    y[i] = i % 2
    for (let j = 0; j < nFeatures; j++) X[i * nFeatures + j] = i * 10 + j
  }

  const { xSupport, ySupport, nSupport } = selectSupport(X, y, nFeatures, 100, 42, true)
  assert.equal(nSupport, 5)
  assert.equal(xSupport.length, 5 * nFeatures)
  // Should be exact copy
  for (let i = 0; i < rows * nFeatures; i++) {
    assert.equal(xSupport[i], X[i])
  }
})

// ============================================================
// selectSupport() -- deterministic with same seed
// ============================================================

test('selectSupport: deterministic with same seed', () => {
  const nFeatures = 2
  const rows = 50
  const X = new Float32Array(rows * nFeatures)
  const y = new Int32Array(rows)
  for (let i = 0; i < rows; i++) {
    y[i] = i % 4
    for (let j = 0; j < nFeatures; j++) X[i * nFeatures + j] = i + j * 0.5
  }

  const r1 = selectSupport(X, y, nFeatures, 10, 77, true)
  const r2 = selectSupport(X, y, nFeatures, 10, 77, true)
  assert.deepEqual(r1.xSupport, r2.xSupport)
  assert.deepEqual(r1.ySupport, r2.ySupport)
})

// ============================================================
// softmax()
// ============================================================

test('softmax: rows sum to 1', () => {
  const logits = new Float32Array([1, 2, 3, 0, 0, 0, -1, 0, 1])
  const proba = softmax(logits, 3, 3)
  assert.equal(proba.length, 9)
  assert.ok(proba instanceof Float64Array)

  for (let i = 0; i < 3; i++) {
    let sum = 0
    for (let j = 0; j < 3; j++) {
      const p = proba[i * 3 + j]
      assert.ok(p >= 0 && p <= 1, `proba out of range: ${p}`)
      sum += p
    }
    assert.ok(Math.abs(sum - 1.0) < 1e-10, `row ${i} sum=${sum}`)
  }
})

test('softmax: known values', () => {
  // softmax([0, 0]) = [0.5, 0.5]
  const logits = new Float32Array([0, 0])
  const proba = softmax(logits, 1, 2)
  assert.ok(Math.abs(proba[0] - 0.5) < 1e-10)
  assert.ok(Math.abs(proba[1] - 0.5) < 1e-10)
})

test('softmax: large values (numerical stability)', () => {
  const logits = new Float32Array([1000, 1001, 999])
  const proba = softmax(logits, 1, 3)
  let sum = 0
  for (let i = 0; i < 3; i++) {
    assert.ok(isFinite(proba[i]))
    assert.ok(proba[i] >= 0)
    sum += proba[i]
  }
  assert.ok(Math.abs(sum - 1.0) < 1e-10)
})

// ============================================================
// MitraClassifier constructor guard
// ============================================================

test('MitraClassifier: new without create() throws', () => {
  assert.throws(() => new MitraClassifier(), /create/)
})
