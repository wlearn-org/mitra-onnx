import {
  encodeBundle, decodeBundle, encodeJSON, decodeJSON,
  normalizeX, normalizeY,
  NotFittedError, DisposedError, ValidationError,
  accuracy
} from '@wlearn/core'
import { selectSupport, runInference, softmax, sha256Sync } from './shared.js'

const TYPE_ID = 'wlearn.mitra_onnx.classifier@1'
const LOAD_SENTINEL = Symbol('load')

export class MitraClassifier {
  #ort
  #session
  #ownSession
  #xSupport
  #ySupport
  #classes
  #nFeatures
  #nSupport
  #params
  #fitted
  #freed
  #onnxSha256

  constructor(sentinel, ort, session, ownSession, params, onnxSha256) {
    if (sentinel !== LOAD_SENTINEL) {
      throw new Error('Use MitraClassifier.create() to construct instances')
    }
    this.#ort = ort
    this.#session = session
    this.#ownSession = ownSession
    this.#params = { maxSupport: 512, seed: 42, ...params }
    this.#onnxSha256 = onnxSha256 || null
    this.#fitted = false
    this.#freed = false
    this.#xSupport = null
    this.#ySupport = null
    this.#classes = null
    this.#nFeatures = 0
    this.#nSupport = 0
  }

  /**
   * Create a MitraClassifier.
   *
   * @param {Uint8Array|object} onnxSource - ONNX model bytes or a pre-created InferenceSession
   * @param {object} [params] - { maxSupport, seed }
   * @param {object} [opts] - { ort } onnxruntime module (auto-detected if not provided)
   */
  static async create(onnxSource, params = {}, opts = {}) {
    const ort = opts.ort || await detectOrt()
    let session, ownSession, onnxSha256

    if (onnxSource instanceof Uint8Array || ArrayBuffer.isView(onnxSource)) {
      const bytes = onnxSource instanceof Uint8Array ? onnxSource : new Uint8Array(onnxSource.buffer, onnxSource.byteOffset, onnxSource.byteLength)
      onnxSha256 = sha256Sync(bytes)
      session = await ort.InferenceSession.create(bytes.buffer)
      ownSession = true
    } else if (onnxSource && typeof onnxSource === 'object' && typeof onnxSource.run === 'function') {
      // Pre-created session
      session = onnxSource
      ownSession = false
      onnxSha256 = opts.onnxSha256 || null
    } else {
      throw new ValidationError('onnxSource must be Uint8Array (ONNX bytes) or an InferenceSession')
    }

    return new MitraClassifier(LOAD_SENTINEL, ort, session, ownSession, params, onnxSha256)
  }

  get capabilities() {
    return {
      classifier: true,
      regressor: false,
      predictProba: true,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  get isFitted() { return this.#fitted }
  get classes() { this.#ensureFitted(); return this.#classes }
  get nrClass() { this.#ensureFitted(); return this.#classes.length }
  get nrFeature() { this.#ensureFitted(); return this.#nFeatures }

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    this.#ensureNotFreed()
    Object.assign(this.#params, p)
    return this
  }

  fit(X, y) {
    this.#ensureNotFreed()
    if (this.#fitted) {
      throw new ValidationError('Model is already fitted. Create a new instance to fit again.')
    }

    const { data: xData, rows, cols } = normalizeX(X)
    const yData = normalizeY(y)

    if (rows !== yData.length) {
      throw new ValidationError(`X has ${rows} rows but y has ${yData.length} elements`)
    }

    this.#nFeatures = cols

    // Detect classes
    const classSet = new Set()
    for (let i = 0; i < yData.length; i++) classSet.add(yData[i])
    this.#classes = new Int32Array([...classSet].sort((a, b) => a - b))

    // Convert X to Float32 for ONNX
    const xFloat32 = new Float32Array(rows * cols)
    for (let i = 0; i < xData.length; i++) xFloat32[i] = xData[i]

    // Convert y to Int32 for support selection
    const yInt32 = new Int32Array(yData.length)
    for (let i = 0; i < yData.length; i++) yInt32[i] = yData[i]

    // Select support set
    const { xSupport, ySupport, nSupport } = selectSupport(
      xFloat32, yInt32, cols,
      this.#params.maxSupport, this.#params.seed, true
    )

    this.#xSupport = xSupport
    this.#ySupport = ySupport
    this.#nSupport = nSupport
    this.#fitted = true

    return this
  }

  async predict(X) {
    this.#ensureFitted()

    const { xQuery, nQuery } = this.#prepareQuery(X)
    const output = await runInference(
      this.#ort, this.#session,
      this.#xSupport, this.#ySupport, xQuery,
      this.#nFeatures, this.#nSupport, nQuery, true
    )

    const outputData = output.data
    const nClasses = this.#classes.length
    const predictions = new Float64Array(nQuery)

    // Argmax over logits, map back to class labels
    for (let i = 0; i < nQuery; i++) {
      let bestIdx = 0
      let bestVal = -Infinity
      for (let j = 0; j < nClasses; j++) {
        const v = outputData[i * nClasses + j]
        if (v > bestVal) { bestVal = v; bestIdx = j }
      }
      predictions[i] = this.#classes[bestIdx]
    }

    return predictions
  }

  async predictProba(X) {
    this.#ensureFitted()

    const { xQuery, nQuery } = this.#prepareQuery(X)
    const output = await runInference(
      this.#ort, this.#session,
      this.#xSupport, this.#ySupport, xQuery,
      this.#nFeatures, this.#nSupport, nQuery, true
    )

    const nClasses = this.#classes.length
    return softmax(output.data, nQuery, nClasses)
  }

  async score(X, y) {
    const predictions = await this.predict(X)
    const yData = normalizeY(y)
    return accuracy(yData, predictions)
  }

  save() {
    this.#ensureFitted()

    const meta = {
      nFeatures: this.#nFeatures,
      nSupport: this.#nSupport,
      classes: Array.from(this.#classes),
      onnxSha256: this.#onnxSha256,
      seed: this.#params.seed
    }

    return encodeBundle(
      {
        typeId: TYPE_ID,
        params: this.getParams(),
        metadata: { nClasses: this.#classes.length, nFeatures: this.#nFeatures }
      },
      [
        { id: 'meta', data: encodeJSON(meta) },
        { id: 'support_x', data: new Uint8Array(this.#xSupport.buffer, this.#xSupport.byteOffset, this.#xSupport.byteLength) },
        { id: 'support_y', data: new Uint8Array(this.#ySupport.buffer, this.#ySupport.byteOffset, this.#ySupport.byteLength) }
      ]
    )
  }

  static async load(bytes, onnxSource, opts = {}) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return MitraClassifier._fromBundle(manifest, toc, blobs, onnxSource, opts)
  }

  static async _fromBundle(manifest, toc, blobs, onnxSource, opts = {}) {
    if (!onnxSource) {
      throw new ValidationError('onnxSource is required to load a MitraClassifier (ONNX model is not embedded in the bundle)')
    }

    const model = await MitraClassifier.create(onnxSource, manifest.params || {}, opts)

    // Restore meta
    const metaEntry = toc.find(e => e.id === 'meta')
    const meta = decodeJSON(blobs.subarray(metaEntry.offset, metaEntry.offset + metaEntry.length))

    // Restore support_x
    const xEntry = toc.find(e => e.id === 'support_x')
    const xBytes = blobs.slice(xEntry.offset, xEntry.offset + xEntry.length)
    model.#xSupport = new Float32Array(xBytes.buffer, xBytes.byteOffset, xBytes.byteLength / 4)

    // Restore support_y
    const yEntry = toc.find(e => e.id === 'support_y')
    const yBytes = blobs.slice(yEntry.offset, yEntry.offset + yEntry.length)
    model.#ySupport = new Int32Array(yBytes.buffer, yBytes.byteOffset, yBytes.byteLength / 4)

    model.#classes = new Int32Array(meta.classes)
    model.#nFeatures = meta.nFeatures
    model.#nSupport = meta.nSupport
    model.#onnxSha256 = meta.onnxSha256
    model.#fitted = true

    return model
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#ownSession && this.#session) {
      // onnxruntime sessions may have a release() method
      if (typeof this.#session.release === 'function') {
        this.#session.release()
      }
    }

    this.#session = null
    this.#xSupport = null
    this.#ySupport = null
    this.#classes = null
    this.#fitted = false
  }

  #prepareQuery(X) {
    const { data: xData, rows, cols } = normalizeX(X)

    if (cols !== this.#nFeatures) {
      throw new ValidationError(`X has ${cols} features but model was fitted with ${this.#nFeatures}`)
    }

    const xQuery = new Float32Array(rows * cols)
    for (let i = 0; i < xData.length; i++) xQuery[i] = xData[i]

    return { xQuery, nQuery: rows }
  }

  #ensureFitted() {
    this.#ensureNotFreed()
    if (!this.#fitted) throw new NotFittedError()
  }

  #ensureNotFreed() {
    if (this.#freed) throw new DisposedError()
  }
}

async function detectOrt() {
  try {
    return await import('onnxruntime-node')
  } catch {
    try {
      return await import('onnxruntime-web')
    } catch {
      throw new Error('No ONNX Runtime found. Install onnxruntime-node or onnxruntime-web.')
    }
  }
}
