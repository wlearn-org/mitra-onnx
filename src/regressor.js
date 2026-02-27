import {
  encodeBundle, decodeBundle, encodeJSON, decodeJSON,
  normalizeX, normalizeY,
  NotFittedError, DisposedError, ValidationError,
  r2Score
} from '@wlearn/core'
import { selectSupport, runInference, sha256Sync } from './shared.js'

const TYPE_ID = 'wlearn.mitra_onnx.regressor@1'
const LOAD_SENTINEL = Symbol('load')

export class MitraRegressor {
  #ort
  #session
  #ownSession
  #xSupport
  #ySupport
  #nFeatures
  #nSupport
  #params
  #fitted
  #freed
  #onnxSha256

  constructor(sentinel, ort, session, ownSession, params, onnxSha256) {
    if (sentinel !== LOAD_SENTINEL) {
      throw new Error('Use MitraRegressor.create() to construct instances')
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
    this.#nFeatures = 0
    this.#nSupport = 0
  }

  /**
   * Create a MitraRegressor.
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
      session = onnxSource
      ownSession = false
      onnxSha256 = opts.onnxSha256 || null
    } else {
      throw new ValidationError('onnxSource must be Uint8Array (ONNX bytes) or an InferenceSession')
    }

    return new MitraRegressor(LOAD_SENTINEL, ort, session, ownSession, params, onnxSha256)
  }

  get capabilities() {
    return {
      classifier: false,
      regressor: true,
      predictProba: false,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  get isFitted() { return this.#fitted }
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

    // Convert X to Float32 for ONNX
    const xFloat32 = new Float32Array(rows * cols)
    for (let i = 0; i < xData.length; i++) xFloat32[i] = xData[i]

    // Convert y to Float32 for regressor
    const yFloat32 = new Float32Array(yData.length)
    for (let i = 0; i < yData.length; i++) yFloat32[i] = yData[i]

    // Select support set (not stratified for regression)
    const { xSupport, ySupport, nSupport } = selectSupport(
      xFloat32, yFloat32, cols,
      this.#params.maxSupport, this.#params.seed, false
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
      this.#nFeatures, this.#nSupport, nQuery, false
    )

    // Regressor output shape: (B, N_query) or (B, N_query, 1)
    const outputData = output.data
    const predictions = new Float64Array(nQuery)
    for (let i = 0; i < nQuery; i++) {
      predictions[i] = outputData[i]
    }

    return predictions
  }

  async score(X, y) {
    const predictions = await this.predict(X)
    const yData = normalizeY(y)
    return r2Score(yData, predictions)
  }

  save() {
    this.#ensureFitted()

    const meta = {
      nFeatures: this.#nFeatures,
      nSupport: this.#nSupport,
      onnxSha256: this.#onnxSha256,
      seed: this.#params.seed
    }

    return encodeBundle(
      {
        typeId: TYPE_ID,
        params: this.getParams(),
        metadata: { nFeatures: this.#nFeatures }
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
    return MitraRegressor._fromBundle(manifest, toc, blobs, onnxSource, opts)
  }

  static async _fromBundle(manifest, toc, blobs, onnxSource, opts = {}) {
    if (!onnxSource) {
      throw new ValidationError('onnxSource is required to load a MitraRegressor (ONNX model is not embedded in the bundle)')
    }

    const model = await MitraRegressor.create(onnxSource, manifest.params || {}, opts)

    const metaEntry = toc.find(e => e.id === 'meta')
    const meta = decodeJSON(blobs.subarray(metaEntry.offset, metaEntry.offset + metaEntry.length))

    const xEntry = toc.find(e => e.id === 'support_x')
    const xBytes = blobs.slice(xEntry.offset, xEntry.offset + xEntry.length)
    model.#xSupport = new Float32Array(xBytes.buffer, xBytes.byteOffset, xBytes.byteLength / 4)

    const yEntry = toc.find(e => e.id === 'support_y')
    const yBytes = blobs.slice(yEntry.offset, yEntry.offset + yEntry.length)
    model.#ySupport = new Float32Array(yBytes.buffer, yBytes.byteOffset, yBytes.byteLength / 4)

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
      if (typeof this.#session.release === 'function') {
        this.#session.release()
      }
    }

    this.#session = null
    this.#xSupport = null
    this.#ySupport = null
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
