#!/usr/bin/env node

/**
 * Download Mitra ONNX models from GitHub Release.
 *
 * Usage:
 *   node scripts/download-models.mjs [--tag TAG] [--dir DIR]
 *   npx @wlearn/mitra download-models
 */

import { createWriteStream, existsSync, readFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { pipeline } from 'stream/promises'
import { createHash } from 'crypto'

const __dirname = dirname(fileURLToPath(import.meta.url))
const ROOT = join(__dirname, '..')

// Parse args
const args = process.argv.slice(2)
let tag = null
let dir = ROOT

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--tag' && args[i + 1]) { tag = args[++i]; continue }
  if (args[i] === '--dir' && args[i + 1]) { dir = args[++i]; continue }
  if (args[i] === '--help' || args[i] === '-h') {
    console.log('Usage: download-models.mjs [--tag TAG] [--dir DIR]')
    process.exit(0)
  }
}

// Read manifest
const manifest = JSON.parse(readFileSync(join(ROOT, 'models.json'), 'utf8'))
if (!tag) tag = manifest.tag

const BASE_URL = `https://github.com/wlearn-org/mitra-onnx/releases/download/${tag}`

for (const [filename, meta] of Object.entries(manifest.files)) {
  const dest = join(dir, filename)

  if (existsSync(dest)) {
    console.log(`SKIP: ${filename} (already exists)`)
    continue
  }

  const url = `${BASE_URL}/${filename}`
  console.log(`Downloading ${filename} from ${tag}...`)

  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok) {
    console.error(`FAILED: ${filename} -- HTTP ${res.status}`)
    process.exit(1)
  }

  const ws = createWriteStream(dest)
  await pipeline(res.body, ws)

  // Verify SHA-256 if available
  if (meta.sha256) {
    const hash = createHash('sha256')
    const data = readFileSync(dest)
    hash.update(data)
    const actual = hash.digest('hex')
    if (actual !== meta.sha256) {
      console.error(`CHECKSUM MISMATCH: ${filename}`)
      console.error(`  expected: ${meta.sha256}`)
      console.error(`  actual:   ${actual}`)
      process.exit(1)
    }
    console.log(`OK: ${filename} (sha256 verified)`)
  } else {
    console.log(`OK: ${filename}`)
  }
}

console.log('\nDone.')
