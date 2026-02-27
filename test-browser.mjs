/**
 * Playwright test: load Mitra ONNX classifier in Chromium and run inference.
 *
 * Usage: node test-browser.mjs
 */

import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const PORT = 9876
const DIR = path.dirname(new URL(import.meta.url).pathname)

const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
}

// Simple static file server
function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      let filePath
      if (req.url === '/') {
        filePath = path.join(DIR, 'test-browser.html')
      } else if (req.url.startsWith('/ort')) {
        filePath = path.join(DIR, 'node_modules/onnxruntime-web/dist', req.url)
      } else {
        filePath = path.join(DIR, req.url)
      }

      if (!fs.existsSync(filePath)) {
        res.writeHead(404)
        res.end('Not found')
        return
      }

      const ext = path.extname(filePath)
      const mime = MIME[ext] || 'application/octet-stream'

      // Stream large files (ONNX models)
      const stat = fs.statSync(filePath)
      res.writeHead(200, {
        'Content-Type': mime,
        'Content-Length': stat.size,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      })
      fs.createReadStream(filePath).pipe(res)
    })

    server.listen(PORT, () => {
      console.log(`Server listening on http://localhost:${PORT}`)
      resolve(server)
    })
  })
}

async function main() {
  const server = await startServer()

  let exitCode = 1
  try {
    const browser = await chromium.launch({ headless: true })
    const page = await browser.newPage()

    // Collect console output
    const logs = []
    page.on('console', msg => {
      const text = msg.text()
      logs.push(text)
      console.log(`  [browser] ${text}`)
    })

    page.on('pageerror', err => {
      console.error(`  [browser error] ${err.message}`)
    })

    console.log('Navigating to test page...')
    await page.goto(`http://localhost:${PORT}/`, { timeout: 300000 })

    // Wait for the title to change to PASS or FAIL (model load + inference)
    console.log('Waiting for inference to complete (this may take a while for 290MB model)...')
    await page.waitForFunction(
      () => document.title === 'PASS' || document.title === 'FAIL',
      { timeout: 300000 }
    )

    const title = await page.title()
    console.log(`\nBrowser test result: ${title}`)

    if (title === 'PASS') {
      exitCode = 0
    }

    await browser.close()
  } catch (e) {
    console.error(`Test error: ${e.message}`)
  } finally {
    server.close()
  }

  process.exit(exitCode)
}

main()
