#!/usr/bin/env node
/**
 * Smoke test: verifies build artifacts and security invariants.
 * Run: npm run smoke
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const root = path.join(__dirname, '..');
let failures = 0;

function check(label, ok) {
    if (ok) {
        console.log(`  ✓ ${label}`);
    } else {
        console.log(`  ✗ ${label}`);
        failures++;
    }
}

console.log('CRISTAL CODE — Smoke Tests\n');

// 1. Check compiled artifacts exist
check('out/benchmark.js exists', fs.existsSync(path.join(root, 'out', 'benchmark.js')));
check('out/benchmark_v2.js exists', fs.existsSync(path.join(root, 'out', 'benchmark_v2.js')));
check('out/benchmark_paper.js exists', fs.existsSync(path.join(root, 'out', 'benchmark_paper.js')));

// 2. --help should exit 0
try {
    execSync('node out/benchmark.js --help', { cwd: root, stdio: 'pipe' });
    check('benchmark.js --help exits 0', true);
} catch (e) {
    check('benchmark.js --help exits 0', e.status === 0);
}

try {
    execSync('node out/benchmark_v2.js --help', { cwd: root, stdio: 'pipe' });
    check('benchmark_v2.js --help exits 0', true);
} catch (e) {
    check('benchmark_v2.js --help exits 0', e.status === 0);
}

try {
    execSync('node out/benchmark_paper.js --help', { cwd: root, stdio: 'pipe' });
    check('benchmark_paper.js --help exits 0', true);
} catch (e) {
    check('benchmark_paper.js --help exits 0', e.status === 0);
}

// 3. Security: no forbidden patterns in src/
const forbidden = /\.claude\/\.credentials|\.codex\/auth|auth\.openai\.com|oauth\/token|client_id.*app_/;
const srcDir = path.join(root, 'src');
const srcFiles = fs.readdirSync(srcDir).filter(f => f.endsWith('.ts'));
let securityOk = true;
for (const file of srcFiles) {
    const content = fs.readFileSync(path.join(srcDir, file), 'utf-8');
    if (forbidden.test(content)) {
        console.log(`  ✗ SECURITY: forbidden pattern in ${file}`);
        securityOk = false;
        failures++;
    }
}
check('No forbidden credential patterns in src/', securityOk);

// 3b. Check auxiliary scripts exist
check('scripts/setup_mbppplus.py exists', fs.existsSync(path.join(root, 'scripts', 'setup_mbppplus.py')));

// 4. Data file (warning only, not blocking)
const dataExists = fs.existsSync(path.join(root, 'data', 'MbppPlus.jsonl'));
if (dataExists) {
    console.log('  \u2713 data/MbppPlus.jsonl exists');
} else {
    console.log('  \u26a0 data/MbppPlus.jsonl missing (run: python scripts/setup_mbppplus.py)');
}

// 5. No .env file
check('.env does not exist', !fs.existsSync(path.join(root, '.env')));

console.log('');
if (failures === 0) {
    console.log('SMOKE OK ✓');
    process.exit(0);
} else {
    console.log(`SMOKE FAILED: ${failures} issue(s)`);
    process.exit(1);
}
