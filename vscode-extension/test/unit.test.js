#!/usr/bin/env node
/**
 * Minimal unit tests for benchmark_paper.ts core functions.
 * Run: node test/unit.test.js
 */
const assert = require('assert');
let passed = 0;
let failed = 0;

function test(name, fn) {
    try { fn(); passed++; console.log('  \u2713 ' + name); }
    catch (e) { failed++; console.log('  \u2717 ' + name + ': ' + e.message); }
}

console.log('DEBATE EXT \u2014 Unit Tests\n');

const bp = require('../out/benchmark_paper');

if (typeof bp.extractPythonCode === 'function') {
    test('extractPythonCode: fence python', () => {
        const input = '```python\ndef f(): pass\n```';
        assert.strictEqual(bp.extractPythonCode(input), 'def f(): pass');
    });
    test('extractPythonCode: fence py', () => {
        const input = '```py\ndef f(): pass\n```';
        assert.strictEqual(bp.extractPythonCode(input), 'def f(): pass');
    });
    test('extractPythonCode: no fence', () => {
        assert.strictEqual(bp.extractPythonCode('def f(): pass'), 'def f(): pass');
    });
} else {
    console.log('  \u26a0 extractPythonCode not exported \u2014 skip');
}

if (typeof bp.passAtK === 'function') {
    test('passAtK: n=1, c=1, k=1 -> 1.0', () => {
        assert.strictEqual(bp.passAtK(1, 1, 1), 1.0);
    });
    test('passAtK: n=1, c=0, k=1 -> 0.0', () => {
        assert.strictEqual(bp.passAtK(1, 0, 1), 0.0);
    });
    test('passAtK: n=10, c=5, k=1 -> 0.5', () => {
        assert(Math.abs(bp.passAtK(10, 5, 1) - 0.5) < 0.01);
    });
} else {
    console.log('  \u26a0 passAtK not exported \u2014 skip');
}

if (typeof bp.computeStdDev === 'function') {
    test('computeStdDev: [2,4,4,4,5,5,7,9] ~ 2.0', () => {
        assert(Math.abs(bp.computeStdDev([2,4,4,4,5,5,7,9]) - 2.138) < 0.01);
    });
    test('computeStdDev: single value -> 0', () => {
        assert.strictEqual(bp.computeStdDev([42]), 0);
    });
} else {
    console.log('  \u26a0 computeStdDev not exported \u2014 skip');
}

console.log('\n' + passed + ' passed, ' + failed + ' failed');
process.exit(failed > 0 ? 1 : 0);
