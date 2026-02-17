#!/usr/bin/env node
/**
 * CRISTAL CODE — Benchmark v2 (HumanEval + Execution + Judge Debate)
 *
 * Improvements over v1:
 *   - Uses HumanEval (164 Python problems with unit tests)
 *   - Actually EXECUTES generated code → pass/fail (pass@1)
 *   - 3 independent judges: Claude + GPT + DeepSeek
 *   - Judge debate: when scores diverge >20%, judges discuss + re-score
 *   - 5 runs for statistical variance
 *   - Full audit trail: all rounds, debates, scores logged
 *
 * Usage:
 *   npm run bench2                                   # all 164 problems, 6 configs, 5 runs
 *   npm run bench2 -- --runs 1 --limit 10            # quick test: 10 problems, 1 run
 *   npm run bench2 -- --configs claude-solo,openai-solo --limit 20
 *
 * Requires: ANTHROPIC_API_KEY + OPENAI_API_KEY env vars + Ollama running locally.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import * as http from 'http';
import { spawn } from 'child_process';

// ─── Types ───────────────────────────────────────────────────────────

type BenchConfig =
    | 'claude-solo' | 'openai-solo'
    | 'claude-lead' | 'openai-lead'
    | 'claude-orch' | 'openai-orch';

type Agent = 'claude' | 'openai';

const ALL_CONFIGS: BenchConfig[] = [
    'claude-solo', 'openai-solo',
    'claude-lead', 'openai-lead',
    'claude-orch', 'openai-orch',
];

const CONFIG_SHORT: Record<BenchConfig, string> = {
    'claude-solo': 'Cl.Solo', 'openai-solo': 'OA.Solo',
    'claude-lead': 'Cl.Lead', 'openai-lead': 'OA.Lead',
    'claude-orch': 'Cl.Orch', 'openai-orch': 'OA.Orch',
};

interface HumanEvalTask {
    task_id: string;
    prompt: string;
    entry_point: string;
    canonical_solution: string;
    test: string;
}

interface QualityScores {
    correctness: number;
    completeness: number;
    edgeCases: number;
    codeQuality: number;
    readability: number;
    total: number;
    justification: string;
}

interface JudgeRound {
    judge: string;
    scores: Partial<Record<BenchConfig, QualityScores>>;
}

interface DebateRound {
    trigger: string;           // which configs diverged
    messages: { judge: string; argument: string }[];
}

interface JudgeAudit {
    round1: JudgeRound[];
    divergenceDetected: boolean;
    debate?: DebateRound;
    round2?: JudgeRound[];       // re-scores after debate
    finalScores: Partial<Record<BenchConfig, QualityScores>>;
}

interface ConfigRunResult {
    config: BenchConfig;
    generatedCode: string;
    passed: boolean;             // execution test pass/fail
    execOutput: string;          // stdout/stderr from execution
    execTimeMs: number;
    duration: number;            // API call time
    error?: string;
}

interface TaskResult {
    taskId: string;
    entryPoint: string;
    run: number;
    timestamp: string;
    configs: ConfigRunResult[];
    judgeAudit: JudgeAudit;
}

interface BenchV2Report {
    meta: {
        date: string;
        claudeModel: string;
        openaiModel: string;
        ollamaModel: string;
        totalTasks: number;
        selectedTasks: number;
        configs: BenchConfig[];
        runs: number;
        divergenceThreshold: number;
    };
    tasks: TaskResult[];
    summary: {
        passAt1: Partial<Record<BenchConfig, number>>;
        avgQuality: Partial<Record<BenchConfig, number>>;
        debatesTriggered: number;
        totalEvaluations: number;
    };
}

// ─── Checkpoint for resume ───────────────────────────────────────────

interface CheckpointData {
    fingerprint: string;            // hash of params to ensure we resume with same config
    completedKeys: string[];        // "run:taskId" keys already done
    results: TaskResult[];
    debatesTriggered: number;
    lastSaved: string;
}

// ─── API callers (same as v1) ────────────────────────────────────────

function callClaude(
    prompt: string, timeoutMs: number, model: string
): Promise<{ content: string; error?: string }> {
    const apiKey = process.env.ANTHROPIC_API_KEY!;
    return new Promise((resolve) => {
        const body = JSON.stringify({
            model, max_tokens: 16384,
            messages: [{ role: 'user', content: prompt }],
        });
        const req = https.request({
            hostname: 'api.anthropic.com', path: '/v1/messages', method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': apiKey, 'anthropic-version': '2023-06-01',
            },
        }, (res) => {
            let data = '';
            res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    if (json.error) resolve({ content: '', error: json.error.message || JSON.stringify(json.error) });
                    else {
                        const content = json.content?.filter((b: { type: string }) => b.type === 'text')
                            .map((b: { text: string }) => b.text).join('') ?? '';
                        resolve({ content });
                    }
                } catch { resolve({ content: '', error: `Invalid JSON: ${data.slice(0, 200)}` }); }
            });
        });
        req.on('error', (err) => resolve({ content: '', error: String(err) }));
        const timer = setTimeout(() => { req.destroy(); resolve({ content: '', error: `Timeout ${timeoutMs}ms` }); }, timeoutMs);
        req.on('close', () => clearTimeout(timer));
        req.write(body); req.end();
    });
}

function callOpenAI(
    prompt: string, timeoutMs: number, model: string
): Promise<{ content: string; error?: string }> {
    const apiKey = process.env.OPENAI_API_KEY!;
    return new Promise((resolve) => {
        const body = JSON.stringify({
            model, messages: [{ role: 'user', content: prompt }], temperature: 0.2,
        });
        const req = https.request({
            hostname: 'api.openai.com', path: '/v1/chat/completions', method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        }, (res) => {
            let data = '';
            res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    if (json.error) resolve({ content: '', error: json.error.message || JSON.stringify(json.error) });
                    else resolve({ content: json.choices?.[0]?.message?.content ?? '' });
                } catch { resolve({ content: '', error: `Invalid JSON: ${data.slice(0, 200)}` }); }
            });
        });
        req.on('error', (err) => resolve({ content: '', error: String(err) }));
        const timer = setTimeout(() => { req.destroy(); resolve({ content: '', error: `Timeout ${timeoutMs}ms` }); }, timeoutMs);
        req.on('close', () => clearTimeout(timer));
        req.write(body); req.end();
    });
}

const OLLAMA_HOST = process.env.OLLAMA_HOST ?? 'host.docker.internal';
const OLLAMA_PORT = parseInt(process.env.OLLAMA_PORT ?? '11434', 10);

function callOllama(
    prompt: string, timeoutMs: number, model: string
): Promise<{ content: string; error?: string }> {
    return new Promise((resolve) => {
        const body = JSON.stringify({
            model, messages: [{ role: 'user', content: prompt }], temperature: 0.2, stream: false,
        });
        const req = http.request({
            hostname: OLLAMA_HOST, port: OLLAMA_PORT, path: '/v1/chat/completions', method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        }, (res) => {
            let data = '';
            res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    if (json.error) resolve({ content: '', error: json.error.message || JSON.stringify(json.error) });
                    else resolve({ content: json.choices?.[0]?.message?.content ?? '' });
                } catch { resolve({ content: '', error: `Invalid JSON: ${data.slice(0, 200)}` }); }
            });
        });
        req.on('error', (err) => resolve({ content: '', error: String(err) }));
        const timer = setTimeout(() => { req.destroy(); resolve({ content: '', error: `Timeout ${timeoutMs}ms` }); }, timeoutMs);
        req.on('close', () => clearTimeout(timer));
        req.write(body); req.end();
    });
}

function callAgent(
    agent: Agent, prompt: string, timeout: number,
    models: { claude: string; openai: string }
): Promise<{ content: string; error?: string }> {
    return agent === 'claude'
        ? callClaude(prompt, timeout, models.claude)
        : callOpenAI(prompt, timeout, models.openai);
}

// ─── Load HumanEval ──────────────────────────────────────────────────

function loadHumanEval(dataPath: string): HumanEvalTask[] {
    const content = fs.readFileSync(dataPath, 'utf-8');
    return content.trim().split('\n').map(line => JSON.parse(line));
}

// ─── Code extraction ─────────────────────────────────────────────────

function extractPythonCode(response: string, taskPrompt: string): string {
    let code = response;

    // Try to extract from ```python ... ``` blocks
    const codeBlocks = code.match(/```python\n([\s\S]*?)```/g);
    if (codeBlocks) {
        code = codeBlocks.map(b => b.replace(/```python\n/, '').replace(/```$/, '')).join('\n');
    } else {
        // Try ``` ... ``` blocks
        const generic = code.match(/```\n([\s\S]*?)```/g);
        if (generic) {
            code = generic.map(b => b.replace(/```\n/, '').replace(/```$/, '')).join('\n');
        }
    }

    // If the model returned the full function (including signature from prompt),
    // strip the prompt prefix so we only keep the body
    // Find if the code contains the function def from the prompt
    const defMatch = taskPrompt.match(/^(def \w+\()/m);
    if (defMatch && code.includes(defMatch[1])) {
        // Model returned the full function — strip everything up to and including the prompt
        const idx = code.indexOf(defMatch[1]);
        // Find where the prompt part ends in the code
        const promptEnd = taskPrompt.trimEnd();
        if (code.includes(promptEnd)) {
            code = code.slice(code.indexOf(promptEnd) + promptEnd.length);
        } else {
            // Just strip from the def line, keeping only the body
            const lines = code.slice(idx).split('\n');
            // Find first indented line after docstring
            let bodyStart = 0;
            let inDocstring = false;
            for (let i = 0; i < lines.length; i++) {
                if (lines[i].includes('"""') || lines[i].includes("'''")) {
                    inDocstring = !inDocstring;
                    if (!inDocstring && i > 0) { bodyStart = i + 1; break; }
                }
            }
            code = lines.slice(bodyStart).join('\n');
        }
    }

    // Normalize indentation: ensure body uses 4-space indent
    // Detect current indent of first non-empty line
    const lines = code.split('\n');
    const firstIndented = lines.find(l => l.match(/^\s+\S/));
    if (firstIndented) {
        const currentIndent = firstIndented.match(/^(\s+)/)![1];
        if (currentIndent !== '    ' && currentIndent.length > 0) {
            // Re-indent: replace current indent with 4 spaces proportionally
            const indentUnit = currentIndent;
            code = lines.map(l => {
                if (!l.trim()) return l;
                let level = 0;
                let remaining = l;
                while (remaining.startsWith(indentUnit)) {
                    level++;
                    remaining = remaining.slice(indentUnit.length);
                }
                return '    '.repeat(level) + remaining;
            }).join('\n');
        }
    }

    return code;
}

// ─── Code execution (sandboxed Python) ───────────────────────────────

function executeCode(
    taskPrompt: string, code: string, test: string, entryPoint: string, timeoutMs: number = 10_000
): Promise<{ passed: boolean; output: string; timeMs: number }> {
    return new Promise((resolve) => {
        // HumanEval format: prompt has the function signature + docstring,
        // model generates the body. We need to combine them.
        const fullCode = `${taskPrompt}${code}\n\n${test}\ncheck(${entryPoint})\n`;
        const start = Date.now();

        const proc = spawn('python3', ['-c', fullCode], {
            timeout: timeoutMs,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let stdout = '';
        let stderr = '';
        proc.stdout?.on('data', (d: Buffer) => { stdout += d.toString(); });
        proc.stderr?.on('data', (d: Buffer) => { stderr += d.toString(); });

        proc.on('close', (code) => {
            resolve({
                passed: code === 0,
                output: (stdout + stderr).slice(0, 2000),
                timeMs: Date.now() - start,
            });
        });

        proc.on('error', (err) => {
            resolve({ passed: false, output: String(err), timeMs: Date.now() - start });
        });
    });
}

// ─── Code generation (6 configs) ─────────────────────────────────────

const CODE_PROMPT_PREFIX = `Complete the following Python function. Output ONLY the function body (the implementation lines), no explanations, no function signature, no markdown fences. Just the indented code that goes inside the function.

`;

async function generateSolo(
    agent: Agent, task: HumanEvalTask, timeout: number,
    models: { claude: string; openai: string }
): Promise<{ code: string; duration: number; error?: string }> {
    const start = Date.now();
    const r = await callAgent(agent, CODE_PROMPT_PREFIX + task.prompt, timeout, models);
    return { code: r.error ? '' : extractPythonCode(r.content, task.prompt), duration: Date.now() - start, error: r.error };
}

async function generateLeadConsult(
    leader: Agent, task: HumanEvalTask, maxIter: number, timeout: number,
    models: { claude: string; openai: string }
): Promise<{ code: string; duration: number; iterations: number; error?: string }> {
    const consultant: Agent = leader === 'claude' ? 'openai' : 'claude';
    const start = Date.now();

    const gen = await callAgent(leader, CODE_PROMPT_PREFIX + task.prompt, timeout, models);
    if (gen.error) return { code: '', duration: Date.now() - start, iterations: 1, error: gen.error };
    let code = extractPythonCode(gen.content, task.prompt);

    for (let i = 2; i <= maxIter; i++) {
        const reviewPrompt = `Review this Python function for correctness. If it's correct, respond with CONSENSUS_OK. If not, respond with CONSENSUS_KO and explain the issues.\n\nFunction signature:\n${task.prompt}\n\nImplementation:\n${code}`;
        const review = await callAgent(consultant, reviewPrompt, timeout, models);
        if (review.error || review.content.includes('CONSENSUS_OK')) {
            return { code, duration: Date.now() - start, iterations: i };
        }
        const fixPrompt = `Fix the issues in this Python function:\n\n${review.content.slice(0, 2000)}\n\nOriginal function:\n${task.prompt}\n\nCurrent implementation:\n${code}\n\nOutput ONLY the corrected Python code.`;
        const fix = await callAgent(leader, fixPrompt, timeout, models);
        if (fix.error) break;
        code = extractPythonCode(fix.content, task.prompt);
    }
    return { code, duration: Date.now() - start, iterations: maxIter };
}

async function generateOrchCode(
    orchestrator: Agent, task: HumanEvalTask, maxIter: number, timeout: number,
    models: { claude: string; openai: string }
): Promise<{ code: string; duration: number; iterations: number; error?: string }> {
    const coder: Agent = orchestrator === 'claude' ? 'openai' : 'claude';
    const start = Date.now();

    const planPrompt = `You are a technical architect. For the following Python function, produce a detailed implementation plan: algorithm choice, edge cases to handle, time complexity.\n\n${task.prompt}`;
    const plan = await callAgent(orchestrator, planPrompt, timeout, models);
    if (plan.error) return { code: '', duration: Date.now() - start, iterations: 1, error: plan.error };

    const implPrompt = `Implement this Python function following the plan below. Output ONLY the Python code.\n\nFunction:\n${task.prompt}\n\nPlan:\n${plan.content.slice(0, 3000)}`;
    const impl = await callAgent(coder, implPrompt, timeout, models);
    if (impl.error) return { code: '', duration: Date.now() - start, iterations: 1, error: impl.error };
    let code = extractPythonCode(impl.content, task.prompt);

    for (let i = 2; i <= maxIter; i++) {
        const reviewPrompt = `Review this implementation for correctness. Respond with CONSENSUS_OK if correct, or CONSENSUS_KO with issues.\n\nFunction:\n${task.prompt}\n\nImplementation:\n${code}`;
        const review = await callAgent(orchestrator, reviewPrompt, timeout, models);
        if (review.error || review.content.includes('CONSENSUS_OK')) {
            return { code, duration: Date.now() - start, iterations: i };
        }
        const fixPrompt = `Fix the issues:\n${review.content.slice(0, 2000)}\n\nFunction:\n${task.prompt}\n\nCode:\n${code}\n\nOutput ONLY the corrected Python code.`;
        const fix = await callAgent(coder, fixPrompt, timeout, models);
        if (fix.error) break;
        code = extractPythonCode(fix.content, task.prompt);
    }
    return { code, duration: Date.now() - start, iterations: maxIter };
}

async function generateForConfig(
    config: BenchConfig, task: HumanEvalTask, maxIter: number, timeout: number,
    models: { claude: string; openai: string }
): Promise<{ code: string; duration: number; error?: string }> {
    switch (config) {
        case 'claude-solo': return generateSolo('claude', task, timeout, models);
        case 'openai-solo': return generateSolo('openai', task, timeout, models);
        case 'claude-lead': return generateLeadConsult('claude', task, maxIter, timeout, models);
        case 'openai-lead': return generateLeadConsult('openai', task, maxIter, timeout, models);
        case 'claude-orch': return generateOrchCode('claude', task, maxIter, timeout, models);
        case 'openai-orch': return generateOrchCode('openai', task, maxIter, timeout, models);
    }
}

// ─── 3-Judge evaluation with debate ──────────────────────────────────

// criteria used in judge prompts
// const CRITERIA = ['correctness', 'completeness', 'edgeCases', 'codeQuality', 'readability'];
const DIVERGENCE_THRESHOLD = 0.20; // 20%

interface JudgeCaller {
    name: string;
    call: (prompt: string, timeout: number) => Promise<{ content: string; error?: string }>;
}

function buildEvalPrompt(
    taskPrompt: string,
    configs: { label: string; code: string; passed: boolean }[]
): string {
    let outputsSection = '';
    for (const c of configs) {
        outputsSection += `\n## Output ${c.label} [Test: ${c.passed ? 'PASS' : 'FAIL'}]\n${c.code.slice(0, 2500) || '(empty)'}\n`;
    }

    const jsonShape = configs.map(c =>
        `"${c.label}":{"correctness":N,"completeness":N,"edgeCases":N,"codeQuality":N,"readability":N,"justification":"..."}`
    ).join(',');

    return `You are a code quality evaluator. Score each code output independently for the same task.
Be strict and objective. The test results (PASS/FAIL) are provided — factor them into correctness scoring.

## Task
${taskPrompt}
${outputsSection}
## Scoring criteria (1-10 each):
- correctness: Does the code work correctly? (test result is a strong signal)
- completeness: Are all requirements addressed?
- edgeCases: Are edge cases handled?
- codeQuality: Is it clean, idiomatic, well-structured?
- readability: Are naming, comments, clarity good?

Respond ONLY in this exact JSON format, no other text:
{${jsonShape}}`;
}

function buildDebatePrompt(
    taskPrompt: string,
    allScores: { judge: string; scores: Record<string, QualityScores> }[],
    divergentConfigs: string[]
): string {
    let scoresSection = '';
    for (const js of allScores) {
        scoresSection += `\n### ${js.judge}'s scores:\n`;
        for (const [label, s] of Object.entries(js.scores)) {
            scoresSection += `  ${label}: ${s.total}/10 — "${s.justification}"\n`;
        }
    }

    return `You are a code quality evaluator participating in a calibration debate.
The following outputs were scored by 3 independent judges, but there is significant disagreement (>20% divergence) on: ${divergentConfigs.join(', ')}.

## Task
${taskPrompt}

## All judges' scores and justifications:
${scoresSection}

Review the other judges' reasoning carefully. Then provide your REVISED scores.
If you still disagree with the others, explain WHY with specific code references.

Respond ONLY in this JSON format:
{${divergentConfigs.map(l =>
    `"${l}":{"correctness":N,"completeness":N,"edgeCases":N,"codeQuality":N,"readability":N,"justification":"..."}`
).join(',')}}`;
}

function clamp(n: number): number {
    return Math.max(1, Math.min(10, Math.round(n)));
}

function parseScores(content: string, labels: string[]): Record<string, QualityScores> | null {
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;
    try {
        const parsed = JSON.parse(jsonMatch[0]);
        const result: Record<string, QualityScores> = {};
        for (const label of labels) {
            const obj = parsed[label];
            if (!obj) continue;
            const c = clamp(obj.correctness ?? 0);
            const co = clamp(obj.completeness ?? 0);
            const e = clamp(obj.edgeCases ?? 0);
            const q = clamp(obj.codeQuality ?? 0);
            const rd = clamp(obj.readability ?? 0);
            result[label] = {
                correctness: c, completeness: co, edgeCases: e,
                codeQuality: q, readability: rd,
                total: parseFloat(((c + co + e + q + rd) / 5).toFixed(1)),
                justification: String(obj.justification ?? ''),
            };
        }
        return Object.keys(result).length > 0 ? result : null;
    } catch { return null; }
}

function detectDivergence(
    judgeScores: { judge: string; scores: Record<string, QualityScores> }[],
    labels: string[]
): string[] {
    const divergent: string[] = [];
    for (const label of labels) {
        const totals = judgeScores
            .filter(j => j.scores[label])
            .map(j => j.scores[label].total);
        if (totals.length < 2) continue;
        const median = totals.sort((a, b) => a - b)[Math.floor(totals.length / 2)];
        if (median === 0) continue;
        const maxDev = Math.max(...totals.map(t => Math.abs(t - median) / median));
        if (maxDev > DIVERGENCE_THRESHOLD) {
            divergent.push(label);
        }
    }
    return divergent;
}

function averageScores(
    allScores: Record<string, QualityScores>[],
    labels: string[]
): Record<string, QualityScores> {
    const result: Record<string, QualityScores> = {};
    for (const label of labels) {
        const judges = allScores.filter(s => s[label]).map(s => s[label]);
        if (judges.length === 0) continue;
        const avg = (field: keyof Omit<QualityScores, 'total' | 'justification'>) =>
            parseFloat((judges.reduce((a, j) => a + j[field], 0) / judges.length).toFixed(1));
        const c = avg('correctness'), co = avg('completeness'), e = avg('edgeCases');
        const q = avg('codeQuality'), rd = avg('readability');
        result[label] = {
            correctness: c, completeness: co, edgeCases: e,
            codeQuality: q, readability: rd,
            total: parseFloat(((c + co + e + q + rd) / 5).toFixed(1)),
            justification: judges.map(j => j.justification).join(' | '),
        };
    }
    return result;
}

async function evaluateWithDebate(
    taskPrompt: string,
    outputs: { config: BenchConfig; code: string; passed: boolean }[],
    timeout: number,
    models: { claude: string; openai: string; ollama: string },
): Promise<JudgeAudit> {
    const valid = outputs.filter(o => o.code.length > 0);
    const labels = 'ABCDEF'.split('').slice(0, valid.length);
    const shuffled = shuffle(valid);
    const labelMap = new Map<string, BenchConfig>();
    const configToLabel = new Map<BenchConfig, string>();
    for (let i = 0; i < shuffled.length; i++) {
        labelMap.set(labels[i], shuffled[i].config);
        configToLabel.set(shuffled[i].config, labels[i]);
    }

    const evalPrompt = buildEvalPrompt(
        taskPrompt,
        shuffled.map((s, i) => ({ label: labels[i], code: s.code, passed: s.passed }))
    );

    const judges: JudgeCaller[] = [
        { name: 'Claude', call: (p, t) => callClaude(p, t, models.claude) },
        { name: 'GPT', call: (p, t) => callOpenAI(p, t, models.openai) },
        { name: 'DeepSeek', call: (p, t) => callOllama(p, t, models.ollama) },
    ];

    // ─── Round 1: independent scoring ───
    const round1Results = await Promise.all(
        judges.map(async (judge) => {
            try {
                const r = await judge.call(evalPrompt, timeout);
                if (r.error) return { judge: judge.name, scores: null };
                return { judge: judge.name, scores: parseScores(r.content, labels) };
            } catch { return { judge: judge.name, scores: null }; }
        })
    );

    const round1Valid = round1Results.filter(r => r.scores !== null) as
        { judge: string; scores: Record<string, QualityScores> }[];

    const r1Status = round1Results.map(r => `${r.judge}:${r.scores ? 'OK' : 'FAIL'}`).join(' ');
    process.stdout.write(`[R1 ${r1Status}] `);

    if (round1Valid.length === 0) {
        return {
            round1: [], divergenceDetected: false,
            finalScores: {},
        };
    }

    // Convert label-based scores to config-based for storage
    function labelToConfig(scores: Record<string, QualityScores>): Partial<Record<BenchConfig, QualityScores>> {
        const result: Partial<Record<BenchConfig, QualityScores>> = {};
        for (const [label, s] of Object.entries(scores)) {
            const cfg = labelMap.get(label);
            if (cfg) result[cfg] = s;
        }
        return result;
    }

    const round1Audit: JudgeRound[] = round1Valid.map(r => ({
        judge: r.judge,
        scores: labelToConfig(r.scores),
    }));

    // ─── Check divergence ───
    const divergentLabels = detectDivergence(round1Valid, labels);

    if (divergentLabels.length === 0) {
        // No divergence — average and return
        const averaged = averageScores(round1Valid.map(r => r.scores), labels);
        return {
            round1: round1Audit,
            divergenceDetected: false,
            finalScores: labelToConfig(averaged),
        };
    }

    // ─── Round 2: DEBATE ───
    const divergentConfigNames = divergentLabels.map(l => `${l}(${CONFIG_SHORT[labelMap.get(l)!]})`);
    process.stdout.write(`[DEBATE: ${divergentConfigNames.join(',')}] `);

    const debatePrompt = buildDebatePrompt(taskPrompt, round1Valid, divergentLabels);
    const debateMessages: { judge: string; argument: string }[] = [];

    const round2Results = await Promise.all(
        judges.map(async (judge) => {
            try {
                const r = await judge.call(debatePrompt, timeout);
                if (r.error) return { judge: judge.name, scores: null, argument: '' };
                const scores = parseScores(r.content, divergentLabels);
                // Extract justification as the debate argument
                const argument = scores
                    ? Object.values(scores).map(s => s.justification).join('; ')
                    : r.content.slice(0, 500);
                debateMessages.push({ judge: judge.name, argument });
                return { judge: judge.name, scores };
            } catch { return { judge: judge.name, scores: null, argument: '' }; }
        })
    );

    const round2Valid = round2Results.filter(r => r.scores !== null) as
        { judge: string; scores: Record<string, QualityScores> }[];

    const r2Status = round2Results.map(r => `${r.judge}:${r.scores ? 'OK' : 'FAIL'}`).join(' ');
    process.stdout.write(`[R2 ${r2Status}] `);

    // Merge: use round2 scores for divergent labels, round1 for the rest
    const mergedScores = round1Valid.map((r1) => {
        const r2 = round2Valid.find(r => r.judge === r1.judge);
        const merged: Record<string, QualityScores> = { ...r1.scores };
        if (r2) {
            for (const label of divergentLabels) {
                if (r2.scores[label]) merged[label] = r2.scores[label];
            }
        }
        return merged;
    });

    const finalAveraged = averageScores(mergedScores, labels);

    const round2Audit: JudgeRound[] = round2Valid.map(r => ({
        judge: r.judge,
        scores: labelToConfig(r.scores),
    }));

    return {
        round1: round1Audit,
        divergenceDetected: true,
        debate: {
            trigger: divergentConfigNames.join(', '),
            messages: debateMessages,
        },
        round2: round2Audit,
        finalScores: labelToConfig(finalAveraged),
    };
}

// ─── Helpers ─────────────────────────────────────────────────────────

function shuffle<T>(arr: readonly T[]): T[] {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// ─── Checkpoint management ───────────────────────────────────────────

const CHECKPOINT_PATH = path.join(process.cwd(), 'benchmark-results', 'checkpoint_v2.json');

function makeFingerprint(opts: Args, taskCount: number): string {
    // Simple fingerprint: configs + models + runs + task count
    return JSON.stringify({
        configs: opts.configs.sort(),
        claudeModel: opts.claudeModel,
        openaiModel: opts.openaiModel,
        ollamaModel: opts.ollamaModel,
        runs: opts.runs,
        taskCount,
    });
}

function saveCheckpoint(
    fingerprint: string, results: TaskResult[], debatesTriggered: number
): void {
    const data: CheckpointData = {
        fingerprint,
        completedKeys: results.map(r => `${r.run}:${r.taskId}`),
        results,
        debatesTriggered,
        lastSaved: new Date().toISOString(),
    };
    fs.mkdirSync(path.dirname(CHECKPOINT_PATH), { recursive: true });
    // Write to temp file then rename for atomicity
    const tmpPath = CHECKPOINT_PATH + '.tmp';
    fs.writeFileSync(tmpPath, JSON.stringify(data));
    fs.renameSync(tmpPath, CHECKPOINT_PATH);
}

function loadCheckpoint(fingerprint: string): CheckpointData | null {
    if (!fs.existsSync(CHECKPOINT_PATH)) return null;
    try {
        const raw = fs.readFileSync(CHECKPOINT_PATH, 'utf-8');
        const data: CheckpointData = JSON.parse(raw);
        if (data.fingerprint !== fingerprint) {
            console.log('  ⚠ Checkpoint found but config mismatch — starting fresh.');
            return null;
        }
        return data;
    } catch {
        console.log('  ⚠ Corrupt checkpoint file — starting fresh.');
        return null;
    }
}

function deleteCheckpoint(): void {
    try { fs.unlinkSync(CHECKPOINT_PATH); } catch { /* ignore */ }
}


// ─── Main ────────────────────────────────────────────────────────────

interface Args {
    configs: BenchConfig[];
    limit: number;
    runs: number;
    maxIter: number;
    timeout: number;
    claudeModel: string;
    openaiModel: string;
    ollamaModel: string;
    tasks: string[];
    resume: boolean;
}

function parseArgs(): Args {
    const args = process.argv.slice(2);
    let configs: BenchConfig[] = [];
    let limit = 164;
    let runs = 5;
    let maxIter = 2;
    let timeout = 300_000;
    let claudeModel = 'claude-opus-4-6';
    let openaiModel = 'gpt-5.1';
    let ollamaModel = 'deepseek-v3.1:671b-cloud';
    let tasks: string[] = [];
    let resume = false;

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--configs' && args[i + 1]) configs = args[++i].split(',') as BenchConfig[];
        else if (args[i] === '--limit' && args[i + 1]) limit = parseInt(args[++i], 10);
        else if (args[i] === '--runs' && args[i + 1]) runs = parseInt(args[++i], 10);
        else if (args[i] === '--max-iter' && args[i + 1]) maxIter = parseInt(args[++i], 10);
        else if (args[i] === '--timeout' && args[i + 1]) timeout = parseInt(args[++i], 10);
        else if (args[i] === '--claude-model' && args[i + 1]) claudeModel = args[++i];
        else if (args[i] === '--openai-model' && args[i + 1]) openaiModel = args[++i];
        else if (args[i] === '--ollama-model' && args[i + 1]) ollamaModel = args[++i];
        else if (args[i] === '--tasks' && args[i + 1]) tasks = args[++i].split(',');
        else if (args[i] === '--resume') resume = true;
        else if (args[i] === '--help') {
            console.log(`CRISTAL CODE Benchmark v2 — HumanEval + Execution + Judge Debate\n`);
            console.log(`Options:`);
            console.log(`  --configs c1,c2           Select configs (default: all 6)`);
            console.log(`  --limit N                 Max tasks to run (default: 164 = all)`);
            console.log(`  --runs N                  Number of runs (default: 5)`);
            console.log(`  --tasks HumanEval/0,...    Specific task IDs`);
            console.log(`  --max-iter N              Max debate iterations (default: 2)`);
            console.log(`  --timeout N               Timeout per call in ms (default: 300000)`);
            console.log(`  --claude-model MODEL      (default: claude-opus-4-6)`);
            console.log(`  --openai-model MODEL      (default: gpt-5.1)`);
            console.log(`  --ollama-model MODEL      (default: deepseek-v3.1:671b-cloud)`);
            console.log(`  --resume                  Resume from last checkpoint if available`);
            process.exit(0);
        }
    }

    if (configs.length === 0) configs = [...ALL_CONFIGS];
    return { configs, limit, runs, maxIter, timeout, claudeModel, openaiModel, ollamaModel, tasks, resume };
}

async function main(): Promise<void> {
    const opts = parseArgs();
    const models = { claude: opts.claudeModel, openai: opts.openaiModel, ollama: opts.ollamaModel };

    // Load HumanEval
    const dataPath = path.join(__dirname, '..', 'data', 'HumanEval.jsonl');
    if (!fs.existsSync(dataPath)) {
        console.error(`ERROR: HumanEval data not found at ${dataPath}`);
        console.error(`Run: curl -sL 'https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz' | gunzip > data/HumanEval.jsonl`);
        process.exit(1);
    }

    let allTasks = loadHumanEval(dataPath);
    if (opts.tasks.length > 0) {
        allTasks = allTasks.filter(t => opts.tasks.includes(t.task_id));
    }
    const selectedTasks = allTasks.slice(0, opts.limit);

    console.log('╔══════════════════════════════════════════════════════════════════════╗');
    console.log('║  CRISTAL CODE — Benchmark v2                                        ║');
    console.log('║  HumanEval + Code Execution + 3-Judge Debate                        ║');
    console.log('╚══════════════════════════════════════════════════════════════════════╝\n');

    const hasAnthropic = !!process.env.ANTHROPIC_API_KEY;
    const hasOpenAI = !!process.env.OPENAI_API_KEY;

    console.log(`  Claude      : ${opts.claudeModel} ${hasAnthropic ? '(key set)' : '(MISSING)'}`);
    console.log(`  OpenAI      : ${opts.openaiModel} ${hasOpenAI ? '(key set)' : '(MISSING)'}`);
    console.log(`  Ollama judge: ${opts.ollamaModel} (${OLLAMA_HOST}:${OLLAMA_PORT})`);
    console.log(`  Judges      : Claude + GPT + DeepSeek (debate on >20% divergence)`);
    console.log(`  Configs     : ${opts.configs.map(c => CONFIG_SHORT[c]).join(', ')}`);
    console.log(`  Tasks       : ${selectedTasks.length} / ${allTasks.length} HumanEval problems`);
    console.log(`  Runs        : ${opts.runs}`);
    console.log(`  Max iters   : ${opts.maxIter}`);
    console.log(`  Timeout     : ${opts.timeout}ms\n`);
    console.log(`  Total work  : ${selectedTasks.length} × ${opts.configs.length} configs × ${opts.runs} runs = ${selectedTasks.length * opts.configs.length * opts.runs} generations + executions\n`);

    if (!hasAnthropic) { console.error('ERROR: Set ANTHROPIC_API_KEY.'); process.exit(1); }
    if (!hasOpenAI) { console.error('ERROR: Set OPENAI_API_KEY.'); process.exit(1); }

    // ─── Checkpoint / resume ─────────────────────────────────────
    const fingerprint = makeFingerprint(opts, selectedTasks.length);
    let allResults: TaskResult[] = [];
    let debatesTriggered = 0;
    const completedKeys = new Set<string>();

    if (opts.resume) {
        const checkpoint = loadCheckpoint(fingerprint);
        if (checkpoint) {
            allResults = checkpoint.results;
            debatesTriggered = checkpoint.debatesTriggered;
            for (const key of checkpoint.completedKeys) completedKeys.add(key);
            console.log(`  ✓ Resumed from checkpoint: ${completedKeys.size} tasks already done (saved ${checkpoint.lastSaved})`);
            console.log(`  Remaining: ${selectedTasks.length * opts.runs - completedKeys.size} tasks\n`);
        } else {
            console.log('  No valid checkpoint found — starting fresh.\n');
        }
    }

    console.log('═'.repeat(90));

    for (let run = 1; run <= opts.runs; run++) {
        console.log(`\n${'█'.repeat(90)}`);
        console.log(`█  RUN ${run}/${opts.runs}`);
        console.log(`${'█'.repeat(90)}\n`);

        for (let ti = 0; ti < selectedTasks.length; ti++) {
            const task = selectedTasks[ti];
            const taskKey = `${run}:${task.task_id}`;

            // Skip already-completed tasks (checkpoint resume)
            if (completedKeys.has(taskKey)) {
                console.log(`  ⏭ [${ti + 1}/${selectedTasks.length}] ${task.task_id} — already done (checkpoint)`);
                continue;
            }

            console.log(`\n▶ [${ti + 1}/${selectedTasks.length}] ${task.task_id} — ${task.entry_point}`);
            console.log(`  ${task.prompt.split('\n').find(l => l.trim().startsWith('\"\"\"'))?.trim().slice(0, 70) || ''}...\n`);

            const configResults: ConfigRunResult[] = [];

            for (const cfg of opts.configs) {
                process.stdout.write(`  ${CONFIG_SHORT[cfg].padEnd(8)} `);

                // Generate code
                const gen = await generateForConfig(cfg, task, opts.maxIter, opts.timeout, { claude: opts.claudeModel, openai: opts.openaiModel });

                if (gen.error) {
                    console.log(`ERROR: ${gen.error.slice(0, 50)}`);
                    configResults.push({
                        config: cfg, generatedCode: '', passed: false,
                        execOutput: '', execTimeMs: 0, duration: gen.duration, error: gen.error,
                    });
                    continue;
                }

                // Execute code + tests
                const exec = await executeCode(task.prompt, gen.code, task.test, task.entry_point);

                console.log(`${gen.duration}ms, ${exec.passed ? 'PASS ✓' : 'FAIL ✗'} (exec: ${exec.timeMs}ms)`);

                configResults.push({
                    config: cfg, generatedCode: gen.code, passed: exec.passed,
                    execOutput: exec.output, execTimeMs: exec.timeMs, duration: gen.duration,
                });
            }

            // Judge evaluation with debate
            const withCode = configResults.filter(c => c.generatedCode.length > 0);
            let judgeAudit: JudgeAudit;

            if (withCode.length > 0) {
                process.stdout.write('  Judges  ');
                judgeAudit = await evaluateWithDebate(
                    task.prompt,
                    withCode.map(c => ({ config: c.config, code: c.generatedCode, passed: c.passed })),
                    opts.timeout,
                    models,
                );

                if (judgeAudit.divergenceDetected) debatesTriggered++;

                // Print final scores
                const scores = opts.configs
                    .filter(c => judgeAudit.finalScores[c])
                    .map(c => `${CONFIG_SHORT[c]}=${judgeAudit.finalScores[c]!.total}/10`);
                console.log(scores.join('  ') + (judgeAudit.divergenceDetected ? ' [DEBATED]' : ''));
            } else {
                judgeAudit = { round1: [], divergenceDetected: false, finalScores: {} };
                console.log('  Judges  SKIPPED (no code generated)');
            }

            allResults.push({
                taskId: task.task_id,
                entryPoint: task.entry_point,
                run,
                timestamp: new Date().toISOString(),
                configs: configResults,
                judgeAudit,
            });

            // Save checkpoint after each task
            completedKeys.add(taskKey);
            saveCheckpoint(fingerprint, allResults, debatesTriggered);

            console.log('─'.repeat(70));
        }
    }

    // ─── Summary ─────────────────────────────────────────────────

    console.log('\n╔══════════════════════════════════════════════════════════════════════════════════╗');
    console.log('║                              PASS@1 RESULTS                                     ║');
    console.log('╚══════════════════════════════════════════════════════════════════════════════════╝\n');

    const passAt1: Partial<Record<BenchConfig, number>> = {};
    for (const cfg of opts.configs) {
        const runs = allResults.filter(r => r.configs.find(c => c.config === cfg));
        const passed = runs.filter(r => r.configs.find(c => c.config === cfg && c.passed));
        if (runs.length > 0) {
            passAt1[cfg] = parseFloat((passed.length / runs.length * 100).toFixed(1));
            console.log(`  ${CONFIG_SHORT[cfg].padEnd(10)} ${passAt1[cfg]}% (${passed.length}/${runs.length})`);
        }
    }

    console.log('\n╔══════════════════════════════════════════════════════════════════════════════════╗');
    console.log('║                   QUALITY (3 judges with debate on divergence)                   ║');
    console.log('╚══════════════════════════════════════════════════════════════════════════════════╝\n');

    const avgQuality: Partial<Record<BenchConfig, number>> = {};
    for (const cfg of opts.configs) {
        const scores = allResults
            .filter(r => r.judgeAudit.finalScores[cfg])
            .map(r => r.judgeAudit.finalScores[cfg]!.total);
        if (scores.length > 0) {
            avgQuality[cfg] = parseFloat((scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1));
            console.log(`  ${CONFIG_SHORT[cfg].padEnd(10)} ${avgQuality[cfg]}/10 avg (n=${scores.length})`);
        }
    }

    console.log(`\n  Judge debates triggered: ${debatesTriggered}/${allResults.length} evaluations (${(debatesTriggered / Math.max(1, allResults.length) * 100).toFixed(1)}%)`);

    // ─── Save report ─────────────────────────────────────────────

    const report: BenchV2Report = {
        meta: {
            date: new Date().toISOString(),
            claudeModel: opts.claudeModel,
            openaiModel: opts.openaiModel,
            ollamaModel: opts.ollamaModel,
            totalTasks: allTasks.length,
            selectedTasks: selectedTasks.length,
            configs: opts.configs,
            runs: opts.runs,
            divergenceThreshold: DIVERGENCE_THRESHOLD,
        },
        tasks: allResults.map(r => ({
            ...r,
            configs: r.configs.map(c => ({
                ...c,
                generatedCode: c.generatedCode.slice(0, 5000),
                execOutput: c.execOutput.slice(0, 2000),
            })),
        })),
        summary: {
            passAt1, avgQuality,
            debatesTriggered,
            totalEvaluations: allResults.length,
        },
    };

    const resultsDir = path.join(process.cwd(), 'benchmark-results');
    fs.mkdirSync(resultsDir, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const reportPath = path.join(resultsDir, `bench_v2_${ts}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n  Report saved to: ${reportPath}`);

    // Clean up checkpoint — benchmark completed successfully
    deleteCheckpoint();
    console.log(`  Checkpoint cleared (benchmark complete).\n`);
}

main().catch(err => {
    console.error('FATAL:', err);
    process.exit(1);
});
