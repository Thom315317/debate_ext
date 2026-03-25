#!/usr/bin/env node
/**
 * DEBATE EXT — Paper-Grade Benchmark (MBPP+ / EvalPlus)
 *
 * Generators (mid-tier, via Ollama):
 *   gen1 = Qwen3-Coder 480B    gen2 = DeepSeek-V3.1 671B
 *
 * Judges (frontier, neutral — never generate):
 *   Claude Sonnet 4.5 + GPT-4.1 (primary)
 *   Gemini 2.5 Pro (tie-breaker via Google AI)
 *
 * Paper metrics:
 *   - pass@k (Chen et al. 2021 unbiased estimator)
 *   - Inter-judge: Spearman rank correlation + Cohen's kappa
 *   - Bug-catching: recall, FPR, AUC (via risk_fail_prob)
 *   - Escalation audit: debates triggered, tie-breaker stats, score shift
 *   - Token/cost tracking per provider
 *
 * Judge panel (3-level):
 *   R1: 2 judges score independently
 *   R2: if |delta| > τ=2.0 or risk disagreement → judges debate + re-score
 *   TB: if still divergent → Gemini tie-breaker (explicit error handling)
 *
 * Usage:
 *   npm run bench:mbppplus                              # 30 tasks, 8 configs, 3 runs
 *   npm run bench:mbppplus -- --dry-run --limit 5       # validate pipeline, zero API calls
 *   npm run bench:mbppplus -- --limit 10 --runs 1 --configs gen1-solo,gen2-solo
 *
 * Requires: ANTHROPIC_API_KEY + OPENAI_API_KEY + GEMINI_API_KEY env vars + Ollama.
 * Data: python scripts/setup_mbppplus.py → data/MbppPlus.jsonl
 */

import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import * as http from 'http';
import { spawn } from 'child_process';

// ═══════════════════════════════════════════════════════════════════════
// A. Types & Interfaces
// ═══════════════════════════════════════════════════════════════════════

type BenchConfig =
    | 'gen1-solo' | 'gen2-solo'
    | 'gen1-lead' | 'gen2-lead'
    | 'gen1-orch' | 'gen2-orch'
    | 'gen1-selfrefine' | 'gen2-selfrefine'
    | 'gen1-ximprove' | 'gen2-ximprove'
    | 'gen1-xbest' | 'gen2-xbest'
    | 'gen1-xfusion' | 'gen2-xfusion';

type Agent = 'gen1' | 'gen2';

type ExecStatus = 'pass' | 'fail' | 'eval_error';

const ALL_CONFIGS: BenchConfig[] = [
    'gen1-solo', 'gen2-solo',
    'gen1-lead', 'gen2-lead',
    'gen1-orch', 'gen2-orch',
    'gen1-selfrefine', 'gen2-selfrefine',
    'gen1-ximprove', 'gen2-ximprove',
    'gen1-xbest', 'gen2-xbest',
    'gen1-xfusion', 'gen2-xfusion',
];

const CONFIG_SHORT: Record<BenchConfig, string> = {
    'gen1-solo': 'QC.Solo', 'gen2-solo': 'DS.Solo',
    'gen1-lead': 'QC.Lead', 'gen2-lead': 'DS.Lead',
    'gen1-orch': 'QC.Orch', 'gen2-orch': 'DS.Orch',
    'gen1-selfrefine': 'QC.SRef', 'gen2-selfrefine': 'DS.SRef',
    'gen1-ximprove': 'QC.XImp', 'gen2-ximprove': 'DS.XImp',
    'gen1-xbest': 'QC.XBst', 'gen2-xbest': 'DS.XBst',
    'gen1-xfusion': 'QC.XFus', 'gen2-xfusion': 'DS.XFus',
};

interface MbppPlusTask {
    task_id: string;
    prompt: string;
    code: string;            // canonical solution
    test_list: string[];
    test_setup_code: string;
    entry_point: string;
    test_list_plus: string[];
}

interface QualityScores {
    correctness: number;
    completeness: number;
    edgeCases: number;
    codeQuality: number;
    readability: number;
    total: number;
    justification: string;
    risk_fail_prob: number;  // 0.0–1.0: probability hidden tests fail
}

interface JudgeRound {
    judge: string;
    round: 'R1' | 'R2' | 'TB';
    scores: Partial<Record<BenchConfig, QualityScores>>;
}

interface DebateRound {
    trigger: string;
    triggerType: 'score_divergence' | 'risk_disagreement' | 'both';
    messages: { judge: string; argument: string }[];
}

interface JudgeAudit {
    round1: JudgeRound[];
    divergenceDetected: boolean;
    divergentConfigs?: string[];
    debate?: DebateRound;
    round2?: JudgeRound[];
    tiebreakerUsed: boolean;
    tiebreakerError?: string;
    j0: Partial<Record<BenchConfig, QualityScores>>;  // R1 average (baseline)
    j1: Partial<Record<BenchConfig, QualityScores>>;  // post-debate
    j2: Partial<Record<BenchConfig, QualityScores>>;  // final (includes TB)
    finalScores: Partial<Record<BenchConfig, QualityScores>>;
    binaryVerdicts?: Partial<Record<BenchConfig, { pass: boolean; confidence: number; reason: string }[]>>;
}

interface TokenUsage {
    provider: string;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
}

interface ConfigRunResult {
    config: BenchConfig;
    generatedCode: string;
    execStatus: ExecStatus;
    execOutput: string;
    execTimeMs: number;
    duration: number;
    apiCallCount: number;
    execStatusPlus: ExecStatus;
    execOutputPlus: string;
    execTimeMsPlus: number;
    error?: string;
    tokens?: TokenUsage;
}

interface TaskResult {
    taskId: string;
    entryPoint: string;
    taskPrompt: string;
    run: number;
    timestamp: string;
    configs: ConfigRunResult[];
    judgeAudit: JudgeAudit;
}

interface PaperMetrics {
    passAtK: Record<string, Partial<Record<BenchConfig, number>>>;
    interJudge: { spearmanR1: number | null; cohensKappaR1: number | null };
    bugCatching: { recall: number | null; fpr: number | null; auc: number | null };
    escalation: {
        debatesTriggered: number;
        totalEvaluations: number;
        tiebreakerUsed: number;
        tiebreakerErrors: number;
        avgScoreShift: number | null;
        triggerBreakdown: { scoreDivergence: number; riskDisagreement: number; both: number };
    };
    variance: Partial<Record<BenchConfig, { mean: number; stdDev: number; n: number }>>;
    mcnemar: { pair: string; b: number; c: number; chi2: number; pLevel: string }[];
}

interface PaperReport {
    meta: {
        date: string;
        dataset: string;
        gen1Model: string;
        gen2Model: string;
        claudeJudgeModel: string;
        openaiJudgeModel: string;
        tiebreakerModel: string;
        seed: number;
        tau: number;
        totalTasks: number;
        selectedTasks: number;
        configs: BenchConfig[];
        runs: number;
        judgeBlind: boolean;
        noDebate: boolean;
        noJudge?: boolean;
        judgeStrategy?: 'scoring' | 'binary';
        judgeProvider?: 'api' | 'ollama';
        judge1Model?: string;
        judge2Model?: string;
        rejudgedFrom?: string;
        mergedFrom?: string[];
    };
    tasks: TaskResult[];
    summary: {
        passAt1: Partial<Record<BenchConfig, number>>;
        passAt1Plus: Partial<Record<BenchConfig, number>>;
        avgQuality: Partial<Record<BenchConfig, number>>;
        costEfficiency: Partial<Record<BenchConfig, { avgCalls: number; qualityPerCall: number; passPerCall: number }>>;
        paperMetrics: PaperMetrics;
        tokenUsage: Record<string, { promptTokens: number; completionTokens: number; totalTokens: number }>;
    };
}

// ═══════════════════════════════════════════════════════════════════════
// B. API Callers — Token-Tracking Wrappers
// ═══════════════════════════════════════════════════════════════════════

interface ApiResult {
    content: string;
    error?: string;
    tokens?: TokenUsage;
}

const tokenLog: TokenUsage[] = [];

function trackTokens(t: TokenUsage | undefined): void {
    if (t && t.totalTokens > 0) tokenLog.push(t);
}

// NOTE: temperature=0.2 set explicitly (v1.x bug: was absent, using API default 1.0)
function callClaude(prompt: string, timeoutMs: number, model: string): Promise<ApiResult> {
    const apiKey = process.env.ANTHROPIC_API_KEY!;
    return new Promise((resolve) => {
        const body = JSON.stringify({
            model, max_tokens: 16384, temperature: 0.2,
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
                    if (json.error) {
                        resolve({ content: '', error: json.error.message || JSON.stringify(json.error) });
                    } else {
                        const content = json.content?.filter((b: { type: string }) => b.type === 'text')
                            .map((b: { text: string }) => b.text).join('') ?? '';
                        const tokens: TokenUsage | undefined = json.usage ? {
                            provider: 'anthropic',
                            promptTokens: json.usage.input_tokens ?? 0,
                            completionTokens: json.usage.output_tokens ?? 0,
                            totalTokens: (json.usage.input_tokens ?? 0) + (json.usage.output_tokens ?? 0),
                        } : undefined;
                        trackTokens(tokens);
                        resolve({ content, tokens });
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

function callOpenAI(prompt: string, timeoutMs: number, model: string): Promise<ApiResult> {
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
                    if (json.error) {
                        resolve({ content: '', error: json.error.message || JSON.stringify(json.error) });
                    } else {
                        const content = json.choices?.[0]?.message?.content ?? '';
                        const tokens: TokenUsage | undefined = json.usage ? {
                            provider: 'openai',
                            promptTokens: json.usage.prompt_tokens ?? 0,
                            completionTokens: json.usage.completion_tokens ?? 0,
                            totalTokens: (json.usage.prompt_tokens ?? 0) + (json.usage.completion_tokens ?? 0),
                        } : undefined;
                        trackTokens(tokens);
                        resolve({ content, tokens });
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

const OLLAMA_HOST = process.env.OLLAMA_HOST ?? 'host.docker.internal';
const OLLAMA_PORT = parseInt(process.env.OLLAMA_PORT ?? '11434', 10);

function callOllama(prompt: string, timeoutMs: number, model: string, agentColor: string = ''): Promise<ApiResult> {
    return new Promise((resolve) => {
        const body = JSON.stringify({
            model, messages: [{ role: 'user', content: prompt }], temperature: 0.2, max_tokens: 4096, stream: true,
        });
        let fullContent = '';
        let sseBuffer = '';
        let done = false;
        let state: 'init' | 'think' | 'code' | 'normal' = 'init';
        let usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } | null = null;
        const R = '\x1b[0m';
        const brightColor = agentColor === '\x1b[36m' ? '\x1b[96m' : agentColor === '\x1b[35m' ? '\x1b[95m' : agentColor;

        const finish = (result: ApiResult) => {
            if (done) return;
            done = true;
            clearTimeout(timer);
            if (agentColor) process.stdout.write(R);
            if (fullContent.length > 0) process.stdout.write('\n');
            resolve(result);
        };

        const buildTokens = (): TokenUsage => {
            const estPrompt = Math.ceil(prompt.length / 4);
            const estCompletion = Math.ceil(fullContent.length / 4);
            return {
                provider: 'ollama',
                promptTokens: usage?.prompt_tokens ?? estPrompt,
                completionTokens: usage?.completion_tokens ?? estCompletion,
                totalTokens: (usage?.prompt_tokens ?? estPrompt) + (usage?.completion_tokens ?? estCompletion),
            };
        };

        const req = http.request({
            hostname: OLLAMA_HOST, port: OLLAMA_PORT, path: '/v1/chat/completions', method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        }, (res) => {
            if (res.statusCode !== 200) {
                let errData = '';
                res.on('data', (chunk: Buffer) => { errData += chunk.toString(); });
                res.on('end', () => finish({ content: '', error: `HTTP ${res.statusCode}: ${errData.slice(0, 200)}` }));
                return;
            }
            res.on('data', (chunk: Buffer) => {
                if (done) return;
                sseBuffer += chunk.toString();
                const lines = sseBuffer.split('\n');
                sseBuffer = lines.pop() ?? '';
                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith('data: ')) continue;
                    const payload = trimmed.slice(6);
                    if (payload === '[DONE]') {
                        const tokens = buildTokens();
                        trackTokens(tokens);
                        finish({ content: fullContent, tokens });
                        req.destroy();
                        return;
                    }
                    try {
                        const json = JSON.parse(payload);
                        if (json.usage) usage = json.usage;
                        const fr = json.choices?.[0]?.finish_reason;
                        if (fr === 'stop' || fr === 'length') {
                            const tokens = buildTokens();
                            trackTokens(tokens);
                            finish({ content: fullContent, tokens });
                            req.destroy();
                            return;
                        }
                        const delta = json.choices?.[0]?.delta?.content ?? '';
                        if (delta) {
                            fullContent += delta;
                            const lo = Math.max(fullContent.lastIndexOf('<think>'), fullContent.lastIndexOf('<thinking>'));
                            const lc = Math.max(fullContent.lastIndexOf('</think>'), fullContent.lastIndexOf('</thinking>'));
                            const inThink = lo >= 0 && lo > lc;
                            const tripleCount = (fullContent.match(/```/g) || []).length;
                            const inCode = !inThink && tripleCount % 2 === 1;
                            const prev = state;
                            state = inThink ? 'think' : inCode ? 'code' : 'normal';
                            if (state !== prev && agentColor) {
                                process.stdout.write(R);
                                if (state === 'think') process.stdout.write('\x1b[2m' + agentColor);
                                else if (state === 'code') process.stdout.write('\x1b[1m' + brightColor);
                                else process.stdout.write(agentColor);
                            }
                            process.stdout.write(delta);
                        }
                    } catch { /* partial SSE chunk */ }
                }
            });
            res.on('end', () => {
                // Flush any remaining buffer
                if (sseBuffer.trim().length > 0) {
                    const trimmed = sseBuffer.trim();
                    if (trimmed.startsWith('data: ') && trimmed.slice(6) !== '[DONE]') {
                        try {
                            const json = JSON.parse(trimmed.slice(6));
                            const delta = json.choices?.[0]?.delta?.content ?? '';
                            if (delta) fullContent += delta;
                        } catch {}
                    }
                }
                const tokens = buildTokens();
                trackTokens(tokens);
                finish({ content: fullContent, tokens });
            });
        });
        req.on('error', (err) => finish({ content: '', error: String(err) }));
        const timer = setTimeout(() => { req.destroy(); finish({ content: '', error: `Timeout ${timeoutMs}ms` }); }, timeoutMs);
        req.write(body); req.end();
    });
}

// Non-streaming Ollama call for judge evaluation (no ANSI output)
function callOllamaJudge(prompt: string, timeoutMs: number, model: string, maxTokens: number = 16384): Promise<ApiResult> {
    return new Promise((resolve) => {
        const body = JSON.stringify({
            model, messages: [{ role: 'user', content: prompt }], temperature: 0.2, max_tokens: maxTokens, stream: false, think: false,
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
                    if (json.error) {
                        resolve({ content: '', error: String(json.error) });
                        return;
                    }
                    const content = json.choices?.[0]?.message?.content ?? '';
                    const tokens: TokenUsage = {
                        provider: 'ollama',
                        promptTokens: json.usage?.prompt_tokens ?? Math.ceil(prompt.length / 4),
                        completionTokens: json.usage?.completion_tokens ?? Math.ceil(content.length / 4),
                        totalTokens: (json.usage?.prompt_tokens ?? 0) + (json.usage?.completion_tokens ?? 0),
                    };
                    trackTokens(tokens);
                    resolve({ content, tokens });
                } catch { resolve({ content: '', error: `Invalid JSON: ${data.slice(0, 200)}` }); }
            });
        });
        req.on('error', (err) => resolve({ content: '', error: String(err) }));
        const timer = setTimeout(() => { req.destroy(); resolve({ content: '', error: `Timeout ${timeoutMs}ms` }); }, timeoutMs);
        req.on('close', () => clearTimeout(timer));
        req.write(body); req.end();
    });
}

function callGemini(prompt: string, timeoutMs: number, model: string): Promise<ApiResult> {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) return Promise.resolve({ content: '', error: 'GEMINI_API_KEY not set' });
    return new Promise((resolve) => {
        const body = JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.2 },
        });
        const req = https.request({
            hostname: 'generativelanguage.googleapis.com',
            path: `/v1beta/models/${model}:generateContent?key=${apiKey}`,
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        }, (res) => {
            let data = '';
            res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    if (json.error) {
                        resolve({ content: '', error: json.error.message || JSON.stringify(json.error) });
                    } else {
                        const content = json.candidates?.[0]?.content?.parts
                            ?.map((p: { text?: string }) => p.text ?? '').join('') ?? '';
                        const meta = json.usageMetadata;
                        const tokens: TokenUsage | undefined = meta ? {
                            provider: 'gemini',
                            promptTokens: meta.promptTokenCount ?? 0,
                            completionTokens: meta.candidatesTokenCount ?? 0,
                            totalTokens: (meta.promptTokenCount ?? 0) + (meta.candidatesTokenCount ?? 0),
                        } : undefined;
                        trackTokens(tokens);
                        resolve({ content, tokens });
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

function callAgent(
    agent: Agent, prompt: string, timeout: number,
    models: { gen1: string; gen2: string }
): Promise<ApiResult> {
    const color = agent === 'gen1' ? '\x1b[36m' : '\x1b[35m';
    return agent === 'gen1'
        ? callOllama(prompt, timeout, models.gen1, color)
        : callOllama(prompt, timeout, models.gen2, color);
}

// ═══════════════════════════════════════════════════════════════════════
// C. MBPP+ Loader
// ═══════════════════════════════════════════════════════════════════════

function loadMbppPlus(dataPath: string): MbppPlusTask[] {
    const content = fs.readFileSync(dataPath, 'utf-8');
    return content.trim().split('\n').map(line => {
        const obj = JSON.parse(line);
        return {
            task_id: obj.task_id,
            prompt: obj.prompt,
            code: obj.code ?? obj.canonical_solution ?? '',
            test_list: Array.isArray(obj.test_list) ? obj.test_list : [],
            test_setup_code: obj.test_setup_code ?? '',
            entry_point: obj.entry_point ?? 'unknown',
            test_list_plus: Array.isArray(obj.test_list_plus) ? obj.test_list_plus : [],
        };
    });
}

// ═══════════════════════════════════════════════════════════════════════
// D. Code Extraction
// ═══════════════════════════════════════════════════════════════════════

function extractPythonCode(response: string): string {
    let code = response;

    // Extract from ```python / ```py / ``` blocks
    const pyBlocks = code.match(/```(?:python|py)?\s*\n([\s\S]*?)```/g);
    if (pyBlocks) {
        code = pyBlocks.map(b => b.replace(/```(?:python|py)?\s*\n/, '').replace(/```$/, '')).join('\n');
        return code.trim();
    }

    // Fallback: extract Python functions from raw text (no markdown fences)
    const lines = code.split('\n');
    let inFunction = false;
    const extracted: string[] = [];
    const imports: string[] = [];
    const pending: string[] = []; // buffer blank lines — only flush if followed by valid content

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        if (!inFunction && /^(import\s|from\s)/.test(line)) {
            imports.push(line);
        } else if (!inFunction && (/^def\s+\w+/.test(line) || /^@\w+/.test(line))) {
            inFunction = true;
            extracted.push(line);
        } else if (inFunction) {
            if (/^\s+/.test(line)) {
                // Indented line — part of current function body; flush pending blanks
                extracted.push(...pending, line);
                pending.length = 0;
            } else if (line.trim() === '') {
                // Blank line — buffer it; only include if followed by def/class/import/decorator
                pending.push(line);
            } else if (/^(def\s+\w+|class\s+\w+|@\w+)/.test(line)) {
                // New top-level def / class / decorator — flush pending, continue
                extracted.push(...pending, line);
                pending.length = 0;
            } else if (/^(import\s|from\s)/.test(line)) {
                pending.length = 0;
                imports.push(line);
            } else {
                // Orphan code at indent 0 (assertions, prints, bare expressions) — stop
                pending.length = 0;
                break;
            }
        }
    }

    if (extracted.length > 0) {
        code = [...imports, ...extracted].join('\n');
    }

    return code.trim();
}

// ═══════════════════════════════════════════════════════════════════════
// E. Code Execution (tri-state: pass / fail / eval_error)
// ═══════════════════════════════════════════════════════════════════════

function executeCode(
    code: string, testList: string[], testSetup: string, timeoutMs: number = 10_000
): Promise<{ status: ExecStatus; output: string; timeMs: number }> {
    return new Promise((resolve) => {
        const fullCode = [
            testSetup,
            code,
            '',
            ...testList,
        ].filter(Boolean).join('\n');

        const start = Date.now();

        let proc;
        try {
            proc = spawn('python3', ['-c', fullCode], {
                timeout: timeoutMs,
                stdio: ['ignore', 'pipe', 'pipe'],
            });
        } catch (spawnErr) {
            // E2BIG: argument list too long — code + tests exceed OS limit
            resolve({ status: 'eval_error', output: `spawn error: ${String(spawnErr)}`, timeMs: Date.now() - start });
            return;
        }

        let stdout = '';
        let stderr = '';
        let killed = false;

        proc.stdout?.on('data', (d: Buffer) => { stdout += d.toString(); });
        proc.stderr?.on('data', (d: Buffer) => { stderr += d.toString(); });

        const timer = setTimeout(() => {
            killed = true;
            proc.kill('SIGKILL');
        }, timeoutMs);

        proc.on('close', (exitCode) => {
            clearTimeout(timer);
            const output = (stdout + stderr).slice(0, 2000);
            if (killed) {
                resolve({ status: 'eval_error', output: `Timeout ${timeoutMs}ms\n${output}`, timeMs: Date.now() - start });
            } else if (exitCode === 0) {
                resolve({ status: 'pass', output, timeMs: Date.now() - start });
            } else {
                resolve({ status: 'fail', output, timeMs: Date.now() - start });
            }
        });

        proc.on('error', (err) => {
            clearTimeout(timer);
            resolve({ status: 'eval_error', output: String(err), timeMs: Date.now() - start });
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════
// F. Generation Configs (14 configs × 7 strategies)
// ═══════════════════════════════════════════════════════════════════════

function codePrompt(task: MbppPlusTask): string {
    return `Write a complete Python function that solves the following task. Output ONLY the Python code, no explanations, no markdown fences. IMPORTANT: the function MUST be named exactly \`${task.entry_point}\`. You MUST preserve ALL context from the prompt: include ALL import statements (e.g. from typing import List, import math) and ALL helper functions defined before \`${task.entry_point}\`.\n\n${task.prompt}`;
}

async function generateSolo(
    agent: Agent, task: MbppPlusTask, timeout: number,
    models: { gen1: string; gen2: string }
): Promise<{ code: string; duration: number; apiCallCount: number; error?: string }> {
    const start = Date.now();
    const r = await callAgent(agent, codePrompt(task), timeout, models);
    return {
        code: r.error ? '' : extractPythonCode(r.content),
        duration: Date.now() - start,
        apiCallCount: 1,
        error: r.error,
    };
}

async function generateLeadConsult(
    leader: Agent, task: MbppPlusTask, maxIter: number, timeout: number,
    models: { gen1: string; gen2: string }
): Promise<{ code: string; duration: number; apiCallCount: number; error?: string }> {
    const consultant: Agent = leader === 'gen1' ? 'gen2' : 'gen1';
    const start = Date.now();
    let apiCalls = 0;

    apiCalls++;
    const gen = await callAgent(leader, codePrompt(task), timeout, models);
    if (gen.error) return { code: '', duration: Date.now() - start, apiCallCount: apiCalls, error: gen.error };
    let code = extractPythonCode(gen.content);

    for (let i = 2; i <= maxIter; i++) {
        const reviewPrompt = `Review this Python function for correctness against the task description. If it's correct, respond with CONSENSUS_OK. If not, respond with CONSENSUS_KO and explain the issues.\n\nTask: ${task.prompt}\n\nImplementation:\n${code}`;
        apiCalls++;
        const review = await callAgent(consultant, reviewPrompt, timeout, models);
        if (review.error || review.content.includes('CONSENSUS_OK')) {
            return { code, duration: Date.now() - start, apiCallCount: apiCalls };
        }
        const fixPrompt = `Fix the issues in this Python function (keep the name \`${task.entry_point}\`):\n\n${review.content.slice(0, 2000)}\n\nTask: ${task.prompt}\n\nCurrent implementation:\n${code}\n\nOutput ONLY the corrected Python code (function named \`${task.entry_point}\`, include all imports and helper functions).`;
        apiCalls++;
        const fix = await callAgent(leader, fixPrompt, timeout, models);
        if (fix.error) break;
        code = extractPythonCode(fix.content);
    }
    return { code, duration: Date.now() - start, apiCallCount: apiCalls };
}

async function generateOrchCode(
    orchestrator: Agent, task: MbppPlusTask, maxIter: number, timeout: number,
    models: { gen1: string; gen2: string }
): Promise<{ code: string; duration: number; apiCallCount: number; error?: string }> {
    const coder: Agent = orchestrator === 'gen1' ? 'gen2' : 'gen1';
    const start = Date.now();
    let apiCalls = 0;

    const planPrompt = `You are a technical architect. For the following task, produce a detailed implementation plan: algorithm choice, edge cases to handle, time complexity.\n\nTask: ${task.prompt}`;
    apiCalls++;
    const plan = await callAgent(orchestrator, planPrompt, timeout, models);
    if (plan.error) return { code: '', duration: Date.now() - start, apiCallCount: apiCalls, error: plan.error };

    const implPrompt = `Implement this Python function following the plan below. The function MUST be named \`${task.entry_point}\`. Include all imports and helper functions from the prompt. Output ONLY the Python code.\n\nTask: ${task.prompt}\n\nPlan:\n${plan.content.slice(0, 3000)}`;
    apiCalls++;
    const impl = await callAgent(coder, implPrompt, timeout, models);
    if (impl.error) return { code: '', duration: Date.now() - start, apiCallCount: apiCalls, error: impl.error };
    let code = extractPythonCode(impl.content);

    for (let i = 2; i <= maxIter; i++) {
        const reviewPrompt = `Review this implementation for correctness. Respond with CONSENSUS_OK if correct, or CONSENSUS_KO with issues.\n\nTask: ${task.prompt}\n\nImplementation:\n${code}`;
        apiCalls++;
        const review = await callAgent(orchestrator, reviewPrompt, timeout, models);
        if (review.error || review.content.includes('CONSENSUS_OK')) {
            return { code, duration: Date.now() - start, apiCallCount: apiCalls };
        }
        const fixPrompt = `Fix the issues (keep function name \`${task.entry_point}\`):\n${review.content.slice(0, 2000)}\n\nTask: ${task.prompt}\n\nCode:\n${code}\n\nOutput ONLY the corrected Python code (function named \`${task.entry_point}\`, include all imports and helper functions).`;
        apiCalls++;
        const fix = await callAgent(coder, fixPrompt, timeout, models);
        if (fix.error) break;
        code = extractPythonCode(fix.content);
    }
    return { code, duration: Date.now() - start, apiCallCount: apiCalls };
}

async function generateSelfRefine(
    agent: Agent, task: MbppPlusTask, maxIter: number, timeout: number,
    models: { gen1: string; gen2: string }
): Promise<{ code: string; duration: number; apiCallCount: number; error?: string }> {
    const start = Date.now();
    let apiCalls = 0;
    apiCalls++;
    const gen = await callAgent(agent, codePrompt(task), timeout, models);
    if (gen.error) return { code: '', duration: Date.now() - start, apiCallCount: apiCalls, error: gen.error };
    let code = extractPythonCode(gen.content);

    for (let i = 2; i <= maxIter; i++) {
        const reviewPrompt = `Review your code for correctness, edge cases, and quality.\nIf you find issues, provide the corrected complete version (keep function name \`${task.entry_point}\`, include all imports and helpers).\nIf it's correct, respond with CONSENSUS_OK.\n\nTask: ${task.prompt}\n\nImplementation:\n${code}`;
        apiCalls++;
        const review = await callAgent(agent, reviewPrompt, timeout, models);
        if (review.error) break;
        if (review.content.includes('CONSENSUS_OK')) {
            return { code, duration: Date.now() - start, apiCallCount: apiCalls };
        }
        const extracted = extractPythonCode(review.content);
        if (extracted) code = extracted;
    }
    return { code, duration: Date.now() - start, apiCallCount: apiCalls };
}


// ─── Cross-Review: ximprove ───────────────────────────────────────────
// Both generate independently, other reviews primary's code, primary improves.
async function generateCrossImprove(
    primary: Agent, task: MbppPlusTask, maxIter: number, timeout: number,
    models: { gen1: string; gen2: string }
): Promise<{ code: string; duration: number; apiCallCount: number; error?: string }> {
    const other: Agent = primary === 'gen1' ? 'gen2' : 'gen1';
    const start = Date.now();
    let apiCalls = 0;

    // Step 1: Both generate independently (parallel)
    apiCalls += 2;
    const [genPrimary, genOther] = await Promise.all([
        callAgent(primary, codePrompt(task), timeout, models),
        callAgent(other, codePrompt(task), timeout, models),
    ]);
    if (genPrimary.error) return { code: '', duration: Date.now() - start, apiCallCount: apiCalls, error: genPrimary.error };
    let code = extractPythonCode(genPrimary.content);
    const otherCode = genOther.error ? '' : extractPythonCode(genOther.content);

    for (let i = 2; i <= maxIter; i++) {
        // Step 2: Other reviews primary's code (with awareness of its own solution)
        const reviewPrompt = `You generated your own solution to this task. Now review this OTHER implementation for correctness, edge cases, and quality. Point out any bugs or improvements.\n\nTask: ${task.prompt}\n\nYour solution (for reference):\n${otherCode.slice(0, 3000)}\n\nImplementation to review:\n${code}\n\nRespond with CONSENSUS_OK if correct, or describe the issues.`;
        apiCalls++;
        const review = await callAgent(other, reviewPrompt, timeout, models);
        if (review.error || review.content.includes('CONSENSUS_OK')) {
            return { code, duration: Date.now() - start, apiCallCount: apiCalls };
        }

        // Step 3: Primary improves based on feedback
        const fixPrompt = `Fix the issues found by another model's review (keep function name \`${task.entry_point}\`):\n\n${review.content.slice(0, 2000)}\n\nTask: ${task.prompt}\n\nYour current code:\n${code}\n\nOutput ONLY the corrected Python code (function named \`${task.entry_point}\`, include all imports and helper functions).`;
        apiCalls++;
        const fix = await callAgent(primary, fixPrompt, timeout, models);
        if (fix.error) break;
        code = extractPythonCode(fix.content);
    }
    return { code, duration: Date.now() - start, apiCallCount: apiCalls };
}

// ─── Cross-Review: xbest ─────────────────────────────────────────────
// Both generate, both cross-review & improve, select best by test execution.
// NOTE (paper fairness): xbest executes the task's base tests internally to
// select between two candidates. This is equivalent to pass@2 with oracle
// selection — a strictly stronger setting than pass@1. This must be
// disclosed in the paper to avoid data-leakage concerns from reviewers.
async function generateCrossBest(
    primary: Agent, task: MbppPlusTask, maxIter: number, timeout: number,
    models: { gen1: string; gen2: string }
): Promise<{ code: string; duration: number; apiCallCount: number; error?: string }> {
    const other: Agent = primary === 'gen1' ? 'gen2' : 'gen1';
    const start = Date.now();
    let apiCalls = 0;

    // Step 1: Both generate independently (parallel)
    apiCalls += 2;
    const [genPrimary, genOther] = await Promise.all([
        callAgent(primary, codePrompt(task), timeout, models),
        callAgent(other, codePrompt(task), timeout, models),
    ]);
    if (genPrimary.error && genOther.error) return { code: '', duration: Date.now() - start, apiCallCount: apiCalls, error: genPrimary.error };
    let codePrimary = genPrimary.error ? '' : extractPythonCode(genPrimary.content);
    let codeOther = genOther.error ? '' : extractPythonCode(genOther.content);

    for (let i = 2; i <= maxIter; i++) {
        // Step 2: Cross-review (parallel) — each reviews the other's code
        const reviewPrimaryPrompt = `Review this implementation for correctness. If you find issues, provide the corrected complete version. If correct, respond with CONSENSUS_OK.\n\nTask: ${task.prompt}\n\nYour solution (for reference):\n${codePrimary.slice(0, 3000)}\n\nImplementation to review:\n${codeOther}`;
        const reviewOtherPrompt = `Review this implementation for correctness. If you find issues, provide the corrected complete version. If correct, respond with CONSENSUS_OK.\n\nTask: ${task.prompt}\n\nYour solution (for reference):\n${codeOther.slice(0, 3000)}\n\nImplementation to review:\n${codePrimary}`;

        apiCalls += 2;
        const [revPrimary, revOther] = await Promise.all([
            callAgent(primary, reviewPrimaryPrompt, timeout, models),
            callAgent(other, reviewOtherPrompt, timeout, models),
        ]);

        // Step 3: Each improves their own code based on insights from reviewing the other
        if (revPrimary.content && !revPrimary.content.includes('CONSENSUS_OK')) {
            const fixPrimaryPrompt = `After reviewing another solution, improve your code (keep function name \`${task.entry_point}\`). Fix any issues you found apply to your own solution too.\n\nTask: ${task.prompt}\n\nYour current code:\n${codePrimary}\n\nYour review notes:\n${revPrimary.content.slice(0, 2000)}\n\nOutput ONLY the corrected Python code (function named \`${task.entry_point}\`, include all imports and helper functions).`;
            apiCalls++;
            const fixP = await callAgent(primary, fixPrimaryPrompt, timeout, models);
            if (!fixP.error) codePrimary = extractPythonCode(fixP.content);
        }
        if (revOther.content && !revOther.content.includes('CONSENSUS_OK')) {
            const fixOtherPrompt = `After reviewing another solution, improve your code (keep function name \`${task.entry_point}\`). Fix any issues you found apply to your own solution too.\n\nTask: ${task.prompt}\n\nYour current code:\n${codeOther}\n\nYour review notes:\n${revOther.content.slice(0, 2000)}\n\nOutput ONLY the corrected Python code (function named \`${task.entry_point}\`, include all imports and helper functions).`;
            apiCalls++;
            const fixO = await callAgent(other, fixOtherPrompt, timeout, models);
            if (!fixO.error) codeOther = extractPythonCode(fixO.content);
        }
    }

    // Step 4: Select best by test execution
    if (!codePrimary && codeOther) return { code: codeOther, duration: Date.now() - start, apiCallCount: apiCalls };
    if (codePrimary && !codeOther) return { code: codePrimary, duration: Date.now() - start, apiCallCount: apiCalls };

    const [execPrimary, execOther] = await Promise.all([
        executeCode(codePrimary, task.test_list, task.test_setup_code ?? ''),
        executeCode(codeOther, task.test_list, task.test_setup_code ?? ''),
    ]);

    // Prefer passing code; if both pass or both fail, prefer primary
    if (execPrimary.status === 'pass' && execOther.status !== 'pass') {
        return { code: codePrimary, duration: Date.now() - start, apiCallCount: apiCalls };
    } else if (execOther.status === 'pass' && execPrimary.status !== 'pass') {
        return { code: codeOther, duration: Date.now() - start, apiCallCount: apiCalls };
    }
    return { code: codePrimary, duration: Date.now() - start, apiCallCount: apiCalls };
}

// ─── Cross-Review: xfusion ───────────────────────────────────────────
// Both generate, both review the other, primary fuses everything.
async function generateCrossFusion(
    primary: Agent, task: MbppPlusTask, maxIter: number, timeout: number,
    models: { gen1: string; gen2: string }
): Promise<{ code: string; duration: number; apiCallCount: number; error?: string }> {
    const other: Agent = primary === 'gen1' ? 'gen2' : 'gen1';
    const start = Date.now();
    let apiCalls = 0;

    // Step 1: Both generate independently (parallel)
    apiCalls += 2;
    const [genPrimary, genOther] = await Promise.all([
        callAgent(primary, codePrompt(task), timeout, models),
        callAgent(other, codePrompt(task), timeout, models),
    ]);
    if (genPrimary.error) return { code: '', duration: Date.now() - start, apiCallCount: apiCalls, error: genPrimary.error };
    let codePrimary = extractPythonCode(genPrimary.content);
    const codeOther = genOther.error ? '' : extractPythonCode(genOther.content);

    for (let i = 2; i <= maxIter; i++) {
        // Step 2: Cross-review (parallel)
        const reviewByOtherPrompt = `Review this implementation for correctness, edge cases, and quality. List specific issues if any.\n\nTask: ${task.prompt}\n\nImplementation:\n${codePrimary}`;
        const reviewByPrimaryPrompt = `Review this implementation for correctness, edge cases, and quality. List specific issues if any.\n\nTask: ${task.prompt}\n\nImplementation:\n${codeOther}`;

        apiCalls += 2;
        const [feedbackOnPrimary, feedbackOnOther] = await Promise.all([
            callAgent(other, reviewByOtherPrompt, timeout, models),
            callAgent(primary, reviewByPrimaryPrompt, timeout, models),
        ]);

        // Step 3: Primary fuses both codes + both reviews into final version
        const fusionPrompt = `You have two implementations of the same task, plus reviews of each. Produce the BEST possible version by combining the strengths of both. IMPORTANT: the function MUST be named exactly \`${task.entry_point}\` — do NOT rename it. Output ONLY the Python code (full function).\n\nTask: ${task.prompt}\n\nImplementation A (yours):\n${codePrimary}\n\nReview of A:\n${(feedbackOnPrimary.content || 'No feedback').slice(0, 1500)}\n\nImplementation B (other model):\n${codeOther.slice(0, 3000)}\n\nReview of B:\n${(feedbackOnOther.content || 'No feedback').slice(0, 1500)}\n\nOutput ONLY the fused Python code (function named \`${task.entry_point}\`, include all imports and helper functions).`;
        apiCalls++;
        const fusion = await callAgent(primary, fusionPrompt, timeout, models);
        if (fusion.error) break;
        const extracted = extractPythonCode(fusion.content);
        if (extracted) codePrimary = extracted;
    }
    return { code: codePrimary, duration: Date.now() - start, apiCallCount: apiCalls };
}

async function generateForConfig(
    config: BenchConfig, task: MbppPlusTask, maxIter: number, timeout: number,
    models: { gen1: string; gen2: string }
): Promise<{ code: string; duration: number; apiCallCount: number; error?: string }> {
    switch (config) {
        case 'gen1-solo': return generateSolo('gen1', task, timeout, models);
        case 'gen2-solo': return generateSolo('gen2', task, timeout, models);
        case 'gen1-lead': return generateLeadConsult('gen1', task, maxIter, timeout, models);
        case 'gen2-lead': return generateLeadConsult('gen2', task, maxIter, timeout, models);
        case 'gen1-orch': return generateOrchCode('gen1', task, maxIter, timeout, models);
        case 'gen2-orch': return generateOrchCode('gen2', task, maxIter, timeout, models);
        case 'gen1-selfrefine': return generateSelfRefine('gen1', task, maxIter, timeout, models);
        case 'gen2-selfrefine': return generateSelfRefine('gen2', task, maxIter, timeout, models);
        case 'gen1-ximprove': return generateCrossImprove('gen1', task, maxIter, timeout, models);
        case 'gen2-ximprove': return generateCrossImprove('gen2', task, maxIter, timeout, models);
        case 'gen1-xbest': return generateCrossBest('gen1', task, maxIter, timeout, models);
        case 'gen2-xbest': return generateCrossBest('gen2', task, maxIter, timeout, models);
        case 'gen1-xfusion': return generateCrossFusion('gen1', task, maxIter, timeout, models);
        case 'gen2-xfusion': return generateCrossFusion('gen2', task, maxIter, timeout, models);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// G. Judge Prompt Builder
// ═══════════════════════════════════════════════════════════════════════

function buildEvalPrompt(
    taskPrompt: string,
    candidates: { label: string; code: string; execStatus: ExecStatus }[],
    blind: boolean = true,
): string {
    const codeSection = candidates.map(c => {
        const header = blind
            ? `### Candidate ${c.label}`
            : `### Candidate ${c.label} [exec: ${c.execStatus}]`;
        return `${header}\n\`\`\`python\n${c.code.slice(0, 8000)}\n\`\`\``;
    }).join('\n\n');

    return `You are evaluating code generated by LLMs for the following task.

## Task
${taskPrompt}

## Candidates
${codeSection}

## Instructions
Score each candidate on these criteria (integer 1–10):
- **correctness**: Does the code produce correct output for typical inputs?
- **completeness**: Does it handle all aspects of the task description?
- **edgeCases**: Does it handle edge cases (empty input, boundaries, etc.)?
- **codeQuality**: Is the code well-structured, efficient, idiomatic?
- **readability**: Is the code easy to understand?
- **total**: Overall score (1–10), your holistic judgment.
- **risk_fail_prob**: Float 0.0–1.0, your estimated probability that the code would fail on hidden/unseen test cases.
- **justification**: Brief explanation of your scoring.

Output ONLY a JSON array (no markdown fences, no extra text):
[{"label":"A","correctness":N,"completeness":N,"edgeCases":N,"codeQuality":N,"readability":N,"total":N,"risk_fail_prob":0.X,"justification":"..."},...]`;
}

/**
 * Build the R2 debate prompt for divergent candidates.
 * Note: no blind parameter needed — the debate prompt shows R1 scores
 * and justifications (not raw code or exec status), so there is no
 * information leakage regardless of blind/informed mode.
 */
function buildDebatePrompt(
    taskPrompt: string,
    round1Results: { judge: string; scores: Record<string, QualityScores> }[],
    divergentLabels: string[],
): string {
    const r1Section = round1Results.map(r =>
        `### ${r.judge}\n${divergentLabels.map(l =>
            r.scores[l]
                ? `${l}: total=${r.scores[l].total}, risk=${r.scores[l].risk_fail_prob}, "${r.scores[l].justification}"`
                : `${l}: (no score)`
        ).join('\n')}`
    ).join('\n\n');

    return `The following candidates had divergent scores from two judges. Review the other judge's reasoning and re-score ONLY the listed candidates.

## Task
${taskPrompt}

## Round 1 Scores (divergent candidates: ${divergentLabels.join(', ')})
${r1Section}

Re-evaluate candidates ${divergentLabels.join(', ')} considering the other judge's perspective.

Output ONLY a JSON array with your updated scores:
[{"label":"X","correctness":N,"completeness":N,"edgeCases":N,"codeQuality":N,"readability":N,"total":N,"risk_fail_prob":0.X,"justification":"..."},...]`;
}

// ═══════════════════════════════════════════════════════════════════════
// H. Score Parser
// ═══════════════════════════════════════════════════════════════════════

function parseScores(
    raw: string, expectedLabels: string[]
): Record<string, QualityScores> | null {
    try {
        // Try to extract JSON array from response
        let jsonStr = raw.trim();
        // Strip markdown fences if present
        const fenceMatch = jsonStr.match(/```(?:json)?\n?([\s\S]*?)```/);
        if (fenceMatch) jsonStr = fenceMatch[1].trim();
        // Find JSON array
        const arrMatch = jsonStr.match(/\[[\s\S]*\]/);
        if (!arrMatch) return null;

        const arr: Array<{
            label?: string;
            correctness?: number;
            completeness?: number;
            edgeCases?: number;
            edge_cases?: number;
            codeQuality?: number;
            code_quality?: number;
            readability?: number;
            total?: number;
            risk_fail_prob?: number;
            justification?: string;
        }> = JSON.parse(arrMatch[0]);

        const result: Record<string, QualityScores> = {};
        for (const item of arr) {
            const label = item.label?.toUpperCase();
            if (!label || !expectedLabels.includes(label)) continue;
            result[label] = {
                correctness: item.correctness ?? 5,
                completeness: item.completeness ?? 5,
                edgeCases: item.edgeCases ?? item.edge_cases ?? 5,
                codeQuality: item.codeQuality ?? item.code_quality ?? 5,
                readability: item.readability ?? 5,
                total: item.total ?? 5,
                justification: item.justification ?? '',
                risk_fail_prob: Math.max(0, Math.min(1, item.risk_fail_prob ?? 0.5)),
            };
        }
        return Object.keys(result).length > 0 ? result : null;
    } catch {
        return null;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// I. Divergence Detection — Absolute τ
// ═══════════════════════════════════════════════════════════════════════

const DIVERGENCE_TAU = 2.0; // absolute threshold on /10 scale

interface DivergenceResult {
    divergentLabels: string[];
    triggerType: 'score_divergence' | 'risk_disagreement' | 'both' | 'none';
}

function detectDivergence(
    judgeResults: { judge: string; scores: Record<string, QualityScores> }[],
    labels: string[],
    tau: number = DIVERGENCE_TAU,
): DivergenceResult {
    if (judgeResults.length < 2) return { divergentLabels: [], triggerType: 'none' };

    const divergentLabels: string[] = [];
    let hasScoreDivergence = false;
    let hasRiskDisagreement = false;

    for (const label of labels) {
        const scores = judgeResults.map(j => j.scores[label]).filter(Boolean);
        if (scores.length < 2) continue;

        // Check absolute score divergence
        const totals = scores.map(s => s.total);
        const maxDelta = Math.max(...totals) - Math.min(...totals);
        if (maxDelta > tau) {
            divergentLabels.push(label);
            hasScoreDivergence = true;
            continue;
        }

        // Check risk_fail_prob disagreement (one >0.5, other <0.5)
        const risks = scores.map(s => s.risk_fail_prob);
        const someAbove = risks.some(r => r > 0.5);
        const someBelow = risks.some(r => r < 0.5);
        if (someAbove && someBelow) {
            divergentLabels.push(label);
            hasRiskDisagreement = true;
        }
    }

    let triggerType: DivergenceResult['triggerType'] = 'none';
    if (hasScoreDivergence && hasRiskDisagreement) triggerType = 'both';
    else if (hasScoreDivergence) triggerType = 'score_divergence';
    else if (hasRiskDisagreement) triggerType = 'risk_disagreement';

    return { divergentLabels, triggerType };
}

// ═══════════════════════════════════════════════════════════════════════
// J. Judge Panel — 3-Level Evaluation
// ═══════════════════════════════════════════════════════════════════════

interface JudgeCaller {
    name: string;
    call: (prompt: string, timeout: number) => Promise<ApiResult>;
}

function averageScores(
    allScores: Record<string, QualityScores>[],
    labels: string[],
): Record<string, QualityScores> {
    const result: Record<string, QualityScores> = {};
    for (const label of labels) {
        const judges = allScores.filter(s => s[label]).map(s => s[label]);
        if (judges.length === 0) continue;
        const n = judges.length;
        result[label] = {
            correctness: parseFloat((judges.reduce((a, s) => a + s.correctness, 0) / n).toFixed(1)),
            completeness: parseFloat((judges.reduce((a, s) => a + s.completeness, 0) / n).toFixed(1)),
            edgeCases: parseFloat((judges.reduce((a, s) => a + s.edgeCases, 0) / n).toFixed(1)),
            codeQuality: parseFloat((judges.reduce((a, s) => a + s.codeQuality, 0) / n).toFixed(1)),
            readability: parseFloat((judges.reduce((a, s) => a + s.readability, 0) / n).toFixed(1)),
            total: parseFloat((judges.reduce((a, s) => a + s.total, 0) / n).toFixed(1)),
            justification: judges.map(j => j.justification).join(' | '),
            risk_fail_prob: parseFloat((judges.reduce((a, s) => a + s.risk_fail_prob, 0) / n).toFixed(3)),
        };
    }
    return result;
}

// ── Strategy B: Binary pass/fail prediction (one call per candidate per judge) ──

function buildBinaryPrompt(taskPrompt: string, entryPoint: string, code: string): string {
    return `You are a code evaluation expert. Given a task description and a candidate solution, predict whether this code will pass all unit tests.

Task:
${taskPrompt}

Entry point: ${entryPoint}

Candidate code:
${code}

Answer ONLY with this exact JSON format, nothing else:
{"pass": true, "confidence": 0.85, "reason": "brief reason"}

Rules:
- "pass": true if you believe the code will pass ALL tests, false otherwise
- "confidence": your confidence in this prediction, from 0.0 to 1.0
- "reason": one sentence explaining your prediction`;
}

function parseBinaryResponse(raw: string): { pass: boolean; confidence: number; reason: string } | null {
    try {
        const match = raw.match(/\{[\s\S]*?\}/);
        if (!match) return null;
        const obj = JSON.parse(match[0]);
        if (typeof obj.pass !== 'boolean' || typeof obj.confidence !== 'number') return null;
        return { pass: !!obj.pass, confidence: Math.max(0, Math.min(1, obj.confidence)), reason: String(obj.reason || '') };
    } catch { return null; }
}

async function evaluateWithBinaryStrategy(
    taskPrompt: string,
    entryPoint: string,
    outputs: { config: BenchConfig; code: string; execStatus: ExecStatus }[],
    timeout: number,
    judges: JudgeCaller[],
    tbCall: (p: string, t: number) => Promise<ApiResult>,
): Promise<JudgeAudit> {
    const valid = outputs.filter(o => o.code.length > 0);

    // Dedup identical codes
    const codeToConfigs = new Map<string, BenchConfig[]>();
    for (const o of valid) {
        const existing = codeToConfigs.get(o.code);
        if (existing) { existing.push(o.config); }
        else { codeToConfigs.set(o.code, [o.config]); }
    }
    const unique: typeof valid = [];
    const dupMap = new Map<BenchConfig, BenchConfig>();
    for (const [code, cfgs] of codeToConfigs) {
        const repr = cfgs[0];
        unique.push(valid.find(o => o.config === repr)!);
        for (const cfg of cfgs.slice(1)) dupMap.set(cfg, repr);
    }
    if (dupMap.size > 0) process.stdout.write(`[DEDUP:${valid.length}->${unique.length}] `);

    const binaryVerdicts: Partial<Record<BenchConfig, { pass: boolean; confidence: number; reason: string }[]>> = {};
    const finalScores: Partial<Record<BenchConfig, QualityScores>> = {};
    let tiebreakerUsed = false;
    let tiebreakerError: string | undefined;

    for (const entry of unique) {
        const prompt = buildBinaryPrompt(taskPrompt, entryPoint, entry.code);
        const verdicts: { pass: boolean; confidence: number; reason: string }[] = [];

        // Call both judges in parallel
        const results = await Promise.all(judges.map(j => j.call(prompt, timeout)));
        for (const res of results) {
            const parsed = parseBinaryResponse(res.content);
            if (parsed) verdicts.push(parsed);
        }

        // Divergence: opposite verdict OR |Δconfidence| > 0.3
        const divergent = verdicts.length === 2 &&
            (verdicts[0].pass !== verdicts[1].pass ||
             Math.abs(verdicts[0].confidence - verdicts[1].confidence) > 0.3);

        // Tie-breaker on divergence
        if (divergent) {
            const tbResult = await tbCall(prompt, timeout);
            const tbParsed = parseBinaryResponse(tbResult.content);
            if (tbParsed) {
                verdicts.push(tbParsed);
                tiebreakerUsed = true;
            } else {
                tiebreakerError = tbResult.error || 'Failed to parse tiebreaker response';
            }
        }

        // Store verdicts for this config
        binaryVerdicts[entry.config] = verdicts;

        // Convert to QualityScores for compatibility (binary → 10 or 1 total)
        if (verdicts.length > 0) {
            const passCount = verdicts.filter(v => v.pass).length;
            const majorityPass = passCount > verdicts.length / 2;
            const avgConfidence = verdicts.reduce((a, v) => a + v.confidence, 0) / verdicts.length;
            const total = majorityPass ? Math.round(5 + avgConfidence * 5) : Math.round((1 - avgConfidence) * 5);
            finalScores[entry.config] = {
                correctness: total, completeness: total, edgeCases: total,
                codeQuality: total, readability: total, total,
                justification: verdicts.map(v => `${v.pass ? 'PASS' : 'FAIL'}@${v.confidence.toFixed(2)}: ${v.reason}`).join(' | '),
                risk_fail_prob: majorityPass ? 1 - avgConfidence : avgConfidence,
            };
        }

        // Propagate to duplicates
        for (const [dup, repr] of dupMap) {
            if (repr === entry.config) {
                binaryVerdicts[dup] = binaryVerdicts[entry.config];
                if (finalScores[entry.config]) finalScores[dup] = finalScores[entry.config];
            }
        }
    }

    return {
        round1: [], divergenceDetected: tiebreakerUsed, tiebreakerUsed, tiebreakerError,
        j0: finalScores, j1: finalScores, j2: finalScores, finalScores,
        binaryVerdicts,
    };
}

// ── Strategy A: Scoring (existing panel evaluation) ──

async function evaluateWithPanel(
    taskPrompt: string,
    outputs: { config: BenchConfig; code: string; execStatus: ExecStatus }[],
    timeout: number,
    judges: JudgeCaller[],
    tbCall: (p: string, t: number) => Promise<ApiResult>,
    tau: number = DIVERGENCE_TAU,
    blind: boolean = true,
    noDebate: boolean = false,
): Promise<JudgeAudit> {
    const valid = outputs.filter(o => o.code.length > 0);

    // Dedup: group configs that produced identical code, send only unique codes to judges
    const codeToConfigs = new Map<string, BenchConfig[]>();
    for (const o of valid) {
        const existing = codeToConfigs.get(o.code);
        if (existing) { existing.push(o.config); }
        else { codeToConfigs.set(o.code, [o.config]); }
    }
    // Build unique entries (one per distinct code), keeping first config as representative
    const unique: typeof valid = [];
    const dupMap = new Map<BenchConfig, BenchConfig>(); // duplicate -> representative
    for (const [code, cfgs] of codeToConfigs) {
        const repr = cfgs[0];
        const entry = valid.find(o => o.config === repr)!;
        unique.push(entry);
        for (const cfg of cfgs.slice(1)) {
            dupMap.set(cfg, repr);
        }
    }
    if (dupMap.size > 0) {
        process.stdout.write(`[DEDUP:${valid.length}->${unique.length}] `);
    }

    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const labels = alphabet.split('').slice(0, unique.length);
    const shuffled = shuffle(unique);
    const labelMap = new Map<string, BenchConfig>();
    for (let i = 0; i < shuffled.length; i++) {
        labelMap.set(labels[i], shuffled[i].config);
    }

    const evalPrompt = buildEvalPrompt(
        taskPrompt,
        shuffled.map((s, i) => ({ label: labels[i], code: s.code, execStatus: s.execStatus })),
        blind,
    );

    // Convert label-keyed scores to config-keyed, propagating to duplicates
    function labelToConfig(scores: Record<string, QualityScores>): Partial<Record<BenchConfig, QualityScores>> {
        const result: Partial<Record<BenchConfig, QualityScores>> = {};
        for (const [label, s] of Object.entries(scores)) {
            const cfg = labelMap.get(label);
            if (cfg) result[cfg] = s;
        }
        // Propagate scores to configs with identical code
        for (const [dup, repr] of dupMap) {
            if (result[repr] && !result[dup]) {
                result[dup] = result[repr];
            }
        }
        return result;
    }

    // ─── Round 1: independent scoring ───
    const round1Results = await Promise.all(
        judges.map(async (judge) => {
            for (let attempt = 0; attempt < 2; attempt++) {
                try {
                    const r = await judge.call(evalPrompt, timeout);
                    if (r.error) { console.error(`\n  [${judge.name}] API error (attempt ${attempt + 1}): ${r.error.slice(0, 200)}`); continue; }
                    const scores = parseScores(r.content, labels);
                    if (!scores) { console.error(`\n  [${judge.name}] Parse fail (attempt ${attempt + 1}) — raw (500 chars): ${r.content.slice(0, 500)}`); continue; }
                    return { judge: judge.name, scores };
                } catch (e) { console.error(`\n  [${judge.name}] Exception (attempt ${attempt + 1}): ${String(e).slice(0, 200)}`); continue; }
            }
            return { judge: judge.name, scores: null };
        })
    );

    const round1Valid = round1Results.filter(r => r.scores !== null) as
        { judge: string; scores: Record<string, QualityScores> }[];

    const r1Status = round1Results.map(r => `${r.judge}:${r.scores ? 'OK' : 'FAIL'}`).join(' ');
    process.stdout.write(`[R1 ${r1Status}] `);

    if (round1Valid.length === 0) {
        const empty: Partial<Record<BenchConfig, QualityScores>> = {};
        return {
            round1: [], divergenceDetected: false, tiebreakerUsed: false,
            j0: empty, j1: empty, j2: empty, finalScores: empty,
        };
    }

    if (round1Valid.length < round1Results.length) {
        const failed = round1Results.filter(r => r.scores === null).map(r => r.judge);
        console.warn(`\n  \u26a0 Only ${round1Valid.length}/${round1Results.length} judges responded (${failed.join(', ')} failed)`);
    }

    const round1Audit: JudgeRound[] = round1Valid.map(r => ({
        judge: r.judge, round: 'R1' as const,
        scores: labelToConfig(r.scores),
    }));

    // j0: R1 baseline average
    const j0Scores = averageScores(round1Valid.map(r => r.scores), labels);
    const j0 = labelToConfig(j0Scores);

    // ─── Divergence check ───
    const { divergentLabels, triggerType } = detectDivergence(round1Valid, labels, tau);

    if (divergentLabels.length === 0 || noDebate) {
        return {
            round1: round1Audit, divergenceDetected: !noDebate && divergentLabels.length > 0,
            tiebreakerUsed: false,
            j0, j1: j0, j2: j0, finalScores: j0,
        };
    }

    // ─── Round 2: DEBATE ───
    const divergentConfigNames = divergentLabels.map(l => `${l}(${CONFIG_SHORT[labelMap.get(l)!]})`);
    process.stdout.write(`[DEBATE:${triggerType} ${divergentConfigNames.join(',')}] `);

    const debatePrompt = buildDebatePrompt(taskPrompt, round1Valid, divergentLabels);
    const debateMessages: { judge: string; argument: string }[] = [];

    const round2Results = await Promise.all(
        judges.map(async (judge) => {
            try {
                const r = await judge.call(debatePrompt, timeout);
                if (r.error) return { judge: judge.name, scores: null };
                const scores = parseScores(r.content, divergentLabels);
                const argument = scores
                    ? Object.values(scores).map(s => s.justification).join('; ')
                    : r.content.slice(0, 500);
                debateMessages.push({ judge: judge.name, argument });
                return { judge: judge.name, scores };
            } catch { return { judge: judge.name, scores: null }; }
        })
    );

    const round2Valid = round2Results.filter(r => r.scores !== null) as
        { judge: string; scores: Record<string, QualityScores> }[];

    const r2Status = round2Results.map(r => `${r.judge}:${r.scores ? 'OK' : 'FAIL'}`).join(' ');
    process.stdout.write(`[R2 ${r2Status}] `);

    // Merge: use R2 for divergent labels, R1 for the rest
    const mergedScores: Record<string, QualityScores>[] = round1Valid.map((r1) => {
        const r2 = round2Valid.find(r => r.judge === r1.judge);
        const merged: Record<string, QualityScores> = { ...r1.scores };
        if (r2) {
            for (const label of divergentLabels) {
                if (r2.scores[label]) merged[label] = r2.scores[label];
            }
        }
        return merged;
    });

    const j1Scores = averageScores(mergedScores, labels);
    const j1 = labelToConfig(j1Scores);

    const round2Audit: JudgeRound[] = round2Valid.map(r => ({
        judge: r.judge, round: 'R2' as const,
        scores: labelToConfig(r.scores),
    }));

    // Check if divergence persists after R2
    const stillDivergent: string[] = [];
    for (const label of divergentLabels) {
        const scores = mergedScores.filter(s => s[label]).map(s => s[label]);
        if (scores.length >= 2) {
            const totals = scores.map(s => s.total);
            if (Math.max(...totals) - Math.min(...totals) > tau) {
                stillDivergent.push(label);
            }
        }
    }

    // ─── Tie-breaker: Gemini (only if divergence persists) ───
    let tiebreakerUsed = false;
    let tiebreakerError: string | undefined;

    if (stillDivergent.length > 0 && tbCall) {
        const tbLabels = stillDivergent.map(l => `${l}(${CONFIG_SHORT[labelMap.get(l)!]})`);
        process.stdout.write(`[TB: ${tbLabels.join(',')}] `);
        tiebreakerUsed = true;

        try {
            const tbResult = await tbCall(evalPrompt, timeout);
            if (tbResult.error) {
                tiebreakerError = tbResult.error;
                process.stdout.write(`[TB-ERR: ${tbResult.error.slice(0, 40)}] `);
            } else {
                const tbScores = parseScores(tbResult.content, labels);
                if (tbScores) {
                    mergedScores.push(tbScores);
                } else {
                    tiebreakerError = 'Failed to parse tie-breaker scores';
                    process.stdout.write('[TB-PARSE-ERR] ');
                }
            }
        } catch (e) {
            tiebreakerError = String(e);
            process.stdout.write(`[TB-ERR: ${String(e).slice(0, 40)}] `);
        }
    }

    const j2Scores = averageScores(mergedScores, labels);
    const j2 = labelToConfig(j2Scores);

    return {
        round1: round1Audit,
        divergenceDetected: true,
        divergentConfigs: divergentLabels.map(l => CONFIG_SHORT[labelMap.get(l)!]),
        debate: { trigger: divergentConfigNames.join(', '), triggerType: triggerType as DebateRound['triggerType'], messages: debateMessages },
        round2: round2Audit,
        tiebreakerUsed,
        tiebreakerError,
        j0, j1, j2,
        finalScores: j2,
    };
}

// ═══════════════════════════════════════════════════════════════════════
// K. Paper Metrics (all inline, no external deps)
// ═══════════════════════════════════════════════════════════════════════

// --- pass@k (Chen et al. 2021, log-space for numerical stability) ---

function logBinom(n: number, k: number): number {
    if (k > n || k < 0) return -Infinity;
    if (k === 0 || k === n) return 0;
    let result = 0;
    for (let i = 0; i < k; i++) {
        result += Math.log(n - i) - Math.log(i + 1);
    }
    return result;
}

function passAtK(n: number, c: number, k: number): number {
    // pass@k = 1 - C(n-c, k) / C(n, k)
    if (n < k) return c > 0 ? 1.0 : 0.0;
    if (c === 0) return 0.0;
    if (c >= n) return 1.0;
    const logRatio = logBinom(n - c, k) - logBinom(n, k);
    return 1.0 - Math.exp(logRatio);
}

function computePassAtK(
    results: TaskResult[], configs: BenchConfig[], k: number, usePlus: boolean = false
): Partial<Record<BenchConfig, number>> {
    const out: Partial<Record<BenchConfig, number>> = {};
    for (const cfg of configs) {
        // Group by task_id to get n (runs per task) and c (passes)
        const byTask = new Map<string, { n: number; c: number }>();
        for (const r of results) {
            const cr = r.configs.find(c => c.config === cfg);
            if (!cr) continue;
            const entry = byTask.get(r.taskId) ?? { n: 0, c: 0 };
            entry.n++;
            const status = usePlus ? cr.execStatusPlus : cr.execStatus;
            if (status === 'pass') entry.c++;
            byTask.set(r.taskId, entry);
        }
        if (byTask.size === 0) continue;
        // Average pass@k across tasks
        let sum = 0;
        for (const { n, c } of byTask.values()) {
            sum += passAtK(n, c, k);
        }
        out[cfg] = parseFloat((sum / byTask.size * 100).toFixed(1));
    }
    return out;
}

// --- Spearman rank correlation ---

function assignRanks(values: number[]): number[] {
    const indexed = values.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => a.v - b.v);
    const ranks = new Array<number>(values.length);
    let i = 0;
    while (i < indexed.length) {
        let j = i;
        while (j < indexed.length && indexed[j].v === indexed[i].v) j++;
        // Average rank for ties
        const avgRank = (i + j + 1) / 2; // 1-based
        for (let k = i; k < j; k++) {
            ranks[indexed[k].i] = avgRank;
        }
        i = j;
    }
    return ranks;
}

function spearmanCorrelation(x: number[], y: number[]): number | null {
    const n = Math.min(x.length, y.length);
    if (n < 3) return null;
    const rx = assignRanks(x.slice(0, n));
    const ry = assignRanks(y.slice(0, n));
    let sumD2 = 0;
    for (let i = 0; i < n; i++) {
        const d = rx[i] - ry[i];
        sumD2 += d * d;
    }
    return parseFloat((1 - (6 * sumD2) / (n * (n * n - 1))).toFixed(4));
}

// --- Cohen's kappa ---

function cohensKappa(x: boolean[], y: boolean[]): number | null {
    const n = Math.min(x.length, y.length);
    if (n < 2) return null;
    // 2x2 contingency: both true, both false, x-only, y-only
    let a = 0, b = 0, c = 0, d = 0;
    for (let i = 0; i < n; i++) {
        if (x[i] && y[i]) a++;
        else if (!x[i] && !y[i]) d++;
        else if (x[i] && !y[i]) b++;
        else c++;
    }
    const po = (a + d) / n;
    const pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n);
    if (pe >= 1) return null;
    return parseFloat(((po - pe) / (1 - pe)).toFixed(4));
}

// --- Bug-catching: recall, FPR, AUC ---

interface BugCatchingResult {
    recall: number | null;
    fpr: number | null;
    auc: number | null;
}

function computeBugCatching(
    results: TaskResult[],
): BugCatchingResult {
    // Collect pairs: (actual_bug: boolean, risk_fail_prob: number)
    const pairs: { actualBug: boolean; risk: number }[] = [];

    for (const r of results) {
        for (const cr of r.configs) {
            const score = r.judgeAudit.finalScores[cr.config];
            if (!score) continue;
            pairs.push({
                actualBug: cr.execStatus !== 'pass',
                risk: score.risk_fail_prob,
            });
        }
    }

    if (pairs.length < 2) return { recall: null, fpr: null, auc: null };

    // Recall and FPR at threshold 0.5
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (const p of pairs) {
        const predicted = p.risk > 0.5;
        if (p.actualBug && predicted) tp++;
        else if (p.actualBug && !predicted) fn++;
        else if (!p.actualBug && predicted) fp++;
        else tn++;
    }

    const recall = (tp + fn) > 0 ? parseFloat((tp / (tp + fn)).toFixed(4)) : null;
    const fpr = (fp + tn) > 0 ? parseFloat((fp / (fp + tn)).toFixed(4)) : null;

    // AUC via trapezoidal rule
    const sorted = [...pairs].sort((a, b) => b.risk - a.risk);
    const totalPos = pairs.filter(p => p.actualBug).length;
    const totalNeg = pairs.filter(p => !p.actualBug).length;

    if (totalPos === 0 || totalNeg === 0) return { recall, fpr, auc: null };

    let auc = 0;
    let tpCount = 0;
    let fpCount = 0;
    let prevTPR = 0;
    let prevFPR = 0;

    for (const p of sorted) {
        if (p.actualBug) tpCount++;
        else fpCount++;
        const curTPR = tpCount / totalPos;
        const curFPR = fpCount / totalNeg;
        // Trapezoidal rule
        auc += (curFPR - prevFPR) * (curTPR + prevTPR) / 2;
        prevTPR = curTPR;
        prevFPR = curFPR;
    }

    return {
        recall,
        fpr,
        auc: parseFloat(auc.toFixed(4)),
    };
}

// --- Escalation audit ---

interface EscalationStats {
    debatesTriggered: number;
    totalEvaluations: number;
    tiebreakerUsed: number;
    tiebreakerErrors: number;
    avgScoreShift: number | null;
    triggerBreakdown: { scoreDivergence: number; riskDisagreement: number; both: number };
}

function computeEscalationStats(results: TaskResult[]): EscalationStats {
    let debatesTriggered = 0;
    let tiebreakerUsed = 0;
    let tiebreakerErrors = 0;
    const scoreShifts: number[] = [];
    const triggerBreakdown = { scoreDivergence: 0, riskDisagreement: 0, both: 0 };

    for (const r of results) {
        const a = r.judgeAudit;
        if (a.divergenceDetected) {
            debatesTriggered++;
            if (a.debate) {
                if (a.debate.triggerType === 'score_divergence') triggerBreakdown.scoreDivergence++;
                else if (a.debate.triggerType === 'risk_disagreement') triggerBreakdown.riskDisagreement++;
                else if (a.debate.triggerType === 'both') triggerBreakdown.both++;
            }
        }
        if (a.tiebreakerUsed) tiebreakerUsed++;
        if (a.tiebreakerError) tiebreakerErrors++;

        // Score shift: |j2.total - j0.total| averaged over configs
        for (const cfg of Object.keys(a.j0) as BenchConfig[]) {
            const s0 = a.j0[cfg];
            const s2 = a.j2[cfg];
            if (s0 && s2) {
                scoreShifts.push(Math.abs(s2.total - s0.total));
            }
        }
    }

    return {
        debatesTriggered,
        totalEvaluations: results.length,
        tiebreakerUsed,
        tiebreakerErrors,
        avgScoreShift: scoreShifts.length > 0
            ? parseFloat((scoreShifts.reduce((a, b) => a + b, 0) / scoreShifts.length).toFixed(3))
            : null,
        triggerBreakdown,
    };
}

// --- Compute all paper metrics ---

function mcNemar(b: number, c: number): { chi2: number; pLevel: string } {
    if (b + c === 0) return { chi2: 0, pLevel: 'n.s.' };
    const chi2 = ((b - c) ** 2) / (b + c);
    const pLevel = chi2 > 6.63 ? 'p<0.01' : chi2 > 3.84 ? 'p<0.05' : 'n.s.';
    return { chi2, pLevel };
}

function computeStdDev(values: number[]): number {
    if (values.length < 2) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sqDiffs = values.map(v => (v - mean) ** 2);
    return Math.sqrt(sqDiffs.reduce((a, b) => a + b, 0) / (values.length - 1));
}

function computePaperMetrics(
    results: TaskResult[], configs: BenchConfig[], runs: number
): PaperMetrics {
    // pass@k
    const passAtKResults: Record<string, Partial<Record<BenchConfig, number>>> = {};
    passAtKResults['k1'] = computePassAtK(results, configs, 1);
    passAtKResults['k1_plus'] = computePassAtK(results, configs, 1, true);
    if (runs >= 5) {
        passAtKResults['k5'] = computePassAtK(results, configs, 5);
        passAtKResults['k5_plus'] = computePassAtK(results, configs, 5, true);
    }

    // Inter-judge: Spearman + Cohen's kappa from R1 scores
    const claudeR1Totals: number[] = [];
    const gptR1Totals: number[] = [];
    const claudeR1OK: boolean[] = [];
    const gptR1OK: boolean[] = [];

    for (const r of results) {
        const claudeR1 = r.judgeAudit.round1.find(j => j.judge === 'Claude' && j.round === 'R1');
        const gptR1 = r.judgeAudit.round1.find(j => j.judge === 'GPT' && j.round === 'R1');
        if (!claudeR1 || !gptR1) continue;

        for (const cfg of configs) {
            const cs = claudeR1.scores[cfg];
            const gs = gptR1.scores[cfg];
            if (cs && gs) {
                claudeR1Totals.push(cs.total);
                gptR1Totals.push(gs.total);
                claudeR1OK.push(cs.total >= 5);
                gptR1OK.push(gs.total >= 5);
            }
        }
    }

    const spearmanR1 = spearmanCorrelation(claudeR1Totals, gptR1Totals);
    const cohensKappaR1 = cohensKappa(claudeR1OK, gptR1OK);

    // Bug-catching
    const bugCatching = computeBugCatching(results);

    // Escalation
    const escalation = computeEscalationStats(results);

    // Variance (stddev) of pass@1 per config across tasks
    const variance: Partial<Record<BenchConfig, { mean: number; stdDev: number; n: number }>> = {};
    for (const cfg of configs) {
        // Collect per-task pass rate (0 or 100 for each run)
        const byTask = new Map<string, number[]>();
        for (const r of results) {
            const cr = r.configs.find(c => c.config === cfg);
            if (!cr) continue;
            const arr = byTask.get(r.taskId) ?? [];
            arr.push(cr.execStatus === 'pass' ? 100 : 0);
            byTask.set(r.taskId, arr);
        }
        // Per-task pass rate (mean across runs)
        const taskRates: number[] = [];
        for (const vals of byTask.values()) {
            taskRates.push(vals.reduce((a, b) => a + b, 0) / vals.length);
        }
        if (taskRates.length > 0) {
            const mean = parseFloat((taskRates.reduce((a, b) => a + b, 0) / taskRates.length).toFixed(1));
            const stdDev = parseFloat(computeStdDev(taskRates).toFixed(1));
            variance[cfg] = { mean, stdDev, n: taskRates.length };
        }
    }

    // McNemar paired tests: solo vs each collab config, per generator
    const mcnemarResults: { pair: string; b: number; c: number; chi2: number; pLevel: string }[] = [];
    const generators: Array<{ prefix: string; solo: BenchConfig; collabs: BenchConfig[] }> = [
        { prefix: 'gen1', solo: 'gen1-solo', collabs: ['gen1-lead', 'gen1-orch', 'gen1-selfrefine', 'gen1-ximprove', 'gen1-xbest', 'gen1-xfusion'] },
        { prefix: 'gen2', solo: 'gen2-solo', collabs: ['gen2-lead', 'gen2-orch', 'gen2-selfrefine', 'gen2-ximprove', 'gen2-xbest', 'gen2-xfusion'] },
    ];
    for (const gen of generators) {
        if (!configs.includes(gen.solo)) continue;
        for (const collab of gen.collabs) {
            if (!configs.includes(collab)) continue;
            // Base tests
            let b = 0, c = 0;
            // Plus tests
            let bPlus = 0, cPlus = 0;
            for (const r of results) {
                const soloR = r.configs.find(cr => cr.config === gen.solo);
                const collabR = r.configs.find(cr => cr.config === collab);
                if (!soloR || !collabR) continue;
                // Base
                if (soloR.execStatus === 'pass' && collabR.execStatus !== 'pass') b++;
                if (soloR.execStatus !== 'pass' && collabR.execStatus === 'pass') c++;
                // Plus
                if (soloR.execStatusPlus === 'pass' && collabR.execStatusPlus !== 'pass') bPlus++;
                if (soloR.execStatusPlus !== 'pass' && collabR.execStatusPlus === 'pass') cPlus++;
            }
            const test = mcNemar(b, c);
            mcnemarResults.push({
                pair: `${CONFIG_SHORT[gen.solo]} vs ${CONFIG_SHORT[collab]} (base)`,
                b, c, chi2: parseFloat(test.chi2.toFixed(2)), pLevel: test.pLevel,
            });
            const testPlus = mcNemar(bPlus, cPlus);
            mcnemarResults.push({
                pair: `${CONFIG_SHORT[gen.solo]} vs ${CONFIG_SHORT[collab]} (plus)`,
                b: bPlus, c: cPlus, chi2: parseFloat(testPlus.chi2.toFixed(2)), pLevel: testPlus.pLevel,
            });
        }
    }

    return {
        passAtK: passAtKResults,
        interJudge: { spearmanR1, cohensKappaR1 },
        bugCatching,
        escalation,
        variance,
        mcnemar: mcnemarResults,
    };
}

// ═══════════════════════════════════════════════════════════════════════
// L. Helpers (PRNG, shuffle, checkpoint)
// ═══════════════════════════════════════════════════════════════════════

function mulberry32(seed: number): () => number {
    return function () {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

let _rng: () => number = Math.random;

function shuffle<T>(arr: readonly T[], rng: () => number = _rng): T[] {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// ─── Checkpoint ───

interface CheckpointData {
    fingerprint: string;
    completedKeys: string[];
    results: TaskResult[];
    lastSaved: string;
}

let CHECKPOINT_PATH = path.join(process.cwd(), 'benchmark-results', 'checkpoint_paper.json');

function makeFingerprint(opts: Args, taskCount: number): string {
    return JSON.stringify({
        configs: opts.configs.sort(),
        gen1Model: opts.gen1Model, gen2Model: opts.gen2Model,
        claudeJudgeModel: opts.claudeJudgeModel, openaiJudgeModel: opts.openaiJudgeModel,
        runs: opts.runs, taskCount,
    });
}

function saveCheckpoint(fingerprint: string, results: TaskResult[]): void {
    const data: CheckpointData = {
        fingerprint,
        completedKeys: results.map(r => `${r.run}:${r.taskId}`),
        results,
        lastSaved: new Date().toISOString(),
    };
    fs.mkdirSync(path.dirname(CHECKPOINT_PATH), { recursive: true });
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
            console.log('  Warning: Checkpoint config mismatch — starting fresh.');
            return null;
        }
        return data;
    } catch {
        console.log('  Warning: Corrupt checkpoint — starting fresh.');
        return null;
    }
}

function deleteCheckpoint(): void {
    try { fs.unlinkSync(CHECKPOINT_PATH); } catch { /* ignore */ }
}

// ═══════════════════════════════════════════════════════════════════════
// M. Token usage aggregation
// ═══════════════════════════════════════════════════════════════════════

function aggregateTokens(): Record<string, { promptTokens: number; completionTokens: number; totalTokens: number }> {
    const agg: Record<string, { promptTokens: number; completionTokens: number; totalTokens: number }> = {};
    for (const t of tokenLog) {
        if (!agg[t.provider]) agg[t.provider] = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };
        agg[t.provider].promptTokens += t.promptTokens;
        agg[t.provider].completionTokens += t.completionTokens;
        agg[t.provider].totalTokens += t.totalTokens;
    }
    return agg;
}

// ═══════════════════════════════════════════════════════════════════════
// N. CLI
// ═══════════════════════════════════════════════════════════════════════

interface Args {
    configs: BenchConfig[];
    limit: number;
    runs: number;
    maxIter: number;
    timeout: number;
    gen1Model: string;
    gen2Model: string;
    claudeJudgeModel: string;
    openaiJudgeModel: string;
    judge1Model: string;
    judge2Model: string;
    tiebreakerModel: string;
    judgeProvider: 'api' | 'ollama';
    tau: number;
    seed: number;
    tasks: string[];
    resume: boolean;
    dryRun: boolean;
    judgeBlind: boolean;
    noDebate: boolean;
    rejudgeFrom: string | null;
    reextract: string | null;
    merge: string | null;
    offset: number;
    checkpointFile: string | null;
    dataset: string;
    noJudge: boolean;
    judgeStrategy: 'scoring' | 'binary';
}

function parseArgs(): Args {
    const args = process.argv.slice(2);
    let configs: BenchConfig[] = [];
    let limit = 30;
    let runs = 3;
    let maxIter = 2;
    let timeout = 600_000;
    let gen1Model = 'qwen3-coder:480b-cloud';
    let gen2Model = 'deepseek-v3.1:671b-cloud';
    let claudeJudgeModel = 'claude-sonnet-4-5-20250929';
    let openaiJudgeModel = 'gpt-4.1';
    let judge1Model = 'kimi-k2.5:cloud';
    let judge2Model = 'minimax-m2.7:cloud';
    let tiebreakerModel = 'glm-5:cloud';
    let judgeProvider: 'api' | 'ollama' = 'api';
    let tau = 2.0;
    let seed = 42;
    let tasks: string[] = [];
    let resume = false;
    let dryRun = false;
    let judgeBlind = true;
    let noDebate = false;
    let rejudgeFrom: string | null = null;
    let reextract: string | null = null;
    let merge: string | null = null;
    let offset = 0;
    let checkpointFile: string | null = null;
    let dataset = 'mbpp';
    let noJudge = false;
    let judgeStrategy: 'scoring' | 'binary' = 'scoring';

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--configs' && args[i + 1]) configs = args[++i].split(',') as BenchConfig[];
        else if (args[i] === '--limit' && args[i + 1]) limit = parseInt(args[++i], 10);
        else if (args[i] === '--runs' && args[i + 1]) runs = parseInt(args[++i], 10);
        else if (args[i] === '--max-iter' && args[i + 1]) maxIter = parseInt(args[++i], 10);
        else if (args[i] === '--timeout' && args[i + 1]) timeout = parseInt(args[++i], 10);
        else if (args[i] === '--gen1-model' && args[i + 1]) gen1Model = args[++i];
        else if (args[i] === '--gen2-model' && args[i + 1]) gen2Model = args[++i];
        else if (args[i] === '--claude-judge-model' && args[i + 1]) claudeJudgeModel = args[++i];
        else if (args[i] === '--openai-judge-model' && args[i + 1]) openaiJudgeModel = args[++i];
        else if (args[i] === '--tiebreaker-model' && args[i + 1]) tiebreakerModel = args[++i];
        else if (args[i] === '--judge1-model' && args[i + 1]) judge1Model = args[++i];
        else if (args[i] === '--judge2-model' && args[i + 1]) judge2Model = args[++i];
        else if (args[i] === '--judge-provider' && args[i + 1]) judgeProvider = args[++i] as 'api' | 'ollama';
        else if (args[i] === '--judge-threshold' && args[i + 1]) tau = parseFloat(args[++i]);
        else if (args[i] === '--seed' && args[i + 1]) seed = parseInt(args[++i], 10);
        else if (args[i] === '--tasks' && args[i + 1]) tasks = args[++i].split(',');
        else if (args[i] === '--resume') resume = true;
        else if (args[i] === '--dry-run') dryRun = true;
        else if (args[i] === '--judge-informed') judgeBlind = false;
        else if (args[i] === '--no-debate') noDebate = true;
        else if (args[i] === '--rejudge-from' && args[i + 1]) rejudgeFrom = args[++i];
        else if (args[i] === '--reextract' && args[i + 1]) reextract = args[++i];
        else if (args[i] === '--merge' && args[i + 1]) merge = args[++i];
        else if (args[i] === '--offset' && args[i + 1]) offset = parseInt(args[++i], 10);
        else if (args[i] === '--checkpoint-file' && args[i + 1]) checkpointFile = args[++i];
        else if (args[i] === '--dataset' && args[i + 1]) dataset = args[++i];
        else if (args[i] === '--no-judge') noJudge = true;
        else if (args[i] === '--judge-strategy' && args[i + 1]) judgeStrategy = args[++i] as 'scoring' | 'binary';
        else if (args[i] === '--help') {
            console.log(`DEBATE EXT — Paper-Grade Benchmark (MBPP+ / EvalPlus)\n`);
            console.log(`Generates code with 14 configs (7 strategies × 2 generators), executes tests,`);
            console.log(`evaluates with 2-judge debate + tie-breaker, computes paper metrics.\n`);
            console.log(`Options:`);
            console.log(`  --configs c1,c2              Select configs (default: all 14)`);
            console.log(`  --limit N                    Max tasks to run (default: 30)`);
            console.log(`  --runs N                     Number of runs (default: 3)`);
            console.log(`  --tasks Mbpp/0,...            Specific task IDs`);
            console.log(`  --max-iter N                 Max debate iterations (default: 2)`);
            console.log(`  --timeout N                  Timeout per call in ms (default: 600000)`);
            console.log(`  --gen1-model MODEL           Gen1 model (default: qwen3-coder:480b-cloud)`);
            console.log(`  --gen2-model MODEL           Gen2 model (default: deepseek-v3.1:671b-cloud)`);
            console.log(`  --claude-judge-model MODEL   Claude judge (default: claude-sonnet-4-5-20250929)`);
            console.log(`  --openai-judge-model MODEL   GPT judge (default: gpt-4.1)`);
            console.log(`  --tiebreaker-model MODEL     Tie-breaker (default: gemini-2.5-pro)`);
            console.log(`  --judge1-model MODEL         Judge 1 for Ollama provider (default: kimi-k2.5:cloud)`);
            console.log(`  --judge2-model MODEL         Judge 2 for Ollama provider (default: gemini-3-flash-preview:cloud)`);
            console.log(`  --judge-provider api|ollama  Judge provider (default: api)`);
            console.log(`  --judge-threshold N          Divergence tau, absolute /10 (default: 2.0)`);
            console.log(`  --seed N                     PRNG seed (default: 42)`);
            console.log(`  --resume                     Resume from last checkpoint`);
            console.log(`  --dry-run                    Simulate run with mock scores, no API calls`);
            console.log(`  --judge-informed             Disable blind mode (show exec status to judges)`);
            console.log(`  --no-debate                  Skip R2 debate + tie-breaker (R1 scores only)`);
            console.log(`  --no-judge                   Skip judge evaluation entirely (generation + exec only)`);
            console.log(`  --judge-strategy scoring|binary  Judge strategy: scoring (default) or binary`);
            console.log(`  --reextract <path.json>       Re-extract code + re-execute tests (fix orphan lines)`);
            console.log(`  --rejudge-from <path.json>   Re-judge from existing report (new judge settings)`);
            console.log(`  --merge <p1.json,p2.json>    Merge multiple reports (dedup by run+taskId)`);
            console.log(`  --offset N                   Skip first N tasks (default: 0)`);
        console.log(`  --checkpoint-file <path>     Custom checkpoint file (for parallel runs)`);
        console.log(`  --dataset mbpp|humaneval     Dataset to use (default: mbpp)`);
            console.log(`\nData setup:`);
            console.log(`  python scripts/setup_mbppplus.py    # creates data/MbppPlus.jsonl`);
            process.exit(0);
        }
    }

    if (configs.length === 0) configs = [...ALL_CONFIGS];
    return {
        configs, limit, runs, maxIter, timeout,
        gen1Model, gen2Model, claudeJudgeModel, openaiJudgeModel,
        tiebreakerModel, tau, seed, tasks, resume, dryRun, judgeBlind, noDebate,
        rejudgeFrom, reextract, merge, offset, checkpointFile, dataset, noJudge, judgeStrategy,
        judge1Model, judge2Model, judgeProvider,
    };
}

// ═══════════════════════════════════════════════════════════════════════
// O. Main
// ═══════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
    const opts = parseArgs();
    tokenLog.length = 0;

    // Custom checkpoint path for parallel runs
    if (opts.checkpointFile) {
        CHECKPOINT_PATH = path.resolve(opts.checkpointFile);
    }

    // ─── --merge mode ───
    if (opts.merge) {
        const paths = opts.merge.split(',').map(p => p.trim());
        console.log(`\nMERGE MODE: combining ${paths.length} reports...`);
        const allMergedResults: TaskResult[] = [];
        const seen = new Set<string>();
        let mergedDataset = '';
        for (const p of paths) {
            if (!fs.existsSync(p)) { console.error(`ERROR: file not found: ${p}`); process.exit(1); }
            const raw: PaperReport = JSON.parse(fs.readFileSync(p, 'utf-8'));
            if (!mergedDataset && raw.meta?.dataset) mergedDataset = raw.meta.dataset;
            for (const t of raw.tasks) {
                const key = `${t.run}:${t.taskId}`;
                if (!seen.has(key)) {
                    seen.add(key);
                    allMergedResults.push(t);
                }
            }
            console.log(`  Loaded ${raw.tasks.length} results from ${path.basename(p)} (${seen.size} unique so far)`);
        }
        console.log(`  Total unique results: ${allMergedResults.length}`);
        const mergedConfigs = [...new Set(allMergedResults.flatMap(r => r.configs.map(c => c.config)))] as BenchConfig[];
        const mergedRuns = new Set(allMergedResults.map(r => r.run)).size;
        const paperMetrics = computePaperMetrics(allMergedResults, mergedConfigs, mergedRuns);
        const passAt1 = computePassAtK(allMergedResults, mergedConfigs, 1);
        const passAt1Plus = computePassAtK(allMergedResults, mergedConfigs, 1, true);
        const avgQuality: Partial<Record<BenchConfig, number>> = {};
        for (const cfg of mergedConfigs) {
            const scores = allMergedResults
                .filter(r => r.judgeAudit.finalScores[cfg])
                .map(r => r.judgeAudit.finalScores[cfg]!.total);
            if (scores.length > 0) {
                avgQuality[cfg] = parseFloat((scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1));
            }
        }
        const costEfficiency: Partial<Record<BenchConfig, { avgCalls: number; qualityPerCall: number; passPerCall: number }>> = {};
        for (const cfg of mergedConfigs) {
            const calls = allMergedResults.flatMap(r => r.configs.filter(c => c.config === cfg).map(c => c.apiCallCount));
            if (calls.length > 0) {
                const avg = calls.reduce((a, b) => a + b, 0) / calls.length;
                const qpc = (avgQuality[cfg] ?? 0) / avg;
                const ppc = (passAt1Plus[cfg] ?? passAt1[cfg] ?? 0) / avg;
                costEfficiency[cfg] = { avgCalls: parseFloat(avg.toFixed(1)), qualityPerCall: parseFloat(qpc.toFixed(2)), passPerCall: parseFloat(ppc.toFixed(2)) };
            }
        }
        const report: PaperReport = {
            meta: {
                date: new Date().toISOString(),
                dataset: (mergedDataset || 'MBPP+ (EvalPlus)') + ' — MERGED',
                gen1Model: 'various', gen2Model: 'various',
                claudeJudgeModel: 'various', openaiJudgeModel: 'various',
                tiebreakerModel: 'various',
                seed: 0, tau: 0,
                totalTasks: new Set(allMergedResults.map(r => r.taskId)).size,
                selectedTasks: new Set(allMergedResults.map(r => r.taskId)).size,
                configs: mergedConfigs,
                runs: mergedRuns,
                judgeBlind: true, noDebate: false,
            },
            tasks: allMergedResults,
            summary: { passAt1, passAt1Plus, avgQuality, costEfficiency, paperMetrics, tokenUsage: {} },
        };
        const resultsDir = path.join(process.cwd(), 'benchmark-results');
        fs.mkdirSync(resultsDir, { recursive: true });
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const reportPath = path.join(resultsDir, `bench_paper_merged_${ts}.json`);
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        console.log(`\n  Merged report saved to: ${reportPath}`);
        console.log(`  pass@1  (base):    ${JSON.stringify(passAt1)}`);
        console.log(`  pass@1+ (strict):  ${JSON.stringify(passAt1Plus)}  ← métrique canonique`);
        process.exit(0);
    }

    // ─── --reextract mode ───
    if (opts.reextract) {
        if (!fs.existsSync(opts.reextract)) {
            console.error(`ERROR: source report not found: ${opts.reextract}`);
            process.exit(1);
        }
        const source: PaperReport = JSON.parse(fs.readFileSync(opts.reextract, 'utf-8'));
        const dataset = source.meta?.dataset?.toLowerCase().includes('humaneval') ? 'humaneval' : 'mbpp';
        const dataFile = dataset === 'humaneval' ? 'HumanEvalPlus.jsonl' : 'MbppPlus.jsonl';
        const dataPath = path.join(__dirname, '..', 'data', dataFile);
        if (!fs.existsSync(dataPath)) {
            console.error(`ERROR: data file not found: ${dataPath}`);
            process.exit(1);
        }
        const taskData = loadMbppPlus(dataPath);
        const taskMap = new Map(taskData.map(t => [t.task_id, t]));

        console.log(`\nREEXTRACT MODE: re-extracting + re-executing ${source.tasks.length} results from ${path.basename(opts.reextract)}`);
        console.log(`  Dataset: ${source.meta?.dataset}  Data file: ${dataFile}`);

        let modified = 0;
        let improved = 0;
        let nameErrorsBefore = 0;
        let nameErrorsAfter = 0;

        for (let ti = 0; ti < source.tasks.length; ti++) {
            const task = source.tasks[ti];
            const data = taskMap.get(task.taskId);
            if (!data) continue;

            for (const cfg of task.configs) {
                const oldCode = cfg.generatedCode;
                const hadNameError = cfg.execOutput.includes('NameError');
                if (hadNameError) nameErrorsBefore++;

                // Re-extract
                const newCode = extractPythonCode(oldCode);
                if (newCode === oldCode) {
                    if (hadNameError) nameErrorsAfter++;
                    continue;
                }

                modified++;
                // Re-execute base tests
                const execBase = await executeCode(newCode, data.test_list, data.test_setup_code);
                cfg.generatedCode = newCode;
                cfg.execStatus = execBase.status;
                cfg.execOutput = execBase.output;
                cfg.execTimeMs = execBase.timeMs;

                // Re-execute plus tests
                if (data.test_list_plus && data.test_list_plus.length > 0) {
                    const execPlus = await executeCode(newCode, data.test_list_plus, data.test_setup_code);
                    cfg.execStatusPlus = execPlus.status;
                    cfg.execOutputPlus = execPlus.output;
                    cfg.execTimeMsPlus = execPlus.timeMs;
                }

                const hasNameErrorNow = cfg.execOutput.includes('NameError');
                if (hasNameErrorNow) nameErrorsAfter++;
                if (hadNameError && !hasNameErrorNow) improved++;
            }

            if ((ti + 1) % 100 === 0) {
                console.log(`  Progress: ${ti + 1}/${source.tasks.length}  modified=${modified}  improved=${improved}`);
            }
        }

        // Recompute summary
        const allConfigs = [...new Set(source.tasks.flatMap(r => r.configs.map(c => c.config)))] as BenchConfig[];
        const allRuns = new Set(source.tasks.map(r => r.run)).size;
        const passAt1 = computePassAtK(source.tasks, allConfigs, 1);
        const passAt1Plus = computePassAtK(source.tasks, allConfigs, 1, true);

        source.meta.dataset = (source.meta.dataset || '') + ' — REEXTRACTED';
        source.summary = { ...source.summary, passAt1, passAt1Plus };

        const resultsDir = path.join(process.cwd(), 'benchmark-results');
        fs.mkdirSync(resultsDir, { recursive: true });
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const reportPath = path.join(resultsDir, `bench_paper_reextracted_${ts}.json`);
        fs.writeFileSync(reportPath, JSON.stringify(source, null, 2));

        console.log(`\nREEXTRACT COMPLETE`);
        console.log(`  Codes modified:    ${modified}`);
        console.log(`  NameErrors before: ${nameErrorsBefore}`);
        console.log(`  NameErrors after:  ${nameErrorsAfter}`);
        console.log(`  Improved:          ${improved}`);
        console.log(`  pass@1  (base):    ${JSON.stringify(passAt1)}`);
        console.log(`  pass@1+ (MBPP+):   ${JSON.stringify(passAt1Plus)}  ← métrique canonique`);
        console.log(`  Report saved to:   ${reportPath}`);
        process.exit(0);
    }

    // ─── --rejudge-from mode ───
    if (opts.rejudgeFrom) {
        if (!fs.existsSync(opts.rejudgeFrom)) {
            console.error(`ERROR: source report not found: ${opts.rejudgeFrom}`);
            process.exit(1);
        }
        const source: PaperReport = JSON.parse(fs.readFileSync(opts.rejudgeFrom, 'utf-8'));
        const sourceTasks = source.tasks;
        console.log(`\nREJUDGE MODE: re-judging ${sourceTasks.length} results from ${path.basename(opts.rejudgeFrom)}`);
        console.log(`  Judge mode  : ${opts.judgeBlind ? 'BLIND' : 'INFORMED'}`);
        console.log(`  Debate      : ${opts.noDebate ? 'OFF' : 'ON'}`);
        console.log(`  Tau         : ${opts.tau}`);

        const judges: JudgeCaller[] = opts.judgeProvider === 'ollama'
            ? [
                { name: 'Judge1', call: (p, t) => callOllamaJudge(p, t, opts.judge1Model) },
                { name: 'Judge2', call: (p, t) => callOllamaJudge(p, t, opts.judge2Model) },
            ]
            : [
                { name: 'Claude', call: (p, t) => callClaude(p, t, opts.claudeJudgeModel) },
                { name: 'GPT', call: (p, t) => callOpenAI(p, t, opts.openaiJudgeModel) },
            ];
        const tbCall = opts.judgeProvider === 'ollama'
            ? (p: string, t: number) => callOllamaJudge(p, t, opts.tiebreakerModel)
            : (p: string, t: number) => callGemini(p, t, opts.tiebreakerModel);
        const rejudgedResults: TaskResult[] = [];
        const rejudgeFingerprint = 'rejudge:' + opts.rejudgeFrom;
        const rejudgeCheckpoint = loadCheckpoint(rejudgeFingerprint);
        const rejudgeCompleted = new Set<string>();
        if (rejudgeCheckpoint) {
            for (const r of rejudgeCheckpoint.results) rejudgedResults.push(r);
            for (const k of rejudgeCheckpoint.completedKeys) rejudgeCompleted.add(k);
            console.log(`  Resuming from checkpoint: ${rejudgedResults.length} already done`);
        }

        let rejudgeConsecFail = 0;
        const MAX_REJUDGE_FAILURES = 3;

        for (let idx = 0; idx < sourceTasks.length; idx++) {
            const srcTask = sourceTasks[idx];
            const taskKey = `${srcTask.run}:${srcTask.taskId}`;
            if (rejudgeCompleted.has(taskKey)) continue;
            console.log(`\n[${idx + 1}/${sourceTasks.length}] Re-judging ${srcTask.taskId} run=${srcTask.run}`);

            const taskPrompt = srcTask.taskPrompt || srcTask.taskId;
            const outputs = srcTask.configs.map(c => ({
                config: c.config, code: c.generatedCode, execStatus: c.execStatus,
            }));

            let judgeAudit: JudgeAudit;
            if (outputs.some(o => o.code.length > 0)) {
                if (opts.judgeStrategy === 'binary') {
                    const entryPoint = srcTask.entryPoint || srcTask.taskId.replace(/.*\//, 'task');
                    judgeAudit = await evaluateWithBinaryStrategy(
                        taskPrompt, entryPoint, outputs,
                        opts.timeout, judges, tbCall,
                    );
                } else {
                    judgeAudit = await evaluateWithPanel(
                        taskPrompt, outputs, opts.timeout, judges,
                        tbCall, opts.tau, opts.judgeBlind, opts.noDebate,
                    );
                }
            } else {
                const empty: Partial<Record<BenchConfig, QualityScores>> = {};
                judgeAudit = {
                    round1: [], divergenceDetected: false, tiebreakerUsed: false,
                    j0: empty, j1: empty, j2: empty, finalScores: empty,
                };
            }

            // Circuit-breaker for rejudge
            const hasScores = Object.keys(judgeAudit.finalScores).length > 0;
            const r1AllFailed = judgeAudit.round1.length === 0;

            if (r1AllFailed || !hasScores) {
                rejudgeConsecFail++;
                console.error(`  \u26a0 Judge failure ${rejudgeConsecFail}/${MAX_REJUDGE_FAILURES} \u2014 task will be retried on --resume`);
                if (rejudgeConsecFail >= MAX_REJUDGE_FAILURES) {
                    console.error('\n\ud83d\uded1 3 judge failures in a row \u2014 likely API quota/payment issue.');
                    console.error('   Saving checkpoint. Fix your API credits and rerun with --resume.\n');
                    saveCheckpoint(rejudgeFingerprint, rejudgedResults);
                    process.exit(2);
                }
                continue;
            }
            rejudgeConsecFail = 0;

            rejudgedResults.push({
                taskId: srcTask.taskId,
                entryPoint: srcTask.entryPoint,
                taskPrompt,
                run: srcTask.run,
                timestamp: new Date().toISOString(),
                configs: srcTask.configs,
                judgeAudit,
            });
            rejudgeCompleted.add(taskKey);
            saveCheckpoint(rejudgeFingerprint, rejudgedResults);
        }

        const allResults = rejudgedResults;
        const mergedConfigs = [...new Set(allResults.flatMap(r => r.configs.map(c => c.config)))] as BenchConfig[];
        const mergedRuns = new Set(allResults.map(r => r.run)).size;
        const paperMetrics = computePaperMetrics(allResults, mergedConfigs, mergedRuns);
        const passAt1 = computePassAtK(allResults, mergedConfigs, 1);
        const passAt1Plus = computePassAtK(allResults, mergedConfigs, 1, true);
        const avgQuality: Partial<Record<BenchConfig, number>> = {};
        for (const cfg of mergedConfigs) {
            const scores = allResults
                .filter(r => r.judgeAudit.finalScores[cfg])
                .map(r => r.judgeAudit.finalScores[cfg]!.total);
            if (scores.length > 0) {
                avgQuality[cfg] = parseFloat((scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1));
            }
        }
        const costEfficiency: Partial<Record<BenchConfig, { avgCalls: number; qualityPerCall: number; passPerCall: number }>> = {};
        for (const cfg of mergedConfigs) {
            const calls = allResults.flatMap(r => r.configs.filter(c => c.config === cfg).map(c => c.apiCallCount));
            if (calls.length > 0) {
                const avg = calls.reduce((a, b) => a + b, 0) / calls.length;
                const qpc = (avgQuality[cfg] ?? 0) / avg;
                const ppc = (passAt1Plus[cfg] ?? passAt1[cfg] ?? 0) / avg;
                costEfficiency[cfg] = { avgCalls: parseFloat(avg.toFixed(1)), qualityPerCall: parseFloat(qpc.toFixed(2)), passPerCall: parseFloat(ppc.toFixed(2)) };
            }
        }
        const tokensAgg = aggregateTokens();

        console.log(`\nREJUDGE COMPLETE — ${allResults.length} results re-judged`);
        for (const cfg of mergedConfigs) {
            if (passAt1[cfg] !== undefined) {
                const displayedPass = passAt1Plus[cfg] ?? passAt1[cfg];
                console.log(`  ${cfg.padEnd(18)} pass@1+=${displayedPass}% quality=${avgQuality[cfg] ?? 'N/A'}/10`);
            }
        }

        const report: PaperReport = {
            meta: {
                date: new Date().toISOString(),
                dataset: (source.meta?.dataset || 'MBPP+ (EvalPlus)') + ' — REJUDGED from ' + path.basename(opts.rejudgeFrom),
                gen1Model: source.meta.gen1Model, gen2Model: source.meta.gen2Model,
                claudeJudgeModel: opts.claudeJudgeModel, openaiJudgeModel: opts.openaiJudgeModel,
                tiebreakerModel: opts.tiebreakerModel,
                seed: source.meta.seed, tau: opts.tau,
                totalTasks: new Set(allResults.map(r => r.taskId)).size,
                selectedTasks: new Set(allResults.map(r => r.taskId)).size,
                configs: mergedConfigs,
                runs: mergedRuns,
                judgeBlind: opts.judgeBlind, noDebate: opts.noDebate,
                judgeStrategy: opts.judgeStrategy,
                judgeProvider: opts.judgeProvider,
                judge1Model: opts.judgeProvider === 'ollama' ? opts.judge1Model : undefined,
                judge2Model: opts.judgeProvider === 'ollama' ? opts.judge2Model : undefined,
            },
            tasks: allResults.map(r => ({
                ...r,
                configs: r.configs.map(c => ({
                    ...c,
                    generatedCode: (c.generatedCode || '').slice(0, 5000),
                    execOutput: (c.execOutput || '').slice(0, 2000),
                })),
            })),
            summary: { passAt1, passAt1Plus, avgQuality, costEfficiency, paperMetrics, tokenUsage: tokensAgg },
        };
        const resultsDir = path.join(process.cwd(), 'benchmark-results');
        fs.mkdirSync(resultsDir, { recursive: true });
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const reportPath = path.join(resultsDir, `bench_paper_rejudge_${ts}.json`);
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        console.log(`  Rejudge report saved to: ${reportPath}`);
        deleteCheckpoint();
        process.exit(0);
    }

    const genModels = { gen1: opts.gen1Model, gen2: opts.gen2Model };
    const judges: JudgeCaller[] = opts.judgeProvider === 'ollama'
        ? [
            { name: 'Judge1', call: (p, t) => callOllamaJudge(p, t, opts.judge1Model) },
            { name: 'Judge2', call: (p, t) => callOllamaJudge(p, t, opts.judge2Model) },
        ]
        : [
            { name: 'Claude', call: (p, t) => callClaude(p, t, opts.claudeJudgeModel) },
            { name: 'GPT', call: (p, t) => callOpenAI(p, t, opts.openaiJudgeModel) },
        ];
    const tbCall = opts.judgeProvider === 'ollama'
        ? (p: string, t: number) => callOllamaJudge(p, t, opts.tiebreakerModel)
        : (p: string, t: number) => callGemini(p, t, opts.tiebreakerModel);
    _rng = mulberry32(opts.seed);

    // Load MBPP+
    const dataFile = opts.dataset === 'humaneval' ? 'HumanEvalPlus.jsonl' : 'MbppPlus.jsonl';
    const dataPath = path.join(__dirname, '..', 'data', dataFile);
    // Auto-separate checkpoint per dataset (prevents cross-contamination)
    if (!opts.checkpointFile && opts.dataset === 'humaneval') {
        CHECKPOINT_PATH = path.join(process.cwd(), 'benchmark-results', 'checkpoint_paper_humaneval.json');
    }
    let allTasks: MbppPlusTask[];

    if (fs.existsSync(dataPath)) {
        allTasks = loadMbppPlus(dataPath);
    } else if (opts.dryRun) {
        // Dry-run without data: generate synthetic task stubs
        const taskPrefix = opts.dataset === 'humaneval' ? 'HumanEval' : 'Mbpp';
        allTasks = Array.from({ length: 50 }, (_, i) => ({
            task_id: `${taskPrefix}/${i + 1}`,
            prompt: `Write a function to compute the ${i + 1}th value in a sequence.`,
            code: `def task_${i + 1}(n):\n    return n * ${i + 1}`,
            test_list: [`assert task_${i + 1}(1) == ${i + 1}`, `assert task_${i + 1}(2) == ${(i + 1) * 2}`],
            test_setup_code: '',
            entry_point: `task_${i + 1}`,
            test_list_plus: [`assert task_${i + 1}(0) == 0`, `assert task_${i + 1}(-1) == -${i + 1}`, `assert task_${i + 1}(100) == ${(i + 1) * 100}`],
        }));
    } else {
        const setupScript = opts.dataset === 'humaneval' ? 'setup_humanevalplus.py' : 'setup_mbppplus.py';
        const dsName = opts.dataset === 'humaneval' ? 'HumanEval+' : 'MBPP+';
        console.error(`ERROR: ${dsName} data not found at ${dataPath}`);
        console.error(`Run: python scripts/${setupScript}`);
        process.exit(1);
    }

    if (opts.tasks.length > 0) {
        allTasks = allTasks.filter(t => opts.tasks.includes(t.task_id));
    }
    const selectedTasks = allTasks.slice(opts.offset, opts.offset + opts.limit);

    const datasetLabel = opts.dataset === 'humaneval' ? 'HumanEval+' : 'MBPP+';
    const cfgCount = opts.configs.length;
    console.log('╔══════════════════════════════════════════════════════════════════════════╗');
    console.log(`║  DEBATE EXT — Paper-Grade Benchmark (${datasetLabel})`.padEnd(75) + '║');
    console.log(`║  ${cfgCount} configs × N tasks × R runs + 2-judge debate + tie-breaker`.padEnd(75) + '║');
    if (opts.dryRun) {
    console.log('║  DRY-RUN MODE — mock scores, no API calls'.padEnd(75) + '║');
    }
    console.log('╚══════════════════════════════════════════════════════════════════════════╝\n');

    const hasAnthropic = !!process.env.ANTHROPIC_API_KEY;
    const hasOpenAI = !!process.env.OPENAI_API_KEY;
    const hasGemini = !!process.env.GEMINI_API_KEY;

    console.log(`  Dataset     : ${opts.dataset === 'humaneval' ? 'HumanEval+ (EvalPlus)' : 'MBPP+ (EvalPlus)'}`);
    console.log(`  Generator 1 : ${opts.gen1Model} (Ollama)`);
    console.log(`  Generator 2 : ${opts.gen2Model} (Ollama)`);
    if (opts.judgeProvider === 'ollama') {
        console.log(`  Judge 1     : ${opts.judge1Model} (Ollama Cloud)`);
        console.log(`  Judge 2     : ${opts.judge2Model} (Ollama Cloud)`);
        console.log(`  Tie-breaker : ${opts.tiebreakerModel} (Ollama Cloud)`);
    } else {
        console.log(`  Judge 1     : ${opts.claudeJudgeModel} (Anthropic) ${hasAnthropic ? 'OK' : 'MISSING'}`);
        console.log(`  Judge 2     : ${opts.openaiJudgeModel} (OpenAI) ${hasOpenAI ? 'OK' : 'MISSING'}`);
        console.log(`  Tie-breaker : ${opts.tiebreakerModel} (Google AI) ${hasGemini ? 'OK' : 'MISSING'}`);
    }
    console.log(`  Seed        : ${opts.seed}`);
    console.log(`  Tau         : ${opts.tau} (absolute /10)`);
    console.log(`  Judge mode  : ${opts.noJudge ? 'OFF (--no-judge)' : opts.judgeBlind ? 'BLIND (default)' : 'INFORMED (ablation)'}`);
    console.log(`  Strategy    : ${opts.judgeStrategy} ${opts.judgeStrategy === 'binary' ? '(B — pass/fail per candidate)' : '(A — scoring absolu)'}`);
    console.log(`  Debate      : ${opts.noJudge ? 'N/A' : opts.noDebate ? 'OFF (R1 only, no escalation)' : 'ON (R1 \u2192 R2 \u2192 TB on divergence)'}`);
    if (opts.offset > 0) console.log(`  Offset      : ${opts.offset} (skip first ${opts.offset} tasks)`);
    console.log(`  Configs     : ${opts.configs.map(c => CONFIG_SHORT[c]).join(', ')}`);
    console.log(`  Tasks       : ${selectedTasks.length} / ${allTasks.length} ${datasetLabel} problems`);
    console.log(`  Runs        : ${opts.runs}`);
    console.log(`  Max iters   : ${opts.maxIter}`);
    console.log(`  Timeout     : ${opts.timeout}ms`);
    console.log(`  Total work  : ${selectedTasks.length} x ${opts.configs.length} configs x ${opts.runs} runs = ${selectedTasks.length * opts.configs.length * opts.runs} generations`);
    if (opts.dryRun) { console.log(`  Mode        : DRY-RUN (mock scores via PRNG, zero API calls)`); }
    console.log('');

    if (!opts.dryRun && !opts.noJudge && opts.judgeProvider === 'api') {
        if (!hasAnthropic) { console.error('ERROR: Set ANTHROPIC_API_KEY.'); process.exit(1); }
        if (!hasOpenAI) { console.error('ERROR: Set OPENAI_API_KEY.'); process.exit(1); }
    }

    // Checkpoint / resume
    const fingerprint = makeFingerprint(opts, selectedTasks.length);
    let allResults: TaskResult[] = [];
    const completedKeys = new Set<string>();

    if (opts.resume) {
        const checkpoint = loadCheckpoint(fingerprint);
        if (checkpoint) {
            allResults = checkpoint.results;
            for (const key of checkpoint.completedKeys) completedKeys.add(key);
            console.log(`  Resumed from checkpoint: ${completedKeys.size} tasks done (saved ${checkpoint.lastSaved})`);
            console.log(`  Remaining: ${selectedTasks.length * opts.runs - completedKeys.size} tasks\n`);
        } else {
            console.log('  No valid checkpoint — starting fresh.\n');
        }
    }

    console.log('═'.repeat(90));

    let consecutiveJudgeFailures = 0;
    const MAX_JUDGE_FAILURES = 3;

    for (let run = 1; run <= opts.runs; run++) {
        console.log(`\n${'█'.repeat(90)}`);
        console.log(`█  RUN ${run}/${opts.runs}`);
        console.log(`${'█'.repeat(90)}\n`);

        for (let ti = 0; ti < selectedTasks.length; ti++) {
            const task = selectedTasks[ti];
            const taskStart = Date.now();
            const taskKey = `${run}:${task.task_id}`;

            if (completedKeys.has(taskKey)) {
                console.log(`  >> [${ti + 1}/${selectedTasks.length}] ${task.task_id} — already done (checkpoint)`);
                continue;
            }

            console.log(`\n>> [${ti + 1}/${selectedTasks.length}] ${task.task_id} — ${task.entry_point}`);
            console.log(`  "${task.prompt.slice(0, 80)}..."\n`);

            const configResults: ConfigRunResult[] = [];
            let judgeAudit: JudgeAudit;

            if (opts.dryRun) {
                // ─── DRY-RUN: mock everything ───
                for (const cfg of opts.configs) {
                    const mockDuration = Math.round(_rng() * 3000 + 500);
                    const mockRand = _rng();
                    const mockStatus: ExecStatus = mockRand > 0.3 ? 'pass' : mockRand > 0.1 ? 'fail' : 'eval_error';
                    configResults.push({
                        config: cfg, generatedCode: `# mock ${cfg}`,
                        execStatus: mockStatus, execOutput: mockStatus === 'pass' ? 'OK' : 'AssertionError',
                        execTimeMs: Math.round(_rng() * 500), duration: mockDuration,
                        apiCallCount: cfg.includes('solo') ? 1 : Math.round(_rng() * 4 + 2),
                        execStatusPlus: mockStatus === 'pass' ? (_rng() > 0.5 ? 'pass' : 'fail') : mockStatus,
                        execOutputPlus: '', execTimeMsPlus: Math.round(_rng() * 500),
                    });
                    console.log(`  ${CONFIG_SHORT[cfg].padEnd(8)} ${mockDuration}ms, ${mockStatus.toUpperCase()} (mock)`);
                }

                // Mock judge evaluation with escalation simulation
                const mockScore = (j: string): QualityScores => {
                    const base = Math.round(_rng() * 5 + 4); // 4–9
                    return {
                        correctness: base, completeness: base, edgeCases: base,
                        codeQuality: base, readability: base, total: base,
                        justification: `mock (${j})`,
                        risk_fail_prob: parseFloat(_rng().toFixed(2)),
                    };
                };

                const r1: JudgeRound[] = [
                    { judge: 'Claude', round: 'R1', scores: {} },
                    { judge: 'GPT', round: 'R1', scores: {} },
                ];
                const j0: Partial<Record<BenchConfig, QualityScores>> = {};
                const j1: Partial<Record<BenchConfig, QualityScores>> = {};
                const j2: Partial<Record<BenchConfig, QualityScores>> = {};
                let diverged = false;
                let tbUsed = false;
                let trigType: 'score_divergence' | 'risk_disagreement' | 'both' | 'none' = 'none';

                for (const cfg of opts.configs) {
                    const s1 = mockScore('Claude'); const s2 = mockScore('GPT');
                    r1[0].scores[cfg] = s1; r1[1].scores[cfg] = s2;
                    const avgR1 = parseFloat(((s1.total + s2.total) / 2).toFixed(1));
                    j0[cfg] = { ...s1, total: avgR1, risk_fail_prob: parseFloat(((s1.risk_fail_prob + s2.risk_fail_prob) / 2).toFixed(3)) };

                    const delta = Math.abs(s1.total - s2.total);
                    const riskDisagree = (s1.risk_fail_prob > 0.5 && s2.risk_fail_prob < 0.5) ||
                                         (s1.risk_fail_prob < 0.5 && s2.risk_fail_prob > 0.5);

                    if (!opts.noDebate && (delta > opts.tau || riskDisagree)) {
                        diverged = true;
                        if (delta > opts.tau && riskDisagree) trigType = 'both';
                        else if (delta > opts.tau) trigType = trigType === 'risk_disagreement' ? 'both' : 'score_divergence';
                        else trigType = trigType === 'score_divergence' ? 'both' : 'risk_disagreement';

                        const s1b = mockScore('Claude-R2'); const s2b = mockScore('GPT-R2');
                        const avgR2 = parseFloat(((s1b.total + s2b.total) / 2).toFixed(1));
                        j1[cfg] = { ...s1b, total: avgR2 };
                        const delta2 = Math.abs(s1b.total - s2b.total);

                        if (delta2 > opts.tau) {
                            tbUsed = true;
                            const sTb = mockScore('Gemini-TB');
                            const avgTb = parseFloat(((s1b.total + s2b.total + sTb.total) / 3).toFixed(1));
                            j2[cfg] = { ...sTb, total: avgTb };
                            console.log(`  ${CONFIG_SHORT[cfg].padEnd(8)} Judge: ${s1.total}/${s2.total} -> DEBATE -> ${s1b.total}/${s2b.total} -> TB -> ${avgTb}/10`);
                        } else {
                            j2[cfg] = j1[cfg];
                            console.log(`  ${CONFIG_SHORT[cfg].padEnd(8)} Judge: ${s1.total}/${s2.total} -> DEBATE -> avg=${avgR2}/10`);
                        }
                    } else {
                        j1[cfg] = j0[cfg];
                        j2[cfg] = j0[cfg];
                        console.log(`  ${CONFIG_SHORT[cfg].padEnd(8)} Judge: ${s1.total}/${s2.total} -> avg=${avgR1}/10`);
                    }
                }

                judgeAudit = {
                    round1: r1, divergenceDetected: diverged,
                    tiebreakerUsed: tbUsed,
                    j0, j1, j2, finalScores: j2,
                };
                if (diverged && trigType !== 'none') {
                    judgeAudit.debate = { trigger: 'mock', triggerType: trigType, messages: [] };
                }
            } else {
                // ─── REAL RUN ───
                for (const cfg of opts.configs) {
                    const cfgColor = cfg.startsWith('gen1') ? '\x1b[36m' : '\x1b[35m';
                    process.stdout.write(`  ${cfgColor}${CONFIG_SHORT[cfg].padEnd(8)}\x1b[0m `);

                    const genStart = Date.now();
                    const gen = await generateForConfig(cfg, task, opts.maxIter, opts.timeout, genModels);
                    const genSec = ((Date.now() - genStart) / 1000).toFixed(0);

                    if (gen.error) {
                        console.log(`\x1b[31mERROR\x1b[0m (${genSec}s): ${gen.error.slice(0, 50)}`);
                        configResults.push({
                            config: cfg, generatedCode: '', execStatus: 'eval_error',
                            execOutput: '', execTimeMs: 0, duration: gen.duration,
                            apiCallCount: gen.apiCallCount,
                            execStatusPlus: 'eval_error', execOutputPlus: '', execTimeMsPlus: 0,
                            error: gen.error,
                        });
                        continue;
                    }

                    const exec = await executeCode(gen.code, task.test_list, task.test_setup_code);
                    const sc = exec.status === 'pass' ? '\x1b[32m' : '\x1b[31m';
                    console.log(`${genSec}s gen, ${sc}${exec.status.toUpperCase()}\x1b[0m (exec: ${exec.timeMs}ms)`);

                    // EvalPlus+ tests: only if base tests pass
                    let execPlus: { status: ExecStatus; output: string; timeMs: number } = { status: exec.status, output: '', timeMs: 0 };
                    if (exec.status === 'pass' && task.test_list_plus.length > 0) {
                        execPlus = await executeCode(gen.code, task.test_list_plus, task.test_setup_code);
                    }

                    configResults.push({
                        config: cfg, generatedCode: gen.code,
                        execStatus: exec.status, execOutput: exec.output,
                        execTimeMs: exec.timeMs, duration: gen.duration,
                        apiCallCount: gen.apiCallCount,
                        execStatusPlus: execPlus.status, execOutputPlus: execPlus.output,
                        execTimeMsPlus: execPlus.timeMs,
                    });
                }

                const withCode = configResults.filter(c => c.generatedCode.length > 0);

                if (opts.noJudge) {
                    const empty: Partial<Record<BenchConfig, QualityScores>> = {};
                    judgeAudit = {
                        round1: [], divergenceDetected: false, tiebreakerUsed: false,
                        j0: empty, j1: empty, j2: empty, finalScores: empty,
                    };
                    console.log('  Judges  SKIPPED (--no-judge)');
                } else if (withCode.length > 0) {
                    process.stdout.write(`  Judges [${opts.judgeStrategy}]  `);
                    if (opts.judgeStrategy === 'binary') {
                        judgeAudit = await evaluateWithBinaryStrategy(
                            task.prompt, task.entry_point,
                            withCode.map(c => ({ config: c.config, code: c.generatedCode, execStatus: c.execStatus })),
                            opts.timeout, judges, tbCall,
                        );
                    } else {
                        judgeAudit = await evaluateWithPanel(
                            task.prompt,
                            withCode.map(c => ({ config: c.config, code: c.generatedCode, execStatus: c.execStatus })),
                            opts.timeout, judges,
                            tbCall, opts.tau, opts.judgeBlind, opts.noDebate,
                        );
                    }

                    const scores = opts.configs
                        .filter(c => judgeAudit.finalScores[c])
                        .map(c => `${CONFIG_SHORT[c]}=${judgeAudit.finalScores[c]!.total}/10`);
                    let suffix = '';
                    if (judgeAudit.divergenceDetected) suffix += ' [DEBATED]';
                    if (judgeAudit.tiebreakerUsed) suffix += ' [TB]';
                    if (judgeAudit.tiebreakerError) suffix += ' [TB-ERR]';
                    console.log(scores.join('  ') + suffix);
                } else {
                    const empty: Partial<Record<BenchConfig, QualityScores>> = {};
                    judgeAudit = {
                        round1: [], divergenceDetected: false, tiebreakerUsed: false,
                        j0: empty, j1: empty, j2: empty, finalScores: empty,
                    };
                    console.log('  Judges  SKIPPED (no code generated)');
                }
            }

            // Circuit-breaker: check if judge produced scores (skip when --no-judge)
            const hasScores = Object.keys(judgeAudit.finalScores).length > 0;
            const r1AllFailed = judgeAudit.round1.length === 0;

            if (opts.noJudge || opts.dryRun) {
                consecutiveJudgeFailures = 0;
            } else if (!r1AllFailed && hasScores) {
                consecutiveJudgeFailures = 0;
            } else if (r1AllFailed || !hasScores) {
                consecutiveJudgeFailures++;
                console.error(`  \u26a0 Judge failure ${consecutiveJudgeFailures}/${MAX_JUDGE_FAILURES} — task will be retried on --resume`);

                // Do NOT add to completedKeys, do NOT save to checkpoint
                // so --resume will retry this task
                if (consecutiveJudgeFailures >= MAX_JUDGE_FAILURES) {
                    console.error('\n\ud83d\uded1 3 judge failures in a row \u2014 likely API quota/payment issue.');
                    console.error('   Saving checkpoint. Fix your API credits and rerun with --resume.\n');
                    saveCheckpoint(fingerprint, allResults);
                    process.exit(2);
                }
                console.log('-'.repeat(70));
                continue;
            }

            allResults.push({
                taskId: task.task_id,
                entryPoint: task.entry_point,
                taskPrompt: task.prompt.slice(0, 500),
                run,
                timestamp: new Date().toISOString(),
                configs: configResults,
                judgeAudit,
            });

            completedKeys.add(taskKey);
            saveCheckpoint(fingerprint, allResults);

            const taskSec = ((Date.now() - taskStart) / 1000).toFixed(0);
            console.log(`  \u23f1 ${taskSec}s total`);
            console.log('-'.repeat(70));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // P. Summary + Report
    // ═══════════════════════════════════════════════════════════════════

    console.log('\n╔══════════════════════════════════════════════════════════════════════════════════╗');
    console.log('║                      PASS@1 RESULTS (base + EvalPlus+)                            ║');
    console.log('╚══════════════════════════════════════════════════════════════════════════════════╝\n');

    const passAt1 = computePassAtK(allResults, opts.configs, 1);
    const passAt1Plus = computePassAtK(allResults, opts.configs, 1, true);
    for (const cfg of opts.configs) {
        if (passAt1[cfg] !== undefined) {
            const base = passAt1[cfg]!;
            const plus = passAt1Plus[cfg] ?? base;
            const delta = parseFloat((plus - base).toFixed(1));
            console.log(`  ${CONFIG_SHORT[cfg].padEnd(10)} ${base}% base  |  ${plus}% plus  (\u0394 = ${delta >= 0 ? '+' : ''}${delta})`);
        }
    }

    console.log('\n╔══════════════════════════════════════════════════════════════════════════════════╗');
    console.log('║                   QUALITY (2 judges + debate + tie-breaker)                      ║');
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

    // Cost-efficiency
    const costEfficiency: Partial<Record<BenchConfig, { avgCalls: number; qualityPerCall: number; passPerCall: number }>> = {};
    console.log('\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557');
    console.log('\u2551                             COST-EFFICIENCY                                     \u2551');
    console.log('\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n');
    for (const cfg of opts.configs) {
        const calls = allResults.flatMap(r => r.configs.filter(c => c.config === cfg).map(c => c.apiCallCount));
        if (calls.length > 0) {
            const avg = calls.reduce((a, b) => a + b, 0) / calls.length;
            const qpc = (avgQuality[cfg] ?? 0) / avg;
            const ppc = (passAt1Plus[cfg] ?? passAt1[cfg] ?? 0) / avg;
            costEfficiency[cfg] = {
                avgCalls: parseFloat(avg.toFixed(1)),
                qualityPerCall: parseFloat(qpc.toFixed(2)),
                passPerCall: parseFloat(ppc.toFixed(1)),
            };
            console.log(`  ${CONFIG_SHORT[cfg].padEnd(10)} avgCalls=${avg.toFixed(1)}  quality/call=${qpc.toFixed(2)}  pass@1/call=${ppc.toFixed(1)}%`);
        }
    }

    // Paper metrics
    const paperMetrics = computePaperMetrics(allResults, opts.configs, opts.runs);

    if (paperMetrics.variance) {
        console.log('\n  Variance (pass@1 across tasks):');
        for (const cfg of opts.configs) {
            const v = paperMetrics.variance[cfg];
            if (v) {
                console.log(`  ${CONFIG_SHORT[cfg].padEnd(10)} ${v.mean.toFixed(1)}% \u00b1 ${v.stdDev.toFixed(1)} (n=${v.n})`);
            }
        }
    }

    if (paperMetrics.mcnemar && paperMetrics.mcnemar.length > 0) {
        console.log('\n  McNemar paired tests:');
        for (const m of paperMetrics.mcnemar) {
            const sig = m.pLevel !== 'n.s.' ? ' *' : '';
            console.log(`    ${m.pair.padEnd(22)} b=${m.b} c=${m.c} \u03c7\u00b2=${m.chi2.toFixed(2)} ${m.pLevel}${sig}`);
        }
    }

    console.log('\n╔══════════════════════════════════════════════════════════════════════════════════╗');
    console.log('║                              PAPER METRICS                                       ║');
    console.log('╚══════════════════════════════════════════════════════════════════════════════════╝\n');

    // pass@k
    for (const [kLabel, values] of Object.entries(paperMetrics.passAtK)) {
        console.log(`  pass@${kLabel.replace('k', '')}:`);
        for (const [cfg, val] of Object.entries(values)) {
            console.log(`    ${CONFIG_SHORT[cfg as BenchConfig]?.padEnd(10) ?? cfg.padEnd(10)} ${val}%`);
        }
    }

    // Inter-judge
    console.log(`\n  Inter-judge agreement (R1):`);
    console.log(`    Spearman rho  : ${paperMetrics.interJudge.spearmanR1 ?? 'N/A'}`);
    console.log(`    Cohen's kappa : ${paperMetrics.interJudge.cohensKappaR1 ?? 'N/A'}`);

    // Bug-catching
    console.log(`\n  Bug-catching (risk_fail_prob vs ground truth):`);
    console.log(`    Recall : ${paperMetrics.bugCatching.recall ?? 'N/A'}`);
    console.log(`    FPR    : ${paperMetrics.bugCatching.fpr ?? 'N/A'}`);
    console.log(`    AUC    : ${paperMetrics.bugCatching.auc ?? 'N/A'}`);

    // Escalation
    const esc = paperMetrics.escalation;
    console.log(`\n  Escalation audit:`);
    console.log(`    Debates triggered  : ${esc.debatesTriggered}/${esc.totalEvaluations} (${(esc.debatesTriggered / Math.max(1, esc.totalEvaluations) * 100).toFixed(1)}%)`);
    console.log(`    Tie-breaker used   : ${esc.tiebreakerUsed}`);
    console.log(`    Tie-breaker errors : ${esc.tiebreakerErrors}`);
    console.log(`    Avg score shift    : ${esc.avgScoreShift ?? 'N/A'}`);
    console.log(`    Triggers: score=${esc.triggerBreakdown.scoreDivergence} risk=${esc.triggerBreakdown.riskDisagreement} both=${esc.triggerBreakdown.both}`);

    // Token usage
    const tokensAgg = aggregateTokens();
    if (Object.keys(tokensAgg).length > 0) {
        console.log(`\n  Token usage:`);
        for (const [provider, usage] of Object.entries(tokensAgg)) {
            console.log(`    ${provider.padEnd(10)} prompt=${usage.promptTokens} completion=${usage.completionTokens} total=${usage.totalTokens}`);
        }
    }

    // ─── Save report JSON ───

    const report: PaperReport = {
        meta: {
            date: new Date().toISOString(),
            dataset: opts.dataset === 'humaneval' ? 'HumanEval+ (EvalPlus)' : 'MBPP+ (EvalPlus)',
            gen1Model: opts.gen1Model,
            gen2Model: opts.gen2Model,
            claudeJudgeModel: opts.claudeJudgeModel,
            openaiJudgeModel: opts.openaiJudgeModel,
            tiebreakerModel: opts.tiebreakerModel,
            seed: opts.seed,
            tau: opts.tau,
            totalTasks: allTasks.length,
            selectedTasks: selectedTasks.length,
            configs: opts.configs,
            runs: opts.runs,
            judgeBlind: opts.judgeBlind,
            noDebate: opts.noDebate,
            noJudge: opts.noJudge,
            judgeStrategy: opts.judgeStrategy,
            judgeProvider: opts.judgeProvider,
            judge1Model: opts.judgeProvider === 'ollama' ? opts.judge1Model : undefined,
            judge2Model: opts.judgeProvider === 'ollama' ? opts.judge2Model : undefined,
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
            passAt1, passAt1Plus, avgQuality, costEfficiency, paperMetrics,
            tokenUsage: tokensAgg,
        },
    };

    const resultsDir = path.join(process.cwd(), 'benchmark-results');
    fs.mkdirSync(resultsDir, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const reportPath = path.join(resultsDir, `bench_paper_${ts}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n  Report saved to: ${reportPath}`);

    deleteCheckpoint();
    console.log(`  Checkpoint cleared (benchmark complete).\n`);
}


// ═══ Test exports (stripped from VSIX by .vscodeignore) ═══
if (typeof module !== 'undefined') {
    module.exports = {
        extractPythonCode,
        passAtK,
        logBinom,
        computeStdDev,
        parseScores,
        detectDivergence,
        spearmanCorrelation,
        cohensKappa,
        mcNemar,
    };
}

// Only run main() when executed directly (not when require'd for tests)
if (require.main === module) {
    main().catch(err => {
        console.error('FATAL:', err);
        process.exit(1);
    });
}
