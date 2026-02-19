import { spawn } from 'child_process';
import * as https from 'https';
import * as vscode from 'vscode';
import { log, logError, showError } from './logger';

export interface CliResult {
    stdout: string;
    stderr: string;
    exitCode: number;
}

export interface CliOptions {
    cwd?: string;
    timeout?: number;
    shell?: boolean;
    stdin?: string;
    env?: Record<string, string>;
}

// ─── SecretStorage (set by extension.ts on activation) ───────────────

let _secrets: vscode.SecretStorage | null = null;

export function setSecretStorage(secrets: vscode.SecretStorage): void {
    _secrets = secrets;
}

// ─── Anthropic API key ───────────────────────────────────────────────

export async function getAnthropicKey(): Promise<string | undefined> {
    return _secrets?.get('debateExt.anthropicApiKey');
}

export async function setAnthropicKey(key: string): Promise<void> {
    await _secrets?.store('debateExt.anthropicApiKey', key);
}

export async function deleteAnthropicKey(): Promise<void> {
    await _secrets?.delete('debateExt.anthropicApiKey');
}

// ─── OpenAI API key ─────────────────────────────────────────────────

export async function getOpenAIKey(): Promise<string | undefined> {
    return _secrets?.get('debateExt.openaiApiKey');
}

export async function setOpenAIKey(key: string): Promise<void> {
    await _secrets?.store('debateExt.openaiApiKey', key);
}

export async function deleteOpenAIKey(): Promise<void> {
    await _secrets?.delete('debateExt.openaiApiKey');
}

// ─── CLI runner (for test commands only) ─────────────────────────────

export function runCommand(
    command: string,
    args: string[],
    options: CliOptions = {}
): Promise<CliResult> {
    return new Promise((resolve, reject) => {
        const timeout = options.timeout ?? 300_000;
        const env = { ...process.env, ...(options.env ?? {}) };
        const useShell = options.shell ?? false;

        const logArgs = args.map(a => a.length > 80 ? a.slice(0, 80) + '...' : a);
        log(`CLI exec: ${command} ${logArgs.join(' ')}${options.cwd ? ' (cwd: ' + options.cwd + ')' : ''}${options.stdin ? ' [stdin]' : ''}`);

        const child = spawn(command, args, {
            cwd: options.cwd,
            env,
            shell: useShell,
            stdio: [options.stdin ? 'pipe' : 'ignore', 'pipe', 'pipe'],
        });

        if (options.stdin && child.stdin) {
            child.stdin.write(options.stdin);
            child.stdin.end();
        }

        let stdout = '';
        let stderr = '';
        let killed = false;

        child.stdout?.on('data', (chunk: Buffer) => { stdout += chunk.toString(); });
        child.stderr?.on('data', (chunk: Buffer) => { stderr += chunk.toString(); });

        const timer = setTimeout(() => {
            killed = true;
            child.kill('SIGTERM');
            setTimeout(() => child.kill('SIGKILL'), 3000);
        }, timeout);

        child.on('close', (code) => {
            clearTimeout(timer);
            if (killed) {
                reject(new Error(`Command timed out after ${timeout}ms: ${command}`));
                return;
            }
            resolve({ stdout, stderr, exitCode: code ?? 1 });
        });

        child.on('error', (err) => {
            clearTimeout(timer);
            reject(err);
        });
    });
}

// ─── Claude API (direct HTTPS) ──────────────────────────────────────

function getClaudeModel(): string {
    return vscode.workspace.getConfiguration('debateExt').get<string>('claudeModel', 'claude-opus-4-6');
}

function getClaudeTimeout(): number {
    return vscode.workspace.getConfiguration('debateExt').get<number>('claudeTimeout', 300_000);
}

export async function callClaude(prompt: string): Promise<CliResult> {
    const apiKey = await getAnthropicKey();
    if (!apiKey) {
        showError('No Anthropic API key. Configure one via the gear icon.');
        throw new Error('No Anthropic API key configured');
    }

    const model = getClaudeModel();
    const timeout = getClaudeTimeout();

    log(`Anthropic API call: model=${model}, prompt=${prompt.slice(0, 80)}...`);

    return new Promise((resolve) => {
        const body = JSON.stringify({
            model,
            max_tokens: 16384,
            messages: [{ role: 'user', content: prompt }],
        });

        const req = https.request({
            hostname: 'api.anthropic.com',
            path: '/v1/messages',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': apiKey,
                'anthropic-version': '2023-06-01',
            },
        }, (res) => {
            let data = '';
            res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    if (json.error) {
                        const errMsg = json.error.message || JSON.stringify(json.error);
                        log(`Anthropic API error: ${errMsg}`);
                        resolve({ stdout: '', stderr: errMsg, exitCode: 1 });
                    } else {
                        const content = json.content
                            ?.filter((b: { type: string }) => b.type === 'text')
                            .map((b: { text: string }) => b.text)
                            .join('') ?? '';
                        log(`Anthropic API OK: ${content.length} chars`);
                        resolve({ stdout: content, stderr: '', exitCode: 0 });
                    }
                } catch {
                    resolve({ stdout: '', stderr: `Invalid JSON: ${data.slice(0, 200)}`, exitCode: 1 });
                }
            });
        });

        req.on('error', (err) => {
            resolve({ stdout: '', stderr: String(err), exitCode: 1 });
        });

        const timer = setTimeout(() => {
            req.destroy();
            resolve({ stdout: '', stderr: `Anthropic API timeout after ${timeout}ms`, exitCode: 1 });
        }, timeout);

        req.on('close', () => clearTimeout(timer));
        req.write(body);
        req.end();
    });
}

// ─── OpenAI API (direct HTTPS) ──────────────────────────────────────

function getOpenAIModel(): string {
    return vscode.workspace.getConfiguration('debateExt').get<string>('openaiModel', 'gpt-5.1');
}

function getOpenAITimeout(): number {
    return vscode.workspace.getConfiguration('debateExt').get<number>('openaiTimeout', 300_000);
}

export async function callOpenAI(prompt: string): Promise<CliResult> {
    const apiKey = await getOpenAIKey();
    if (!apiKey) {
        showError('No OpenAI API key. Configure one via the gear icon.');
        throw new Error('No OpenAI API key configured');
    }

    const model = getOpenAIModel();
    const timeout = getOpenAITimeout();

    log(`OpenAI API call: model=${model}, prompt=${prompt.slice(0, 80)}...`);

    return new Promise((resolve) => {
        const body = JSON.stringify({
            model,
            messages: [{ role: 'user', content: prompt }],
            temperature: 0.2,
        });

        const req = https.request({
            hostname: 'api.openai.com',
            path: '/v1/chat/completions',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
            },
        }, (res) => {
            let data = '';
            res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    if (json.error) {
                        const errMsg = json.error.message || JSON.stringify(json.error);
                        log(`OpenAI API error: ${errMsg}`);
                        resolve({ stdout: '', stderr: errMsg, exitCode: 1 });
                    } else {
                        const content = json.choices?.[0]?.message?.content ?? '';
                        log(`OpenAI API OK: ${content.length} chars`);
                        resolve({ stdout: content, stderr: '', exitCode: 0 });
                    }
                } catch {
                    resolve({ stdout: '', stderr: `Invalid JSON: ${data.slice(0, 200)}`, exitCode: 1 });
                }
            });
        });

        req.on('error', (err) => {
            resolve({ stdout: '', stderr: String(err), exitCode: 1 });
        });

        const timer = setTimeout(() => {
            req.destroy();
            resolve({ stdout: '', stderr: `OpenAI API timeout after ${timeout}ms`, exitCode: 1 });
        }, timeout);

        req.on('close', () => clearTimeout(timer));
        req.write(body);
        req.end();
    });
}

// ─── Test runner ─────────────────────────────────────────────────────

export async function runTests(cwd: string): Promise<CliResult | null> {
    const config = vscode.workspace.getConfiguration('debateExt');
    const testCmd = config.get<string>('testCommand', '').trim();
    if (!testCmd) {
        return null;
    }

    log(`Running tests: ${testCmd}`);
    try {
        const parts = testCmd.split(/\s+/);
        const [cmd, ...args] = parts;
        return await runCommand(cmd, args, { cwd, timeout: 120_000, shell: true });
    } catch (err) {
        logError('Test command failed', err);
        return { stdout: '', stderr: String(err), exitCode: 1 };
    }
}
