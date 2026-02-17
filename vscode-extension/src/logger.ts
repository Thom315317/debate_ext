import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

const OUTPUT_CHANNEL_NAME = 'Debate Orchestrator';

let outputChannel: vscode.OutputChannel | undefined;

export function getOutputChannel(): vscode.OutputChannel {
    if (!outputChannel) {
        outputChannel = vscode.window.createOutputChannel(OUTPUT_CHANNEL_NAME);
    }
    return outputChannel;
}

export function log(message: string): void {
    const timestamp = new Date().toISOString();
    const line = `[${timestamp}] ${message}`;
    getOutputChannel().appendLine(line);
}

export function logError(message: string, error?: unknown): void {
    const errMsg = error instanceof Error ? error.message : String(error ?? '');
    log(`ERROR: ${message}${errMsg ? ' — ' + errMsg : ''}`);
}

export function showInfo(message: string): void {
    log(message);
    vscode.window.showInformationMessage(`Debate Orchestrator: ${message}`);
}

export function showError(message: string): void {
    logError(message);
    vscode.window.showErrorMessage(`Debate Orchestrator: ${message}`);
}

export function showWarning(message: string): void {
    log(`WARN: ${message}`);
    vscode.window.showWarningMessage(`Debate Orchestrator: ${message}`);
}

/** Persist a run log to .debate_memory/runs/ in the workspace */
export function persistRunLog(
    workspaceRoot: string,
    runId: string,
    lines: string[]
): void {
    const runsDir = path.join(workspaceRoot, '.debate_memory', 'runs');
    fs.mkdirSync(runsDir, { recursive: true });
    const logPath = path.join(runsDir, `${runId}.log`);
    fs.writeFileSync(logPath, lines.join('\n'), 'utf-8');
    log(`Run log saved: ${logPath}`);
}

/** Append a JSON line to a .jsonl file inside .debate_memory/ */
export function appendJsonl(
    workspaceRoot: string,
    filename: string,
    data: Record<string, unknown>
): void {
    const memDir = path.join(workspaceRoot, '.debate_memory');
    fs.mkdirSync(memDir, { recursive: true });
    const filePath = path.join(memDir, filename);
    const line = JSON.stringify({ ...data, _ts: new Date().toISOString() }) + '\n';
    fs.appendFileSync(filePath, line, 'utf-8');
}

export function dispose(): void {
    outputChannel?.dispose();
    outputChannel = undefined;
}
