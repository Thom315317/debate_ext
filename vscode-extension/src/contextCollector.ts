import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { log } from './logger';
import { runCommand } from './cliRunner';

export interface CollectedContext {
    selection: string;
    openFiles: string[];
    gitModifiedFiles: string[];
    memoryDecisions: string[];
    memorySnippets: string[];
}

/**
 * Collect context from the VS Code environment and local memory.
 */
export async function collectContext(
    workspaceRoot: string,
    restrictFiles?: string[]
): Promise<CollectedContext> {
    const ctx: CollectedContext = {
        selection: '',
        openFiles: [],
        gitModifiedFiles: [],
        memoryDecisions: [],
        memorySnippets: [],
    };

    // Active selection
    const editor = vscode.window.activeTextEditor;
    if (editor && !editor.selection.isEmpty) {
        ctx.selection = editor.document.getText(editor.selection);
    }

    // Open files
    const openDocs = vscode.workspace.textDocuments
        .filter(d => !d.isUntitled && d.uri.scheme === 'file')
        .map(d => vscode.workspace.asRelativePath(d.uri));
    ctx.openFiles = restrictFiles
        ? openDocs.filter(f => restrictFiles.includes(f))
        : openDocs;

    // Git modified files
    try {
        const result = await runCommand('git', ['diff', '--name-only', 'HEAD'], {
            cwd: workspaceRoot,
            timeout: 10_000,
        });
        if (result.exitCode === 0) {
            const files = result.stdout.trim().split('\n').filter(Boolean);
            ctx.gitModifiedFiles = restrictFiles
                ? files.filter(f => restrictFiles.includes(f))
                : files;
        }
    } catch {
        log('Git not available or not a repo — skipping git context');
    }

    // Memory: decisions
    ctx.memoryDecisions = readJsonlEntries(workspaceRoot, 'decisions.jsonl');
    ctx.memorySnippets = readJsonlEntries(workspaceRoot, 'snippets.jsonl');

    log(`Context: selection=${ctx.selection.length}ch, openFiles=${ctx.openFiles.length}, gitMod=${ctx.gitModifiedFiles.length}, decisions=${ctx.memoryDecisions.length}, snippets=${ctx.memorySnippets.length}`);
    return ctx;
}

/**
 * Filter memory entries by keyword relevance to the prompt.
 */
export function filterRelevantMemory(
    entries: string[],
    prompt: string,
    maxEntries: number = 10
): string[] {
    if (entries.length === 0) { return []; }

    const promptWords = new Set(
        prompt.toLowerCase().split(/\W+/).filter(w => w.length > 3)
    );

    const scored = entries.map(entry => {
        const entryLower = entry.toLowerCase();
        let matches = 0;
        for (const w of promptWords) {
            if (entryLower.includes(w)) { matches++; }
        }
        return { entry, matches };
    });

    return scored
        .filter(s => s.matches > 0)
        .sort((a, b) => b.matches - a.matches)
        .slice(0, maxEntries)
        .map(s => s.entry);
}

/**
 * Build a context string for inclusion in prompts.
 */
export function buildContextPrompt(
    ctx: CollectedContext,
    userPrompt: string
): string {
    const parts: string[] = [];

    if (ctx.selection) {
        parts.push(`## Selected Code\n\`\`\`\n${ctx.selection}\n\`\`\``);
    }

    if (ctx.openFiles.length > 0) {
        parts.push(`## Open Files\n${ctx.openFiles.map(f => `- ${f}`).join('\n')}`);
    }

    if (ctx.gitModifiedFiles.length > 0) {
        parts.push(`## Git Modified Files\n${ctx.gitModifiedFiles.map(f => `- ${f}`).join('\n')}`);
    }

    const relevantDecisions = filterRelevantMemory(ctx.memoryDecisions, userPrompt, 5);
    if (relevantDecisions.length > 0) {
        parts.push(`## Previous Decisions (memory)\n${relevantDecisions.map(d => `- ${d}`).join('\n')}`);
    }

    const relevantSnippets = filterRelevantMemory(ctx.memorySnippets, userPrompt, 3);
    if (relevantSnippets.length > 0) {
        parts.push(`## Relevant Snippets (memory)\n${relevantSnippets.join('\n---\n')}`);
    }

    return parts.length > 0 ? parts.join('\n\n') : '';
}

/**
 * Save a decision to the local memory store.
 */
export function saveDecision(workspaceRoot: string, decision: string): void {
    const memDir = path.join(workspaceRoot, '.debate_memory');
    fs.mkdirSync(memDir, { recursive: true });
    const filePath = path.join(memDir, 'decisions.jsonl');
    const line = JSON.stringify({ text: decision, _ts: new Date().toISOString() }) + '\n';
    fs.appendFileSync(filePath, line, 'utf-8');
}

/**
 * Save a code snippet to the local memory store.
 */
export function saveSnippet(workspaceRoot: string, snippet: string, label: string): void {
    const memDir = path.join(workspaceRoot, '.debate_memory');
    fs.mkdirSync(memDir, { recursive: true });
    const filePath = path.join(memDir, 'snippets.jsonl');
    const line = JSON.stringify({ label, text: snippet, _ts: new Date().toISOString() }) + '\n';
    fs.appendFileSync(filePath, line, 'utf-8');
}

// --- internal ---

function readJsonlEntries(workspaceRoot: string, filename: string): string[] {
    const filePath = path.join(workspaceRoot, '.debate_memory', filename);
    if (!fs.existsSync(filePath)) { return []; }

    try {
        const content = fs.readFileSync(filePath, 'utf-8');
        return content
            .split('\n')
            .filter(Boolean)
            .map(line => {
                try {
                    const obj = JSON.parse(line);
                    return obj.text ?? JSON.stringify(obj);
                } catch {
                    return line;
                }
            });
    } catch {
        return [];
    }
}
