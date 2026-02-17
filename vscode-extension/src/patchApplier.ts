import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { log, logError, showError, showWarning } from './logger';
import { runCommand } from './cliRunner';

export interface PatchHunk {
    filePath: string;
    oldStart: number;
    oldCount: number;
    newStart: number;
    newCount: number;
    lines: string[];
}

export interface ParsedPatch {
    hunks: PatchHunk[];
    raw: string;
}

/**
 * Attempt to extract unified diff from CLI output.
 * The output might contain explanations mixed with diff blocks.
 */
export function extractDiff(output: string): string {
    const lines = output.split('\n');
    const diffLines: string[] = [];
    let inDiff = false;

    for (const line of lines) {
        if (line.startsWith('--- ') || line.startsWith('diff --git')) {
            inDiff = true;
        }
        if (inDiff) {
            diffLines.push(line);
            // End of diff hunk detection: blank line after diff content
            if (line.trim() === '' && diffLines.length > 3) {
                // Keep going, there may be more hunks
            }
        }
    }

    // If no diff found, return the whole output (might be inline code)
    if (diffLines.length === 0) {
        return output;
    }

    return diffLines.join('\n');
}

/**
 * Parse a unified diff string into structured hunks.
 */
export function parseDiff(diff: string): ParsedPatch {
    const hunks: PatchHunk[] = [];
    const lines = diff.split('\n');
    let currentFile = '';
    let currentHunk: PatchHunk | null = null;

    for (const line of lines) {
        // File header (--- a/file or +++ b/file)
        const fileMatch = line.match(/^\+\+\+ (?:b\/)?(.+)/);
        if (fileMatch) {
            currentFile = fileMatch[1].trim();
            continue;
        }

        // Also capture from --- line if +++ not yet seen
        const minusMatch = line.match(/^--- (?:a\/)?(.+)/);
        if (minusMatch && !currentFile) {
            currentFile = minusMatch[1].trim();
            continue;
        }

        // Hunk header
        const hunkMatch = line.match(/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/);
        if (hunkMatch) {
            if (currentHunk) {
                hunks.push(currentHunk);
            }
            currentHunk = {
                filePath: currentFile,
                oldStart: parseInt(hunkMatch[1], 10),
                oldCount: parseInt(hunkMatch[2] ?? '1', 10),
                newStart: parseInt(hunkMatch[3], 10),
                newCount: parseInt(hunkMatch[4] ?? '1', 10),
                lines: [],
            };
            continue;
        }

        // Diff content lines
        if (currentHunk && (line.startsWith('+') || line.startsWith('-') || line.startsWith(' '))) {
            currentHunk.lines.push(line);
        }
    }

    if (currentHunk) {
        hunks.push(currentHunk);
    }

    return { hunks, raw: diff };
}

/**
 * Show a diff preview to the user and ask for confirmation.
 */
export async function previewAndConfirm(
    workspaceRoot: string,
    patch: ParsedPatch
): Promise<boolean> {
    if (patch.hunks.length === 0) {
        showWarning('No valid diff hunks found in the output.');
        return false;
    }

    const files = [...new Set(patch.hunks.map(h => h.filePath))];
    const summary = `Apply changes to ${files.length} file(s)?\n\n${files.map(f => `  • ${f}`).join('\n')}`;

    // Show diff in a temporary document
    const diffDoc = await vscode.workspace.openTextDocument({
        content: patch.raw,
        language: 'diff',
    });
    await vscode.window.showTextDocument(diffDoc, { preview: true, viewColumn: vscode.ViewColumn.Beside });

    const choice = await vscode.window.showInformationMessage(
        summary,
        { modal: true },
        'Apply',
        'Cancel'
    );

    return choice === 'Apply';
}

/**
 * Apply a parsed patch via WorkspaceEdit.
 */
export async function applyPatch(
    workspaceRoot: string,
    patch: ParsedPatch
): Promise<boolean> {
    const edit = new vscode.WorkspaceEdit();
    let applied = 0;

    // Group hunks by file
    const byFile = new Map<string, PatchHunk[]>();
    for (const hunk of patch.hunks) {
        const existing = byFile.get(hunk.filePath) ?? [];
        existing.push(hunk);
        byFile.set(hunk.filePath, existing);
    }

    for (const [filePath, hunks] of byFile) {
        const absPath = path.join(workspaceRoot, filePath);
        const uri = vscode.Uri.file(absPath);

        // Check if file exists
        if (!fs.existsSync(absPath)) {
            // New file: build full content from hunks
            const content = hunks
                .flatMap(h => h.lines)
                .filter(l => l.startsWith('+') || l.startsWith(' '))
                .map(l => l.slice(1))
                .join('\n');
            edit.createFile(uri, { overwrite: false });
            edit.insert(uri, new vscode.Position(0, 0), content);
            applied++;
            continue;
        }

        // Existing file: apply hunks in reverse order (bottom to top)
        const sortedHunks = [...hunks].sort((a, b) => b.oldStart - a.oldStart);

        for (const hunk of sortedHunks) {
            const startLine = Math.max(0, hunk.oldStart - 1);
            const endLine = startLine + hunk.oldCount;

            const newContent = hunk.lines
                .filter(l => l.startsWith('+') || l.startsWith(' '))
                .map(l => l.slice(1))
                .join('\n');

            const range = new vscode.Range(
                new vscode.Position(startLine, 0),
                new vscode.Position(endLine, 0)
            );

            edit.replace(uri, range, newContent + '\n');
        }
        applied++;
    }

    if (applied === 0) {
        showError('No files could be patched.');
        return false;
    }

    const success = await vscode.workspace.applyEdit(edit);
    if (success) {
        log(`Patch applied successfully to ${applied} file(s)`);
    } else {
        logError('WorkspaceEdit.applyEdit returned false');
    }
    return success;
}

/**
 * Fallback: use Python patch_utils.py to try to repair/validate a diff.
 */
export async function fallbackPythonPatch(
    workspaceRoot: string,
    diff: string,
    extensionRoot: string
): Promise<string | null> {
    const config = vscode.workspace.getConfiguration('debateOrchestrator');
    let pythonPath = config.get<string>('pythonPath', '').trim();

    if (!pythonPath) {
        // Auto-detect: look for .venv in the tools directory
        const venvPython = path.join(extensionRoot, '..', 'tools', '.venv', 'bin', 'python3');
        if (fs.existsSync(venvPython)) {
            pythonPath = venvPython;
        } else {
            log('Python venv not found for fallback patch repair');
            return null;
        }
    }

    const scriptPath = path.join(extensionRoot, '..', 'tools', 'scripts', 'patch_utils.py');
    if (!fs.existsSync(scriptPath)) {
        log('patch_utils.py not found');
        return null;
    }

    // Write diff to temp file
    const tmpDir = path.join(workspaceRoot, '.debate_memory', 'tmp');
    fs.mkdirSync(tmpDir, { recursive: true });
    const tmpFile = path.join(tmpDir, `patch_${Date.now()}.diff`);
    fs.writeFileSync(tmpFile, diff, 'utf-8');

    try {
        const result = await runCommand(pythonPath, [scriptPath, 'validate', tmpFile], {
            cwd: workspaceRoot,
            timeout: 15_000,
        });

        if (result.exitCode === 0) {
            log('Python patch validation succeeded');
            return result.stdout.trim() || diff;
        } else {
            log(`Python patch validation failed: ${result.stderr}`);
            return null;
        }
    } catch (err) {
        logError('Python patch fallback error', err);
        return null;
    } finally {
        // Cleanup
        try { fs.unlinkSync(tmpFile); } catch { /* ignore */ }
    }
}
