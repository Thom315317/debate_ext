import * as vscode from 'vscode';
import { DebateMode, detectMode, maxIterations, ModeInput } from './modes';
import { collectContext, buildContextPrompt } from './contextCollector';
import { callClaude, callOpenAI, runTests } from './cliRunner';
import { buildReviewPrompt, parseConsensus, buildCorrectionPrompt, ConsensusKo } from './consensus';
import { extractDiff, parseDiff, previewAndConfirm, applyPatch, fallbackPythonPatch } from './patchApplier';
import { log, logError, persistRunLog, appendJsonl } from './logger';
import { CristalChatProvider } from './chatPanel';

/**
 * Run a debate driven from the sidebar chat panel.
 * All progress is shown as chat messages.
 */
export async function runDebateFromChat(
    chat: CristalChatProvider,
    prompt: string,
    forceMode?: DebateMode
): Promise<void> {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        chat.addMessage('error', 'No workspace folder open. Open a folder first.');
        return;
    }

    // Show user message in chat
    chat.addMessage('user', prompt, forceMode ? `Mode: ${forceMode}` : undefined);
    chat.setInputEnabled(false);
    chat.setStatus('Starting...', true);

    const runId = `run_${Date.now()}`;
    const runLog: string[] = [`=== CRISTAL CODE Run ${runId} ===`, `Prompt: ${prompt}`, ''];

    try {
        // --- Collect context ---
        chat.setStatus('Collecting context...', true);
        const ctx = await collectContext(workspaceRoot);
        const contextPrompt = buildContextPrompt(ctx, prompt);

        if (ctx.selection) {
            chat.addMessage('system', `Context: ${ctx.openFiles.length} open files, ${ctx.gitModifiedFiles.length} git-modified, selection (${ctx.selection.length} chars)`);
        }

        // --- Detect mode ---
        const modeInput: ModeInput = {
            prompt,
            fileCount: ctx.openFiles.length + ctx.gitModifiedFiles.length,
            hasTestRequest: /test/i.test(prompt),
            selectionLength: ctx.selection.length,
        };
        const mode = forceMode ?? detectMode(modeInput);
        const maxIter = maxIterations(mode);

        runLog.push(`Mode: ${mode} (max ${maxIter} iterations)`);
        chat.addMessage('status', `Mode: ${mode} — max ${maxIter} iteration(s)`);

        // Build the initial prompt for Claude
        const fullPrompt = buildClaudePrompt(prompt, contextPrompt);
        let claudeOutput = '';
        let iteration = 0;
        let consensusReached = false;

        // --- ITERATION LOOP ---
        for (iteration = 1; iteration <= maxIter; iteration++) {
            // Step 1: Call Claude
            chat.setStatus(`Iteration ${iteration}/${maxIter} — Claude generating...`, true);
            chat.addMessage('status', `── Iteration ${iteration}/${maxIter} ──`);

            const claudePrompt = iteration === 1 ? fullPrompt : claudeOutput;

            try {
                const claudeResult = await callClaude(claudePrompt);

                if (claudeResult.exitCode !== 0) {
                    runLog.push(`Iter ${iteration}: Claude error (exit ${claudeResult.exitCode})`);
                    chat.addMessage('error', `Claude CLI error (exit ${claudeResult.exitCode}):\n${claudeResult.stderr.slice(0, 500)}`);
                    break;
                }

                claudeOutput = claudeResult.stdout;
                runLog.push(`Iter ${iteration}: Claude output (${claudeOutput.length} chars)`);

                // Show truncated output in chat
                const preview = claudeOutput.length > 800
                    ? claudeOutput.slice(0, 800) + '\n\n... (truncated)'
                    : claudeOutput;
                chat.addMessage('claude', preview, `Iteration ${iteration} — ${claudeOutput.length} chars`);

            } catch (err) {
                runLog.push(`Iter ${iteration}: Claude failed: ${err}`);
                chat.addMessage('error', `Claude call failed: ${err instanceof Error ? err.message : String(err)}`);
                break;
            }

            // SIMPLE mode: skip review
            if (mode === 'SIMPLE') {
                consensusReached = true;
                break;
            }

            // Optional: run tests
            const testOutcome = await maybeRunTests(workspaceRoot, runLog, chat);
            if (testOutcome === 'escalate') {
                chat.addMessage('system', 'Tests failed — noted for review.');
            }

            // Step 2: Codex review
            chat.setStatus(`Iteration ${iteration}/${maxIter} — OpenAI consulting...`, true);
            const reviewPrompt = buildReviewPrompt(prompt, claudeOutput, iteration);

            try {
                const openaiResult = await callOpenAI(reviewPrompt);

                if (openaiResult.exitCode !== 0) {
                    runLog.push(`Iter ${iteration}: OpenAI error (exit ${openaiResult.exitCode})`);
                    chat.addMessage('error', `OpenAI review error — proceeding with current output.`);
                    break;
                }

                const consensus = parseConsensus(openaiResult.stdout);
                runLog.push(`Iter ${iteration}: Consensus = ${consensus.status}`);

                if (consensus.status === 'OK') {
                    chat.addMessage('openai', `CONSENSUS OK\nReason: ${consensus.reason}\nChecks: ${consensus.checks}`, `Iteration ${iteration}`);
                    consensusReached = true;
                    break;
                }

                // CONSENSUS_KO
                const ko = consensus as ConsensusKo;
                const issuesText = ko.issues.map(i => `  - ${i}`).join('\n');
                const changesText = ko.requiredChanges.map(c => `  - ${c}`).join('\n');
                chat.addMessage('openai',
                    `CONSENSUS KO\n\nIssues:\n${issuesText}\n\nRequired changes:\n${changesText}`,
                    `Iteration ${iteration}`
                );

                runLog.push(`Issues: ${ko.issues.join('; ')}`);

                if (iteration < maxIter) {
                    // Claude correction round
                    chat.setStatus(`Iteration ${iteration}/${maxIter} — Claude correcting...`, true);
                    const correctionPrompt = buildCorrectionPrompt(prompt, claudeOutput, ko);

                    const corrResult = await callClaude(correctionPrompt);
                    if (corrResult.exitCode !== 0) {
                        chat.addMessage('error', 'Claude correction failed.');
                        runLog.push(`Correction failed at iter ${iteration}`);
                        break;
                    }
                    claudeOutput = corrResult.stdout;
                    runLog.push(`Correction (${claudeOutput.length} chars)`);

                    const corrPreview = claudeOutput.length > 800
                        ? claudeOutput.slice(0, 800) + '\n\n... (truncated)'
                        : claudeOutput;
                    chat.addMessage('claude', corrPreview, `Correction — iteration ${iteration}`);
                } else {
                    chat.addMessage('system', 'Max iterations reached without consensus.');
                }

            } catch (err) {
                logError('OpenAI review failed', err);
                runLog.push(`OpenAI review failed: ${err}`);
                chat.addMessage('error', `OpenAI review failed: ${err instanceof Error ? err.message : String(err)}`);
                break;
            }
        }

        // --- APPLY PATCH ---
        if (!claudeOutput.trim()) {
            chat.addMessage('error', 'No output produced.');
            chat.setStatus('Failed', false);
            return;
        }

        chat.setStatus('Preparing diff...', true);
        let diffText = extractDiff(claudeOutput);
        let patch = parseDiff(diffText);

        // Python fallback if needed
        if (patch.hunks.length === 0) {
            log('No valid hunks — trying Python fallback');
            const extensionRoot = getExtensionRoot();
            if (extensionRoot) {
                const repaired = await fallbackPythonPatch(workspaceRoot, diffText, extensionRoot);
                if (repaired) {
                    diffText = repaired;
                    patch = parseDiff(diffText);
                }
            }
        }

        if (patch.hunks.length === 0) {
            chat.addMessage('system', 'Could not parse a valid diff. Showing raw output.');
            const doc = await vscode.workspace.openTextDocument({ content: claudeOutput, language: 'text' });
            await vscode.window.showTextDocument(doc);
            chat.setStatus('No diff found', false);
            return;
        }

        // Show diff in chat
        const files = [...new Set(patch.hunks.map(h => h.filePath))];
        chat.addMessage('diff', `${patch.hunks.length} hunk(s) across ${files.length} file(s):\n${files.map(f => `  ${f}`).join('\n')}`, 'Review the diff in the editor →');

        const confirmed = await previewAndConfirm(workspaceRoot, patch);
        if (!confirmed) {
            chat.addMessage('system', 'Patch application cancelled.');
            runLog.push('User cancelled patch.');
            chat.setStatus('Cancelled', false);
            return;
        }

        chat.setStatus('Applying patch...', true);
        const applied = await applyPatch(workspaceRoot, patch);

        if (applied) {
            chat.addMessage('system', `Patch applied successfully to ${files.length} file(s).`);
            chat.setStatus('Done', false);
        } else {
            chat.addMessage('error', 'Patch application failed.');
            chat.setStatus('Failed', false);
        }

        runLog.push(applied ? 'Patch applied.' : 'Patch failed.');

        // Save to memory
        appendJsonl(workspaceRoot, 'decisions.jsonl', {
            text: `[${mode}] ${prompt.slice(0, 200)} → ${applied ? 'OK' : 'FAILED'} (${iteration} iters)`,
        });

    } catch (err) {
        logError('Debate failed', err);
        chat.addMessage('error', `Fatal error: ${err instanceof Error ? err.message : String(err)}`);
        chat.setStatus('Error', false);
        runLog.push(`FATAL: ${err}`);
    } finally {
        persistRunLog(workspaceRoot, runId, runLog);
        chat.setInputEnabled(true);
    }
}

function buildClaudePrompt(userPrompt: string, contextPrompt: string): string {
    const parts = [
        'You are a senior developer and technical lead. You orchestrate the approach AND write the code.',
        'First, briefly plan your approach (2-3 lines max), then produce the solution.',
        'Format your answer as a UNIFIED DIFF (patch -p1 format) that can be applied directly.',
        'Include proper file headers (--- a/... +++ b/...) and hunk headers (@@ ... @@).',
        'Do NOT include explanations outside the diff (except the brief plan at the top).',
    ];

    if (contextPrompt) {
        parts.push('', '# Context', contextPrompt);
    }

    parts.push('', '# Task', userPrompt);

    return parts.join('\n');
}

async function maybeRunTests(
    workspaceRoot: string,
    runLog: string[],
    chat: CristalChatProvider
): Promise<'ok' | 'fail' | 'skip' | 'escalate'> {
    const testResult = await runTests(workspaceRoot);
    if (!testResult) {
        return 'skip';
    }

    runLog.push(`Tests: exit=${testResult.exitCode}`);

    if (testResult.exitCode === 0) {
        chat.addMessage('system', 'Tests passed.');
        return 'ok';
    }

    const stderr = testResult.stderr.slice(0, 300);
    chat.addMessage('error', `Tests failed (exit ${testResult.exitCode}):\n${stderr}`);
    return 'escalate';
}

function getWorkspaceRoot(): string | undefined {
    return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

function getExtensionRoot(): string | undefined {
    const outDir = __dirname;
    return require('path').resolve(outDir, '..');
}
