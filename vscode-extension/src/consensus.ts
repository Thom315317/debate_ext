import { log, logError } from './logger';

export interface ConsensusOk {
    status: 'OK';
    reason: string;
    checks: string;
}

export interface ConsensusKo {
    status: 'KO';
    issues: string[];
    requiredChanges: string[];
}

export type ConsensusResult = ConsensusOk | ConsensusKo;

/**
 * Build the review prompt that forces the orchestrator to answer in structured format.
 */
export function buildReviewPrompt(
    originalRequest: string,
    codeOutput: string,
    iteration: number
): string {
    return `You are a senior code consultant. Review the following code output produced by the lead developer and provide your professional assessment.

## User Request
${originalRequest}

## Code Output (iteration ${iteration})
${codeOutput}

## Instructions
Respond STRICTLY in one of these two formats (no other text):

CONSENSUS_OK
- reason: <one-line reason why the output is correct and complete>
- checks: <comma-separated list of checks passed>

OR

CONSENSUS_KO
- issues:
  - <issue 1>
  - <issue 2>
- required_changes:
  - <change 1>
  - <change 2>

Do NOT add any other text, greeting, or explanation outside this format.`;
}

/**
 * Parse the structured consensus response from the orchestrator.
 */
export function parseConsensus(raw: string): ConsensusResult {
    const trimmed = raw.trim();
    log(`Parsing consensus response (${trimmed.length} chars)`);

    // Try CONSENSUS_OK
    if (trimmed.includes('CONSENSUS_OK')) {
        const reasonMatch = trimmed.match(/- reason:\s*(.+)/i);
        const checksMatch = trimmed.match(/- checks:\s*(.+)/i);
        return {
            status: 'OK',
            reason: reasonMatch?.[1]?.trim() ?? 'Approved',
            checks: checksMatch?.[1]?.trim() ?? '',
        };
    }

    // Try CONSENSUS_KO
    if (trimmed.includes('CONSENSUS_KO')) {
        const issues = extractListItems(trimmed, 'issues');
        const changes = extractListItems(trimmed, 'required_changes');
        return {
            status: 'KO',
            issues: issues.length > 0 ? issues : ['Unspecified issue'],
            requiredChanges: changes,
        };
    }

    // Fallback: try to guess from content
    logError('Could not parse consensus format, treating as KO');
    return {
        status: 'KO',
        issues: ['Response did not follow expected format'],
        requiredChanges: ['Please re-review'],
    };
}

/**
 * Build correction prompt for Claude based on KO feedback.
 */
export function buildCorrectionPrompt(
    originalRequest: string,
    previousOutput: string,
    feedback: ConsensusKo
): string {
    const issuesList = feedback.issues.map(i => `  - ${i}`).join('\n');
    const changesList = feedback.requiredChanges.map(c => `  - ${c}`).join('\n');

    return `The code review found issues with your previous output. Please fix them.

## Original Request
${originalRequest}

## Issues Found
${issuesList}

## Required Changes
${changesList}

## Your Previous Output
${previousOutput}

Please produce the corrected code as a UNIFIED DIFF that can be applied with \`patch -p1\`. Only output the diff, no explanations.`;
}

// --- internal helpers ---

function extractListItems(text: string, sectionName: string): string[] {
    // Find the section header
    const sectionRegex = new RegExp(`-\\s*${sectionName}:\\s*\\n`, 'i');
    const match = sectionRegex.exec(text);
    if (!match) { return []; }

    const afterSection = text.slice(match.index + match[0].length);
    const items: string[] = [];
    const lines = afterSection.split('\n');

    for (const line of lines) {
        const itemMatch = line.match(/^\s+-\s+(.+)/);
        if (itemMatch) {
            items.push(itemMatch[1].trim());
        } else if (line.trim() && !line.match(/^\s*$/)) {
            // Hit next section or non-list content
            break;
        }
    }

    return items;
}
