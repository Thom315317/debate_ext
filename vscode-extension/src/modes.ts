import { log } from './logger';

export type DebateMode = 'SIMPLE' | 'MOYEN' | 'COMPLEXE';

const COMPLEX_KEYWORDS = [
    'refactor', 'migration', 'architecture', 'redesign', 'overhaul',
    'breaking change', 'rewrite', 'multi-module', 'cross-cutting',
    'security', 'authentication', 'database schema',
];

const MEDIUM_KEYWORDS = [
    'test', 'tests', 'fix bug', 'add feature', 'implement',
    'optimize', 'performance', 'update', 'upgrade',
];

export interface ModeInput {
    prompt: string;
    fileCount: number;
    hasTestRequest: boolean;
    selectionLength: number;
}

/**
 * Determine the debate mode based on heuristics.
 */
export function detectMode(input: ModeInput): DebateMode {
    const promptLower = input.prompt.toLowerCase();
    const promptLen = input.prompt.length;

    let score = 0;

    // Prompt length scoring
    if (promptLen > 2000) { score += 3; }
    else if (promptLen > 800) { score += 1; }

    // File count scoring
    if (input.fileCount > 5) { score += 3; }
    else if (input.fileCount > 2) { score += 1; }

    // Keyword scoring
    for (const kw of COMPLEX_KEYWORDS) {
        if (promptLower.includes(kw)) {
            score += 3;
            break;
        }
    }
    for (const kw of MEDIUM_KEYWORDS) {
        if (promptLower.includes(kw)) {
            score += 1;
            break;
        }
    }

    // Test request
    if (input.hasTestRequest) {
        score += 1;
    }

    // Selection length
    if (input.selectionLength > 500) {
        score += 1;
    }

    let mode: DebateMode;
    if (score >= 5) {
        mode = 'COMPLEXE';
    } else if (score >= 2) {
        mode = 'MOYEN';
    } else {
        mode = 'SIMPLE';
    }

    log(`Mode detection: score=${score} → ${mode} (promptLen=${promptLen}, files=${input.fileCount}, tests=${input.hasTestRequest})`);
    return mode;
}

/**
 * Maximum iterations per mode.
 */
export function maxIterations(mode: DebateMode): number {
    switch (mode) {
        case 'SIMPLE': return 1;
        case 'MOYEN': return 2;
        case 'COMPLEXE': return 3;
    }
}
