#!/usr/bin/env node
/**
 * CRISTAL CODE — Benchmark (8 configurations)
 *
 * Generators (mid-tier, via Ollama):
 *   gen1 = Qwen3-Coder 480B    gen2 = MiniMax M2
 *
 * Configs:
 *   gen1-solo       — Gen1 alone
 *   gen2-solo       — Gen2 alone
 *   gen1-lead       — Gen1 codes, Gen2 consults/reviews
 *   gen2-lead       — Gen2 codes, Gen1 consults/reviews
 *   gen1-orch       — Gen1 plans+reviews, Gen2 implements
 *   gen2-orch       — Gen2 plans+reviews, Gen1 implements
 *   gen1-selfrefine — Gen1 self-reviews its own output
 *   gen2-selfrefine — Gen2 self-reviews its own output
 *
 * Judges (frontier, neutral):
 *   Claude Sonnet 4.5 + GPT-4.1 (primary)
 *   DeepSeek-v3.1:671b (tie-breaker via Ollama)
 *
 * Usage:
 *   npm run benchmark                                          # all 100 cases, 8 configs
 *   node out/benchmark.js --configs gen1-solo,gen2-solo        # only solos
 *   node out/benchmark.js --category algorithms                # one category
 *   node out/benchmark.js --cases algo-fibonacci,algo-bfs      # specific cases
 *   node out/benchmark.js --complexity simple                  # filter by difficulty
 *   node out/benchmark.js --max-iter 3 --timeout 300000
 *
 * Requires: ANTHROPIC_API_KEY + OPENAI_API_KEY env vars + Ollama running.
 * No VS Code dependency — runs as a standalone Node.js script.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import * as http from 'http';

// ─── Types ───────────────────────────────────────────────────────────

type Category =
    | 'algorithms'
    | 'data-structures'
    | 'string-text'
    | 'math'
    | 'web-api'
    | 'database'
    | 'frontend'
    | 'concurrency'
    | 'security';

type BenchConfig =
    | 'gen1-solo'       // Gen1 alone
    | 'gen2-solo'       // Gen2 alone
    | 'gen1-lead'       // Gen1 codes, Gen2 consults/reviews
    | 'gen2-lead'       // Gen2 codes, Gen1 consults/reviews
    | 'gen1-orch'       // Gen1 plans+reviews, Gen2 implements
    | 'gen2-orch'       // Gen2 plans+reviews, Gen1 implements
    | 'gen1-selfrefine' // Gen1 self-reviews its own output
    | 'gen2-selfrefine'; // Gen2 self-reviews its own output

const ALL_CONFIGS: BenchConfig[] = [
    'gen1-solo', 'gen2-solo',
    'gen1-lead', 'gen2-lead',
    'gen1-orch', 'gen2-orch',
    'gen1-selfrefine', 'gen2-selfrefine',
];

const CONFIG_SHORT: Record<BenchConfig, string> = {
    'gen1-solo': 'QC.Solo',
    'gen2-solo': 'MM.Solo',
    'gen1-lead': 'QC.Lead',
    'gen2-lead': 'MM.Lead',
    'gen1-orch': 'QC.Orch',
    'gen2-orch': 'MM.Orch',
    'gen1-selfrefine': 'QC.SRef',
    'gen2-selfrefine': 'MM.SRef',
};

type Agent = 'gen1' | 'gen2';

interface BenchmarkCase {
    name: string;
    prompt: string;
    description: string;
    complexity: 'simple' | 'medium' | 'complex';
    category: Category;
}

interface ConfigResult {
    config: BenchConfig;
    duration: number;
    outputChars: number;
    outputLines: number;
    iterations: number;
    consensusReached: boolean;
    output: string;
    error?: string;
}

interface QualityScores {
    correctness: number;    // 1-10
    completeness: number;   // 1-10
    edgeCases: number;      // 1-10
    codeQuality: number;    // 1-10
    readability: number;    // 1-10
    total: number;          // avg
    justification: string;
}

interface JudgeDetail {
    judge: string;
    scores: Partial<Record<BenchConfig, QualityScores>>;
}

interface BenchmarkResult {
    caseName: string;
    category: string;
    complexity: string;
    timestamp: string;
    configs: ConfigResult[];
    quality?: Partial<Record<BenchConfig, QualityScores>>;
    judgeDetails?: JudgeDetail[];
}

interface CategoryStats {
    category: string;
    count: number;
    avgMs: Partial<Record<BenchConfig, number>>;
    avgQuality: Partial<Record<BenchConfig, number>>;
    bestConfig: string;
}

interface BenchmarkReport {
    meta: {
        date: string;
        cwd: string;
        maxIterations: number;
        timeoutMs: number;
        gen1Model: string;
        gen2Model: string;
        claudeJudgeModel: string;
        openaiJudgeModel: string;
        tiebreakerModel: string;
        seed: number;
        totalAvailableCases: number;
        selectedCases: number;
        selectedConfigs: BenchConfig[];
    };
    cases: BenchmarkResult[];
    summary: {
        totalCases: number;
        successfulCases: number;
        avgMs: Partial<Record<BenchConfig, number>>;
        avgQuality: Partial<Record<BenchConfig, number>>;
        bestConfigCount: Partial<Record<BenchConfig, number>>;
        byCategory: CategoryStats[];
    };
}

// ─── Helpers ─────────────────────────────────────────────────────────

function countLines(s: string): number {
    return s.split('\n').filter(l => l.trim()).length;
}

// ─── Claude API (direct HTTPS, requires ANTHROPIC_API_KEY) ──────────

function callClaude(
    prompt: string,
    timeoutMs: number,
    model: string
): Promise<{ content: string; error?: string }> {
    const apiKey = process.env.ANTHROPIC_API_KEY!;
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
                        resolve({ content: '', error: json.error.message || JSON.stringify(json.error) });
                    } else {
                        const content = json.content
                            ?.filter((b: { type: string }) => b.type === 'text')
                            .map((b: { text: string }) => b.text)
                            .join('') ?? '';
                        resolve({ content });
                    }
                } catch {
                    resolve({ content: '', error: `Invalid JSON: ${data.slice(0, 200)}` });
                }
            });
        });

        req.on('error', (err) => {
            resolve({ content: '', error: String(err) });
        });

        const timer = setTimeout(() => {
            req.destroy();
            resolve({ content: '', error: `Anthropic API timeout after ${timeoutMs}ms` });
        }, timeoutMs);

        req.on('close', () => clearTimeout(timer));
        req.write(body);
        req.end();
    });
}

// ─── OpenAI API (direct HTTPS, requires OPENAI_API_KEY) ─────────────

function callOpenAI(
    prompt: string,
    timeoutMs: number,
    model: string
): Promise<{ content: string; error?: string }> {
    const apiKey = process.env.OPENAI_API_KEY!;
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
                        resolve({ content: '', error: json.error.message || JSON.stringify(json.error) });
                    } else {
                        const content = json.choices?.[0]?.message?.content ?? '';
                        resolve({ content });
                    }
                } catch {
                    resolve({ content: '', error: `Invalid JSON: ${data.slice(0, 200)}` });
                }
            });
        });

        req.on('error', (err) => {
            resolve({ content: '', error: String(err) });
        });

        const timer = setTimeout(() => {
            req.destroy();
            resolve({ content: '', error: `OpenAI API timeout after ${timeoutMs}ms` });
        }, timeoutMs);

        req.on('close', () => clearTimeout(timer));
        req.write(body);
        req.end();
    });
}

// ─── Ollama cloud model (DeepSeek via local Ollama) ─────────────────

const OLLAMA_HOST = process.env.OLLAMA_HOST ?? 'host.docker.internal';
const OLLAMA_PORT = parseInt(process.env.OLLAMA_PORT ?? '11434', 10);

function callOllama(
    prompt: string,
    timeoutMs: number,
    model: string
): Promise<{ content: string; error?: string }> {
    return new Promise((resolve) => {
        const body = JSON.stringify({
            model,
            messages: [{ role: 'user', content: prompt }],
            temperature: 0.2,
            stream: false,
        });

        const req = http.request({
            hostname: OLLAMA_HOST,
            port: OLLAMA_PORT,
            path: '/v1/chat/completions',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
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
                        const content = json.choices?.[0]?.message?.content ?? '';
                        resolve({ content });
                    }
                } catch {
                    resolve({ content: '', error: `Invalid JSON: ${data.slice(0, 200)}` });
                }
            });
        });

        req.on('error', (err) => {
            resolve({ content: '', error: String(err) });
        });

        const timer = setTimeout(() => {
            req.destroy();
            resolve({ content: '', error: `Ollama timeout after ${timeoutMs}ms` });
        }, timeoutMs);

        req.on('close', () => clearTimeout(timer));
        req.write(body);
        req.end();
    });
}

// ─── Gemini API (tie-breaker, requires GEMINI_API_KEY) ──────────────

function callGemini(
    prompt: string,
    timeoutMs: number,
    model: string
): Promise<{ content: string; error?: string }> {
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
                        resolve({ content });
                    }
                } catch {
                    resolve({ content: '', error: `Invalid JSON: ${data.slice(0, 200)}` });
                }
            });
        });

        req.on('error', (err) => {
            resolve({ content: '', error: String(err) });
        });

        const timer = setTimeout(() => {
            req.destroy();
            resolve({ content: '', error: `Gemini API timeout after ${timeoutMs}ms` });
        }, timeoutMs);

        req.on('close', () => clearTimeout(timer));
        req.write(body);
        req.end();
    });
}

// ─── 100 benchmark cases (inspired by HumanEval, MBPP, APPS, SWE-bench) ─────
// Distribution: 25 algorithms, 15 data structures, 12 string/text, 10 math,
//               12 web/API, 8 database, 8 frontend, 5 concurrency, 5 security
// Difficulty: ~30 simple, ~45 medium, ~25 complex

const ALL_CASES: BenchmarkCase[] = [

    // ═══════════════════════════════════════════════════════════════════
    // ALGORITHMS (25)
    // ═══════════════════════════════════════════════════════════════════

    {
        name: 'algo-fibonacci',
        prompt: 'Write a Python function that returns the nth Fibonacci number using memoization. Handle edge cases: negative input, n=0, n=1. Include type hints.',
        description: 'Fibonacci with memoization',
        complexity: 'simple',
        category: 'algorithms',
    },
    {
        name: 'algo-binary-search',
        prompt: 'Implement binary search in TypeScript with generics. The function should accept a sorted array and a comparator function. Return the index or -1. Include JSDoc.',
        description: 'Generic binary search',
        complexity: 'simple',
        category: 'algorithms',
    },
    {
        name: 'algo-merge-sort',
        prompt: 'Write a Python implementation of merge sort that works on a list of any comparable type. Include both the recursive merge sort and the merge helper. Add type hints and handle empty lists.',
        description: 'Merge sort implementation',
        complexity: 'simple',
        category: 'algorithms',
    },
    {
        name: 'algo-quicksort',
        prompt: 'Implement quicksort in TypeScript with a randomized pivot selection strategy. Handle duplicates efficiently (3-way partition). The function should sort in-place and return the array.',
        description: 'Quicksort with 3-way partition',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-bfs',
        prompt: 'Write a Python function for breadth-first search on an adjacency list graph. Return the shortest path between two nodes as a list of nodes. Handle disconnected graphs by returning an empty list.',
        description: 'BFS shortest path',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-dfs',
        prompt: 'Implement depth-first search in TypeScript that detects cycles in a directed graph represented as an adjacency list (Map<string, string[]>). Return true if the graph contains a cycle.',
        description: 'DFS cycle detection',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-dijkstra',
        prompt: 'Implement Dijkstra\'s shortest path algorithm in Python using a min-heap. The graph is represented as a dict of dicts: graph[u][v] = weight. Return both the distances and the path reconstruction.',
        description: 'Dijkstra with path reconstruction',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-dp-knapsack',
        prompt: 'Solve the 0/1 knapsack problem in Python using dynamic programming. Given items with weights and values, and a capacity, return the maximum value and the list of selected items. Use bottom-up DP.',
        description: '0/1 Knapsack with item tracking',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-dp-lcs',
        prompt: 'Write a TypeScript function that finds the longest common subsequence of two strings. Return both the length and the actual subsequence string. Use dynamic programming.',
        description: 'Longest Common Subsequence',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-dp-edit-distance',
        prompt: 'Implement the edit distance (Levenshtein distance) algorithm in Python. Return the minimum number of operations (insert, delete, replace) and also reconstruct the sequence of operations performed.',
        description: 'Edit distance with operations',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-topological-sort',
        prompt: 'Write a Python function for topological sort using Kahn\'s algorithm (BFS-based). The input is a list of edges. Detect if the graph has a cycle and raise an error. Return the sorted order.',
        description: 'Topological sort (Kahn)',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-a-star',
        prompt: 'Implement the A* pathfinding algorithm in Python for a 2D grid. Cells can be passable (0) or blocked (1). Use Manhattan distance as heuristic. Return the shortest path as a list of (row, col) tuples.',
        description: 'A* pathfinding on grid',
        complexity: 'complex',
        category: 'algorithms',
    },
    {
        name: 'algo-trie',
        prompt: 'Implement a Trie (prefix tree) in TypeScript with insert, search, startsWith, and delete methods. Include a method to return all words with a given prefix. Use a class-based approach.',
        description: 'Trie with prefix search',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-union-find',
        prompt: 'Implement a Union-Find (disjoint set) data structure in Python with path compression and union by rank. Include methods: find, union, connected, and count_components.',
        description: 'Union-Find with optimizations',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-backtrack-sudoku',
        prompt: 'Write a Python Sudoku solver using backtracking. The board is a 9x9 list of lists (0 = empty). Solve it in-place and return True if solvable. Use constraint propagation to prune the search space.',
        description: 'Sudoku solver with pruning',
        complexity: 'complex',
        category: 'algorithms',
    },
    {
        name: 'algo-backtrack-nqueens',
        prompt: 'Solve the N-Queens problem in Python. Return all valid board configurations for a given N. Each configuration should be a list of column positions. Handle N from 1 to 12.',
        description: 'N-Queens all solutions',
        complexity: 'complex',
        category: 'algorithms',
    },
    {
        name: 'algo-sliding-window',
        prompt: 'Write a TypeScript function that finds the longest substring without repeating characters using the sliding window technique. Return both the length and the substring itself.',
        description: 'Sliding window - longest unique substring',
        complexity: 'simple',
        category: 'algorithms',
    },
    {
        name: 'algo-two-pointers',
        prompt: 'Write a Python function that, given a sorted array and a target sum, finds all unique pairs that sum to the target. Use the two-pointer technique. Handle duplicates correctly.',
        description: 'Two-pointer pair sum',
        complexity: 'simple',
        category: 'algorithms',
    },
    {
        name: 'algo-kadane',
        prompt: 'Implement Kadane\'s algorithm in TypeScript to find the maximum subarray sum. Also return the start and end indices of the subarray. Handle all-negative arrays correctly.',
        description: 'Maximum subarray with indices',
        complexity: 'simple',
        category: 'algorithms',
    },
    {
        name: 'algo-interval-merge',
        prompt: 'Write a Python function that merges overlapping intervals. Input: list of [start, end] pairs. Return merged intervals sorted by start. Handle edge cases: empty list, single interval, touching intervals.',
        description: 'Merge overlapping intervals',
        complexity: 'simple',
        category: 'algorithms',
    },
    {
        name: 'algo-kruskal',
        prompt: 'Implement Kruskal\'s minimum spanning tree algorithm in Python. Use Union-Find for cycle detection. Input: number of vertices and list of (u, v, weight) edges. Return the MST edges and total weight.',
        description: 'Kruskal MST',
        complexity: 'complex',
        category: 'algorithms',
    },
    {
        name: 'algo-rabin-karp',
        prompt: 'Implement the Rabin-Karp string matching algorithm in Python. Return all starting indices where the pattern occurs in the text. Use rolling hash. Handle the case where no match is found.',
        description: 'Rabin-Karp string matching',
        complexity: 'complex',
        category: 'algorithms',
    },
    {
        name: 'algo-lru-cache',
        prompt: 'Implement an LRU Cache in TypeScript with O(1) get and put operations. Use a Map for the hash table and a doubly-linked list for ordering. Support a configurable capacity. Include a class-based API.',
        description: 'LRU Cache O(1)',
        complexity: 'complex',
        category: 'algorithms',
    },
    {
        name: 'algo-heap',
        prompt: 'Implement a min-heap in Python from scratch (no heapq). Support: insert, extract_min, peek, heapify from array, and decrease_key. Include sift_up and sift_down helpers.',
        description: 'Min-heap from scratch',
        complexity: 'medium',
        category: 'algorithms',
    },
    {
        name: 'algo-reservoir-sampling',
        prompt: 'Implement reservoir sampling in Python to select k random items from a stream of unknown length. Prove that each element has equal probability k/n of being selected. Include type hints.',
        description: 'Reservoir sampling',
        complexity: 'medium',
        category: 'algorithms',
    },

    // ═══════════════════════════════════════════════════════════════════
    // DATA STRUCTURES (15)
    // ═══════════════════════════════════════════════════════════════════

    {
        name: 'ds-linked-list',
        prompt: 'Implement a singly linked list in TypeScript with: append, prepend, delete(value), find(value), reverse, toArray, and a static fromArray factory. Use generics.',
        description: 'Singly linked list with generics',
        complexity: 'simple',
        category: 'data-structures',
    },
    {
        name: 'ds-doubly-linked-list',
        prompt: 'Implement a doubly linked list in Python with: append, prepend, delete_node, insert_after, reverse, iterate_forward, iterate_backward. Include __iter__ and __len__ dunder methods.',
        description: 'Doubly linked list',
        complexity: 'medium',
        category: 'data-structures',
    },
    {
        name: 'ds-stack',
        prompt: 'Implement a stack in TypeScript using a linked list (not an array). Support: push, pop, peek, isEmpty, size, and clear. Include a method to iterate over elements from top to bottom.',
        description: 'Stack via linked list',
        complexity: 'simple',
        category: 'data-structures',
    },
    {
        name: 'ds-queue',
        prompt: 'Implement a circular queue (ring buffer) in Python with a fixed capacity. Support: enqueue, dequeue, peek, is_full, is_empty, size. Raise an exception on overflow. Include __repr__.',
        description: 'Circular queue / ring buffer',
        complexity: 'medium',
        category: 'data-structures',
    },
    {
        name: 'ds-bst',
        prompt: 'Implement a binary search tree in TypeScript with: insert, delete, find, findMin, findMax, inOrderTraversal, and isValidBST. Use a class with generic type parameter.',
        description: 'Binary search tree',
        complexity: 'medium',
        category: 'data-structures',
    },
    {
        name: 'ds-avl-tree',
        prompt: 'Implement an AVL tree in Python with automatic rebalancing. Support insert and delete with proper rotation (left, right, left-right, right-left). Include a method to print the tree level by level.',
        description: 'AVL tree with rotations',
        complexity: 'complex',
        category: 'data-structures',
    },
    {
        name: 'ds-hash-map',
        prompt: 'Implement a hash map in TypeScript from scratch using separate chaining. Support: set, get, delete, has, keys, values, entries, size. Handle resizing when load factor exceeds 0.75.',
        description: 'Hash map with resizing',
        complexity: 'complex',
        category: 'data-structures',
    },
    {
        name: 'ds-priority-queue',
        prompt: 'Implement a generic priority queue in Python using a binary heap. Support: enqueue(item, priority), dequeue, peek, update_priority, and is_empty. Items with lower priority numbers are dequeued first.',
        description: 'Priority queue with update',
        complexity: 'medium',
        category: 'data-structures',
    },
    {
        name: 'ds-graph',
        prompt: 'Implement a Graph class in TypeScript supporting both directed and undirected modes. Support: addVertex, addEdge, removeVertex, removeEdge, getNeighbors, hasPath (BFS), and toString.',
        description: 'Graph class (directed/undirected)',
        complexity: 'medium',
        category: 'data-structures',
    },
    {
        name: 'ds-deque',
        prompt: 'Implement a double-ended queue (deque) in Python using a doubly linked list. Support: push_front, push_back, pop_front, pop_back, peek_front, peek_back, size, and __iter__.',
        description: 'Deque via doubly linked list',
        complexity: 'medium',
        category: 'data-structures',
    },
    {
        name: 'ds-bloom-filter',
        prompt: 'Implement a Bloom filter in TypeScript. Support: add(item), mightContain(item), and estimatedFalsePositiveRate. Use multiple hash functions (at least 3). Accept configurable size and hash count.',
        description: 'Bloom filter',
        complexity: 'complex',
        category: 'data-structures',
    },
    {
        name: 'ds-skip-list',
        prompt: 'Implement a skip list in Python with insert, search, and delete operations. Use randomized level generation. Include a method to visualize the skip list levels as ASCII art.',
        description: 'Skip list',
        complexity: 'complex',
        category: 'data-structures',
    },
    {
        name: 'ds-segment-tree',
        prompt: 'Implement a segment tree in Python for range sum queries and point updates. Support: build, update(index, value), query_sum(left, right). Handle edge cases. Include both iterative and recursive approaches.',
        description: 'Segment tree for range queries',
        complexity: 'complex',
        category: 'data-structures',
    },
    {
        name: 'ds-lfu-cache',
        prompt: 'Implement an LFU (Least Frequently Used) Cache in TypeScript with O(1) get and put. When capacity is exceeded, evict the least frequently used item. Break ties by least recently used.',
        description: 'LFU Cache O(1)',
        complexity: 'complex',
        category: 'data-structures',
    },
    {
        name: 'ds-immutable-list',
        prompt: 'Implement a persistent (immutable) linked list in TypeScript. Operations like prepend, append, drop, take, map, filter should return new lists sharing structure with the original. Use generics.',
        description: 'Persistent immutable list',
        complexity: 'complex',
        category: 'data-structures',
    },

    // ═══════════════════════════════════════════════════════════════════
    // STRING / TEXT PROCESSING (12)
    // ═══════════════════════════════════════════════════════════════════

    {
        name: 'str-palindrome',
        prompt: 'Write a Python function that finds the longest palindromic substring in a given string. Use Manacher\'s algorithm or expand-around-center. Return the substring itself.',
        description: 'Longest palindromic substring',
        complexity: 'medium',
        category: 'string-text',
    },
    {
        name: 'str-anagram-group',
        prompt: 'Write a TypeScript function that groups anagrams together from an array of strings. Return a Map<string, string[]> where the key is the sorted characters. Handle empty strings and case sensitivity.',
        description: 'Group anagrams',
        complexity: 'simple',
        category: 'string-text',
    },
    {
        name: 'str-regex-email',
        prompt: 'Write a Python function that validates email addresses using regex following RFC 5322 simplified rules. Return a tuple of (is_valid: bool, parts: dict) where parts includes local, domain, and tld.',
        description: 'Email validation with regex',
        complexity: 'medium',
        category: 'string-text',
    },
    {
        name: 'str-csv-parser',
        prompt: 'Implement a CSV parser in TypeScript that handles: quoted fields, escaped quotes, newlines within quotes, custom delimiters, and headers. Return an array of objects keyed by header names.',
        description: 'CSV parser with edge cases',
        complexity: 'complex',
        category: 'string-text',
    },
    {
        name: 'str-markdown-to-html',
        prompt: 'Write a Python function that converts a subset of Markdown to HTML. Support: headings (h1-h6), bold, italic, links, unordered lists, code blocks (backtick), and paragraphs.',
        description: 'Markdown to HTML converter',
        complexity: 'complex',
        category: 'string-text',
    },
    {
        name: 'str-json-parser',
        prompt: 'Implement a simple JSON parser in TypeScript from scratch (no JSON.parse). Support: strings, numbers, booleans, null, arrays, and objects. Throw descriptive errors for invalid JSON with line/column info.',
        description: 'JSON parser from scratch',
        complexity: 'complex',
        category: 'string-text',
    },
    {
        name: 'str-template-engine',
        prompt: 'Write a Python template engine that supports variable substitution ({{var}}), conditionals ({% if %}...{% endif %}), and loops ({% for item in list %}...{% endfor %}). Return the rendered string.',
        description: 'Simple template engine',
        complexity: 'complex',
        category: 'string-text',
    },
    {
        name: 'str-text-wrap',
        prompt: 'Write a TypeScript function that wraps text to a specified line width. Respect word boundaries (never break mid-word unless the word exceeds the line width). Handle existing newlines. Optionally support indentation.',
        description: 'Word-wrap text formatter',
        complexity: 'simple',
        category: 'string-text',
    },
    {
        name: 'str-diff',
        prompt: 'Implement a simple text diff function in Python that compares two strings line-by-line and outputs a unified diff format. Show added lines with +, removed with -, and context lines.',
        description: 'Unified diff generator',
        complexity: 'medium',
        category: 'string-text',
    },
    {
        name: 'str-slug',
        prompt: 'Write a TypeScript function that converts any string to a URL-friendly slug. Handle: unicode/accented characters (normalize to ASCII), multiple spaces/dashes, leading/trailing whitespace, and special characters.',
        description: 'URL slug generator',
        complexity: 'simple',
        category: 'string-text',
    },
    {
        name: 'str-compression',
        prompt: 'Implement run-length encoding and decoding in Python. encode("aaabbcccc") returns "3a2b4c". decode("3a2b4c") returns "aaabbcccc". Handle single characters and edge cases.',
        description: 'Run-length encoding/decoding',
        complexity: 'simple',
        category: 'string-text',
    },
    {
        name: 'str-tokenizer',
        prompt: 'Write a TypeScript tokenizer/lexer for a simple arithmetic expression language. Tokens: numbers (int/float), operators (+,-,*,/,^), parentheses, whitespace (skip). Return a Token[] with type and value.',
        description: 'Arithmetic expression tokenizer',
        complexity: 'medium',
        category: 'string-text',
    },

    // ═══════════════════════════════════════════════════════════════════
    // MATH / NUMBER THEORY (10)
    // ═══════════════════════════════════════════════════════════════════

    {
        name: 'math-primes-sieve',
        prompt: 'Implement the Sieve of Eratosthenes in Python to find all primes up to n. Also implement a function to check if a single number is prime using the sieve. Optimize memory for large n (use bitarray-like approach).',
        description: 'Sieve of Eratosthenes',
        complexity: 'simple',
        category: 'math',
    },
    {
        name: 'math-gcd-lcm',
        prompt: 'Write TypeScript functions for GCD (Euclidean algorithm), LCM, and extended GCD (returning Bezout coefficients). Handle negative numbers and zero. Include overloads for arrays of numbers.',
        description: 'GCD, LCM, extended Euclidean',
        complexity: 'simple',
        category: 'math',
    },
    {
        name: 'math-matrix-ops',
        prompt: 'Implement a Matrix class in Python supporting: addition, subtraction, multiplication, transpose, determinant (for any NxN), and inverse (using cofactors). Include __repr__ and handle dimension mismatches with clear errors.',
        description: 'Matrix class with determinant/inverse',
        complexity: 'complex',
        category: 'math',
    },
    {
        name: 'math-big-int',
        prompt: 'Implement arbitrary precision integer arithmetic in TypeScript. Support: addition, subtraction, multiplication, division, modulo, and comparison. Store digits as arrays. Handle negative numbers and leading zeros.',
        description: 'BigInt from scratch',
        complexity: 'complex',
        category: 'math',
    },
    {
        name: 'math-combinatorics',
        prompt: 'Write Python functions for: factorial, permutations(n,r), combinations(n,r), and a generator that yields all permutations of a list. Use memoization. Handle large numbers without overflow.',
        description: 'Combinatorics utilities',
        complexity: 'medium',
        category: 'math',
    },
    {
        name: 'math-fraction',
        prompt: 'Implement a Fraction class in TypeScript with: add, subtract, multiply, divide, simplify, toDecimal, and toString (e.g., "3/4"). Auto-simplify after each operation. Handle division by zero.',
        description: 'Fraction arithmetic class',
        complexity: 'medium',
        category: 'math',
    },
    {
        name: 'math-statistics',
        prompt: 'Write a Python statistics module: mean, median, mode, variance, standard_deviation, percentile(data, p), z_score, and correlation(x, y). Handle edge cases: empty data, single value, all same values.',
        description: 'Statistics functions',
        complexity: 'medium',
        category: 'math',
    },
    {
        name: 'math-polynomial',
        prompt: 'Implement a Polynomial class in TypeScript. Support: evaluate(x), add, multiply, derivative, integral, and roots (for degree <= 2). Represent as coefficient array. Include toString for display.',
        description: 'Polynomial class',
        complexity: 'medium',
        category: 'math',
    },
    {
        name: 'math-roman-numerals',
        prompt: 'Write Python functions to convert between integers and Roman numerals. toRoman(int) and fromRoman(str). Handle values 1 to 3999. Validate Roman numeral strings and raise errors for invalid input.',
        description: 'Roman numeral converter',
        complexity: 'simple',
        category: 'math',
    },
    {
        name: 'math-expression-eval',
        prompt: 'Write a TypeScript function that evaluates mathematical expressions given as strings. Support: +, -, *, /, ^ (power), parentheses, and unary minus. Use the shunting-yard algorithm. Handle division by zero.',
        description: 'Expression evaluator (shunting-yard)',
        complexity: 'complex',
        category: 'math',
    },

    // ═══════════════════════════════════════════════════════════════════
    // WEB / API (12)
    // ═══════════════════════════════════════════════════════════════════

    {
        name: 'web-rest-crud',
        prompt: 'Create a Node.js Express CRUD API for a "tasks" resource. Endpoints: GET /tasks, GET /tasks/:id, POST /tasks, PUT /tasks/:id, DELETE /tasks/:id. Use in-memory storage. Include input validation and proper HTTP status codes.',
        description: 'REST CRUD API',
        complexity: 'medium',
        category: 'web-api',
    },
    {
        name: 'web-auth-jwt',
        prompt: 'Implement JWT authentication middleware for Express in TypeScript. Include: generateToken(user), verifyToken middleware, refreshToken endpoint, and proper error responses (401/403). Use HS256. No hardcoded secrets.',
        description: 'JWT auth middleware',
        complexity: 'medium',
        category: 'web-api',
    },
    {
        name: 'web-rate-limiter',
        prompt: 'Write an Express middleware in TypeScript for rate limiting. Support: configurable window (e.g., 15 min), max requests per window, per-IP tracking. Return 429 with Retry-After header. Use a token bucket algorithm.',
        description: 'Rate limiter middleware',
        complexity: 'medium',
        category: 'web-api',
    },
    {
        name: 'web-validation',
        prompt: 'Create a Python validation library with a fluent API. Support: string().min(n).max(n).email().regex(pattern), number().min(n).max(n).integer(), array().of(schema).min(n), object().shape({...}). Return detailed errors.',
        description: 'Schema validation library',
        complexity: 'complex',
        category: 'web-api',
    },
    {
        name: 'web-pagination',
        prompt: 'Write a TypeScript utility for cursor-based pagination of database-like results. Support: first/after (forward) and last/before (backward) pagination. Return pageInfo with hasNextPage, hasPreviousPage, startCursor, endCursor.',
        description: 'Cursor-based pagination',
        complexity: 'medium',
        category: 'web-api',
    },
    {
        name: 'web-graphql-schema',
        prompt: 'Design a GraphQL schema and resolvers in TypeScript for a blog platform. Types: User, Post, Comment. Support: queries (posts, post by id, user posts), mutations (createPost, addComment), and proper relations between types.',
        description: 'GraphQL blog schema',
        complexity: 'complex',
        category: 'web-api',
    },
    {
        name: 'web-websocket-chat',
        prompt: 'Implement a WebSocket chat server in Node.js using the ws library. Support: rooms, join/leave, broadcast to room, private messages, user nicknames, and a /users command to list connected users.',
        description: 'WebSocket chat server',
        complexity: 'complex',
        category: 'web-api',
    },
    {
        name: 'web-middleware-chain',
        prompt: 'Implement a middleware pattern in TypeScript similar to Express/Koa. Create a Pipeline class that supports use(middleware), where middleware is (ctx, next) => Promise<void>. Support error-handling middleware and context passing.',
        description: 'Middleware pipeline',
        complexity: 'medium',
        category: 'web-api',
    },
    {
        name: 'web-http-client',
        prompt: 'Write a Python HTTP client wrapper around urllib/httplib with: get, post, put, delete methods, automatic JSON parsing, timeout configuration, retry with exponential backoff (max 3 retries), and custom headers.',
        description: 'HTTP client with retry',
        complexity: 'medium',
        category: 'web-api',
    },
    {
        name: 'web-cache-layer',
        prompt: 'Create a TypeScript caching layer with: get/set with TTL, LRU eviction, cache-aside pattern (fetchAndCache), stale-while-revalidate support, and cache statistics (hits, misses, hit rate).',
        description: 'Cache layer with TTL and LRU',
        complexity: 'complex',
        category: 'web-api',
    },
    {
        name: 'web-router',
        prompt: 'Implement a simple URL router in Python. Support: path parameters (/users/:id), query string parsing, HTTP method matching, middleware per route, and a 404 handler. Similar to Flask/Express routing.',
        description: 'URL router with path params',
        complexity: 'medium',
        category: 'web-api',
    },
    {
        name: 'web-oauth-flow',
        prompt: 'Write TypeScript code implementing the OAuth 2.0 authorization code flow. Include: generating the auth URL with state/PKCE, exchanging the code for tokens, refreshing tokens, and storing tokens securely in memory.',
        description: 'OAuth 2.0 auth code flow',
        complexity: 'complex',
        category: 'web-api',
    },

    // ═══════════════════════════════════════════════════════════════════
    // DATABASE / SQL (8)
    // ═══════════════════════════════════════════════════════════════════

    {
        name: 'db-migration-roles',
        prompt: 'Write a SQL migration that adds a "roles" table (id, name, permissions JSON, created_at), a "user_roles" junction table with foreign keys and ON DELETE CASCADE, and indexes on user_roles. Include both UP and DOWN migrations.',
        description: 'Roles migration up/down',
        complexity: 'medium',
        category: 'database',
    },
    {
        name: 'db-query-builder',
        prompt: 'Implement a SQL query builder in TypeScript with a fluent API. Support: select, from, where, join (inner/left/right), orderBy, limit, offset, groupBy, having. Generate parameterized queries to prevent SQL injection.',
        description: 'SQL query builder (fluent)',
        complexity: 'complex',
        category: 'database',
    },
    {
        name: 'db-complex-query',
        prompt: 'Write a PostgreSQL query that: finds the top 10 users by total order amount in the last 30 days, includes their order count, average order value, most purchased category, and rank. Use CTEs and window functions.',
        description: 'Complex analytics query',
        complexity: 'medium',
        category: 'database',
    },
    {
        name: 'db-schema-ecommerce',
        prompt: 'Design a complete PostgreSQL schema for an e-commerce platform. Tables: users, products, categories, orders, order_items, reviews, addresses, payments. Include proper indexes, constraints, and relationships.',
        description: 'E-commerce database schema',
        complexity: 'complex',
        category: 'database',
    },
    {
        name: 'db-connection-pool',
        prompt: 'Implement a database connection pool in Python. Support: configurable min/max connections, acquire/release with context manager, health checks, idle timeout, and queue for waiting requests when pool is exhausted.',
        description: 'Connection pool manager',
        complexity: 'complex',
        category: 'database',
    },
    {
        name: 'db-orm-model',
        prompt: 'Create a simple ORM-like model system in TypeScript. Support: define model with fields (string, number, boolean, date), CRUD operations (save, findById, findAll, delete), and basic where clauses. Use in-memory storage.',
        description: 'Mini ORM system',
        complexity: 'complex',
        category: 'database',
    },
    {
        name: 'db-data-migration',
        prompt: 'Write a Python script that migrates data between two database schemas. Old schema has a "users" table with name (full name). New schema splits into first_name and last_name. Handle edge cases: multiple spaces, suffixes, empty names.',
        description: 'Data migration script',
        complexity: 'medium',
        category: 'database',
    },
    {
        name: 'db-index-advisor',
        prompt: 'Write a Python function that analyzes a list of SQL queries and suggests indexes. Parse SELECT/WHERE/JOIN/ORDER BY clauses to identify columns that would benefit from indexing. Output CREATE INDEX statements.',
        description: 'SQL index advisor',
        complexity: 'complex',
        category: 'database',
    },

    // ═══════════════════════════════════════════════════════════════════
    // FRONTEND / UI (8)
    // ═══════════════════════════════════════════════════════════════════

    {
        name: 'fe-react-form',
        prompt: 'Create a React component for a multi-step registration form (step 1: email/password, step 2: profile info, step 3: confirmation). Use TypeScript, validate each step, show a progress bar, and allow going back.',
        description: 'Multi-step React form',
        complexity: 'complex',
        category: 'frontend',
    },
    {
        name: 'fe-react-table',
        prompt: 'Build a reusable React table component in TypeScript. Support: sortable columns, pagination, search/filter, row selection (single and multi), and customizable cell renderers. Accept data as a generic typed array.',
        description: 'Reusable data table component',
        complexity: 'complex',
        category: 'frontend',
    },
    {
        name: 'fe-react-hooks',
        prompt: 'Write 5 custom React hooks in TypeScript: useDebounce(value, delay), useLocalStorage(key, initialValue), useClickOutside(ref, handler), useFetch(url) with loading/error states, useMediaQuery(query).',
        description: '5 custom React hooks',
        complexity: 'medium',
        category: 'frontend',
    },
    {
        name: 'fe-state-machine',
        prompt: 'Implement a finite state machine in TypeScript for a UI flow. Support: defineStates, defineTransitions, onEnter/onExit hooks, guards (conditional transitions), and a React hook useMachine(config).',
        description: 'UI state machine with React hook',
        complexity: 'complex',
        category: 'frontend',
    },
    {
        name: 'fe-virtual-list',
        prompt: 'Implement a virtual scrolling list component in React TypeScript. Only render visible items plus a buffer. Support: variable item heights, scroll-to-index, and smooth scrolling. Handle resize events.',
        description: 'Virtual scrolling list',
        complexity: 'complex',
        category: 'frontend',
    },
    {
        name: 'fe-drag-drop',
        prompt: 'Create a React drag-and-drop sortable list component in TypeScript. Support: reordering items via drag, drop indicators, keyboard accessibility (move with arrow keys), and an onChange callback with new order.',
        description: 'Drag-and-drop sortable list',
        complexity: 'complex',
        category: 'frontend',
    },
    {
        name: 'fe-form-validation',
        prompt: 'Write a React form validation library in TypeScript. Support: required, email, minLength, maxLength, pattern, custom validators. Show errors on blur and on submit. Support nested fields (address.city).',
        description: 'Form validation library',
        complexity: 'medium',
        category: 'frontend',
    },
    {
        name: 'fe-css-grid-layout',
        prompt: 'Create a responsive dashboard layout using CSS Grid and React. Include: header, sidebar (collapsible), main content area, and footer. Support breakpoints for mobile/tablet/desktop. Use CSS modules or styled-components.',
        description: 'Responsive CSS Grid dashboard',
        complexity: 'medium',
        category: 'frontend',
    },

    // ═══════════════════════════════════════════════════════════════════
    // CONCURRENCY / ASYNC (5)
    // ═══════════════════════════════════════════════════════════════════

    {
        name: 'async-promise-pool',
        prompt: 'Implement a promise pool in TypeScript that limits concurrent async tasks to N. Support: addTask(fn), waitAll(), onTaskComplete callback. Handle errors without stopping other tasks. Return results in order.',
        description: 'Promise pool with concurrency limit',
        complexity: 'medium',
        category: 'concurrency',
    },
    {
        name: 'async-retry',
        prompt: 'Write a Python async retry decorator with: configurable max retries, exponential backoff with jitter, retry on specific exception types only, on_retry callback, and a circuit breaker that opens after N consecutive failures.',
        description: 'Async retry with circuit breaker',
        complexity: 'complex',
        category: 'concurrency',
    },
    {
        name: 'async-event-emitter',
        prompt: 'Implement a typed event emitter in TypeScript. Support: on, off, once, emit, and waitFor (returns a promise that resolves on next emit). Use generics so event names and payload types are type-safe.',
        description: 'Typed event emitter',
        complexity: 'medium',
        category: 'concurrency',
    },
    {
        name: 'async-task-queue',
        prompt: 'Implement a persistent task queue in Python using asyncio. Support: enqueue(task), process with N workers, retry failed tasks, priority ordering, and graceful shutdown (finish current tasks on SIGTERM).',
        description: 'Async task queue with workers',
        complexity: 'complex',
        category: 'concurrency',
    },
    {
        name: 'async-debounce-throttle',
        prompt: 'Implement debounce and throttle functions in TypeScript with full type safety. Support: leading/trailing edge options, cancel method, flush method, and proper this-binding. Include both sync and async overloads.',
        description: 'Debounce and throttle utilities',
        complexity: 'medium',
        category: 'concurrency',
    },

    // ═══════════════════════════════════════════════════════════════════
    // SECURITY (5)
    // ═══════════════════════════════════════════════════════════════════

    {
        name: 'sec-input-sanitizer',
        prompt: 'Write a Python input sanitization library. Functions for: sanitize_html (allow only safe tags), prevent_sql_injection (parameterize), prevent_xss (escape), sanitize_filename, and validate_url (reject javascript: and data: URIs).',
        description: 'Input sanitization library',
        complexity: 'medium',
        category: 'security',
    },
    {
        name: 'sec-password-hashing',
        prompt: 'Implement a password hashing module in TypeScript using crypto (no external deps). Support: hash with salt (PBKDF2), verify, configurable iterations/key length, and a function to check if a hash needs rehashing (iterations too low).',
        description: 'Password hashing (PBKDF2)',
        complexity: 'medium',
        category: 'security',
    },
    {
        name: 'sec-csrf-protection',
        prompt: 'Write an Express middleware in TypeScript for CSRF protection using the double-submit cookie pattern. Generate tokens, validate on state-changing requests (POST/PUT/DELETE), and handle AJAX requests with custom headers.',
        description: 'CSRF protection middleware',
        complexity: 'medium',
        category: 'security',
    },
    {
        name: 'sec-encryption',
        prompt: 'Implement AES-256-GCM encryption/decryption in Python using the cryptography library. Support: encrypt(plaintext, key), decrypt(ciphertext, key), key derivation from password (PBKDF2), and secure random key generation.',
        description: 'AES-256-GCM encrypt/decrypt',
        complexity: 'complex',
        category: 'security',
    },
    {
        name: 'sec-rbac',
        prompt: 'Implement a Role-Based Access Control system in TypeScript. Support: roles, permissions, role hierarchy (admin inherits manager permissions), resource-based permissions (e.g., "post:edit:own"), and a checkAccess(user, permission, resource) function.',
        description: 'RBAC authorization system',
        complexity: 'complex',
        category: 'security',
    },
];

// ─── Generic agent caller ────────────────────────────────────────────

async function callAgent(
    agent: Agent,
    prompt: string,
    timeout: number,
    models: { gen1: string; gen2: string }
): Promise<{ content: string; error?: string }> {
    if (agent === 'gen1') {
        return callOllama(prompt, timeout, models.gen1);
    } else {
        return callOllama(prompt, timeout, models.gen2);
    }
}

// ─── Benchmark runners (6 configs) ──────────────────────────────────

/** Solo: single agent, one call */
async function benchSolo(
    agent: Agent,
    prompt: string,
    timeout: number,
    models: { gen1: string; gen2: string },
): Promise<ConfigResult> {
    const config: BenchConfig = agent === 'gen1' ? 'gen1-solo' : 'gen2-solo';
    const start = Date.now();
    const r = await callAgent(agent, prompt, timeout, models);
    return {
        config,
        duration: Date.now() - start,
        outputChars: r.content.length,
        outputLines: countLines(r.content),
        iterations: 1,
        consensusReached: true,
        output: r.content,
        error: r.error,
    };
}

/**
 * Lead/Consult: leader generates code, consultant reviews (OK/KO),
 * leader fixes if needed. Like the current debate but role-configurable.
 */
async function benchLeadConsult(
    leader: Agent,
    prompt: string,
    maxIter: number,
    timeout: number,
    models: { gen1: string; gen2: string },
): Promise<ConfigResult> {
    const config: BenchConfig = leader === 'gen1' ? 'gen1-lead' : 'gen2-lead';
    const consultant: Agent = leader === 'gen1' ? 'gen2' : 'gen1';
    const start = Date.now();
    let lastOutput = '';
    let iteration = 0;

    try {
        iteration = 1;
        const gen = await callAgent(leader, prompt, timeout, models);
        if (gen.error) {
            return makeResult(config, start, iteration, '', gen.error);
        }
        lastOutput = gen.content;

        for (iteration = 2; iteration <= maxIter; iteration++) {
            const reviewPrompt =
                `You are a code consultant. Review this code and respond with CONSENSUS_OK (if good) or CONSENSUS_KO with issues:\n\n${lastOutput.slice(0, 3000)}`;
            const review = await callAgent(consultant, reviewPrompt, timeout, models);
            if (review.error) break;

            if (review.content.includes('CONSENSUS_OK')) {
                return {
                    config, duration: Date.now() - start,
                    outputChars: lastOutput.length, outputLines: countLines(lastOutput),
                    iterations: iteration, consensusReached: true, output: lastOutput,
                };
            }

            const fixPrompt =
                `Fix the issues found by the reviewer:\n\n${review.content.slice(0, 2000)}\n\nOriginal output:\n${lastOutput.slice(0, 2000)}`;
            const fix = await callAgent(leader, fixPrompt, timeout, models);
            if (fix.error) break;
            lastOutput = fix.content;
        }

        return {
            config, duration: Date.now() - start,
            outputChars: lastOutput.length, outputLines: countLines(lastOutput),
            iterations: iteration - 1, consensusReached: false, output: lastOutput,
        };
    } catch (err) {
        return makeResult(config, start, iteration, lastOutput, String(err));
    }
}

/**
 * Orch/Code: orchestrator designs the plan, coder implements,
 * orchestrator reviews and requests corrections if needed.
 */
async function benchOrchCode(
    orchestrator: Agent,
    prompt: string,
    maxIter: number,
    timeout: number,
    models: { gen1: string; gen2: string },
): Promise<ConfigResult> {
    const config: BenchConfig = orchestrator === 'gen1' ? 'gen1-orch' : 'gen2-orch';
    const coder: Agent = orchestrator === 'gen1' ? 'gen2' : 'gen1';
    const start = Date.now();
    let lastOutput = '';
    let iteration = 0;

    try {
        // Orchestrator plans
        iteration = 1;
        const planPrompt =
            `You are a technical architect. For the following task, produce a detailed implementation plan with step-by-step instructions. Be specific about function signatures, algorithms, and edge cases to handle.\n\nTask: ${prompt}`;
        const plan = await callAgent(orchestrator, planPrompt, timeout, models);
        if (plan.error) {
            return makeResult(config, start, iteration, '', plan.error);
        }

        // Coder implements
        const implPrompt =
            `You are a developer. Implement the following plan precisely. Output only the code, well-structured and complete.\n\nPlan:\n${plan.content.slice(0, 3000)}\n\nOriginal task: ${prompt}`;
        const impl = await callAgent(coder, implPrompt, timeout, models);
        if (impl.error) {
            return makeResult(config, start, iteration, '', impl.error);
        }
        lastOutput = impl.content;

        // Orchestrator reviews + coder fixes
        for (iteration = 2; iteration <= maxIter; iteration++) {
            const reviewPrompt =
                `You are a technical lead reviewing an implementation. Check correctness and completeness. Respond with CONSENSUS_OK or CONSENSUS_KO with specific issues:\n\nTask: ${prompt}\n\nImplementation:\n${lastOutput.slice(0, 3000)}`;
            const review = await callAgent(orchestrator, reviewPrompt, timeout, models);
            if (review.error) break;

            if (review.content.includes('CONSENSUS_OK')) {
                return {
                    config, duration: Date.now() - start,
                    outputChars: lastOutput.length, outputLines: countLines(lastOutput),
                    iterations: iteration, consensusReached: true, output: lastOutput,
                };
            }

            const fixPrompt =
                `Fix the issues found by the reviewer:\n\n${review.content.slice(0, 2000)}\n\nYour previous code:\n${lastOutput.slice(0, 2000)}`;
            const fix = await callAgent(coder, fixPrompt, timeout, models);
            if (fix.error) break;
            lastOutput = fix.content;
        }

        return {
            config, duration: Date.now() - start,
            outputChars: lastOutput.length, outputLines: countLines(lastOutput),
            iterations: iteration - 1, consensusReached: false, output: lastOutput,
        };
    } catch (err) {
        return makeResult(config, start, iteration, lastOutput, String(err));
    }
}

function makeResult(
    config: BenchConfig, start: number, iter: number, output: string, error: string
): ConfigResult {
    return {
        config, duration: Date.now() - start,
        outputChars: output.length, outputLines: countLines(output),
        iterations: iter, consensusReached: false, output, error,
    };
}

/** Self-refine: single agent reviews and corrects its own output */
async function benchSelfRefine(
    agent: Agent,
    prompt: string,
    maxIter: number,
    timeout: number,
    models: { gen1: string; gen2: string },
): Promise<ConfigResult> {
    const config: BenchConfig = agent === 'gen1' ? 'gen1-selfrefine' : 'gen2-selfrefine';
    const start = Date.now();
    let lastOutput = '';

    try {
        // Iteration 1: generate
        const gen = await callAgent(agent, prompt, timeout, models);
        if (gen.error) return makeResult(config, start, 1, '', gen.error);
        lastOutput = gen.content;

        // Iterations 2..maxIter: self-review
        for (let i = 2; i <= maxIter; i++) {
            const reviewPrompt =
                `Review your code for correctness, edge cases, and quality.\nIf you find issues, provide the corrected complete version.\nIf it's correct, respond with CONSENSUS_OK.\n\nCode:\n${lastOutput.slice(0, 4000)}`;
            const review = await callAgent(agent, reviewPrompt, timeout, models);
            if (review.error) break;
            if (review.content.includes('CONSENSUS_OK')) {
                return {
                    config, duration: Date.now() - start,
                    outputChars: lastOutput.length, outputLines: countLines(lastOutput),
                    iterations: i, consensusReached: true, output: lastOutput,
                };
            }
            lastOutput = review.content;
        }

        return {
            config, duration: Date.now() - start,
            outputChars: lastOutput.length, outputLines: countLines(lastOutput),
            iterations: maxIter, consensusReached: false, output: lastOutput,
        };
    } catch (err) {
        return makeResult(config, start, maxIter, lastOutput, String(err));
    }
}

/** Dispatch to the right runner */
async function runBenchConfig(
    config: BenchConfig,
    prompt: string,
    maxIter: number,
    timeout: number,
    models: { gen1: string; gen2: string },
): Promise<ConfigResult> {
    switch (config) {
        case 'gen1-solo':       return benchSolo('gen1', prompt, timeout, models);
        case 'gen2-solo':       return benchSolo('gen2', prompt, timeout, models);
        case 'gen1-lead':       return benchLeadConsult('gen1', prompt, maxIter, timeout, models);
        case 'gen2-lead':       return benchLeadConsult('gen2', prompt, maxIter, timeout, models);
        case 'gen1-orch':       return benchOrchCode('gen1', prompt, maxIter, timeout, models);
        case 'gen2-orch':       return benchOrchCode('gen2', prompt, maxIter, timeout, models);
        case 'gen1-selfrefine': return benchSelfRefine('gen1', prompt, maxIter, timeout, models);
        case 'gen2-selfrefine': return benchSelfRefine('gen2', prompt, maxIter, timeout, models);
    }
}

// ─── Quality evaluation (blind, 3 judges: Claude + GPT + DeepSeek) ──

type JudgeName = 'Claude' | 'GPT' | 'DeepSeek';

interface JudgeCaller {
    name: JudgeName;
    call: (prompt: string, timeout: number) => Promise<{ content: string; error?: string }>;
}

function buildEvalPrompt(
    taskPrompt: string,
    shuffled: { config: BenchConfig; output: string }[],
    labels: string[],
): string {
    let outputsSection = '';
    for (let i = 0; i < shuffled.length; i++) {
        outputsSection += `\n## Output ${labels[i]}\n${shuffled[i].output.slice(0, 2500) || '(empty)'}\n`;
    }

    const jsonShape = labels.map(l =>
        `"${l}":{"correctness":N,"completeness":N,"edgeCases":N,"codeQuality":N,"readability":N,"justification":"..."}`
    ).join(',');

    return `You are a code quality evaluator. Score each code output independently for the same task.
Be strict and objective. Empty outputs or outputs without code should score 1 on all criteria.

## Task
${taskPrompt}
${outputsSection}
## Scoring criteria (1-10 each):
- correctness: Does the code work correctly?
- completeness: Are all requirements addressed?
- edgeCases: Are edge cases handled?
- codeQuality: Is it clean, idiomatic, well-structured?
- readability: Are naming, comments, clarity good?

Respond ONLY in this exact JSON format, no other text:
{${jsonShape}}`;
}

function parseJudgeResponse(
    content: string,
    labels: string[],
    labelMap: Map<string, BenchConfig>,
): Partial<Record<BenchConfig, QualityScores>> | null {
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;

    try {
        const parsed = JSON.parse(jsonMatch[0]);
        const result: Partial<Record<BenchConfig, QualityScores>> = {};

        for (const label of labels) {
            const obj = parsed[label];
            if (!obj) continue;
            const cfg = labelMap.get(label)!;
            const c = clamp(obj.correctness ?? 0);
            const co = clamp(obj.completeness ?? 0);
            const e = clamp(obj.edgeCases ?? 0);
            const q = clamp(obj.codeQuality ?? 0);
            const rd = clamp(obj.readability ?? 0);
            result[cfg] = {
                correctness: c, completeness: co, edgeCases: e,
                codeQuality: q, readability: rd,
                total: parseFloat(((c + co + e + q + rd) / 5).toFixed(1)),
                justification: String(obj.justification ?? ''),
            };
        }
        return Object.keys(result).length > 0 ? result : null;
    } catch {
        return null;
    }
}

function averageJudgeScores(
    allScores: Partial<Record<BenchConfig, QualityScores>>[],
    configs: BenchConfig[],
): Partial<Record<BenchConfig, QualityScores>> {
    const result: Partial<Record<BenchConfig, QualityScores>> = {};

    for (const cfg of configs) {
        const judges = allScores.filter(s => s[cfg]).map(s => s[cfg]!);
        if (judges.length === 0) continue;

        const avg = (field: keyof Omit<QualityScores, 'total' | 'justification'>) =>
            parseFloat((judges.reduce((a, j) => a + j[field], 0) / judges.length).toFixed(1));

        const c = avg('correctness');
        const co = avg('completeness');
        const e = avg('edgeCases');
        const q = avg('codeQuality');
        const rd = avg('readability');
        result[cfg] = {
            correctness: c, completeness: co, edgeCases: e,
            codeQuality: q, readability: rd,
            total: parseFloat(((c + co + e + q + rd) / 5).toFixed(1)),
            justification: judges.map((j, i) => `[J${i + 1}] ${j.justification}`).join(' | '),
        };
    }
    return result;
}

async function evaluateQuality(
    prompt: string,
    outputs: { config: BenchConfig; output: string }[],
    timeout: number,
    judgeModels: { claude: string; openai: string },
    tiebreakerModel?: string,
    judgeThreshold = 0.2,
): Promise<{ averaged: Partial<Record<BenchConfig, QualityScores>>; details: JudgeDetail[]; debateTriggered: boolean; tiebreakerUsed: boolean } | null> {
    const valid = outputs.filter(o => o.output.length > 0);
    if (valid.length === 0) return null;

    // Assign random labels to prevent position bias
    const labels = 'ABCDEFGH'.split('').slice(0, valid.length);
    const shuffled = shuffle(valid);
    const labelMap = new Map<string, BenchConfig>();
    const configToLabel = new Map<BenchConfig, string>();
    for (let i = 0; i < shuffled.length; i++) {
        labelMap.set(labels[i], shuffled[i].config);
        configToLabel.set(shuffled[i].config, labels[i]);
    }

    function labelToConfig(scores: Partial<Record<BenchConfig, QualityScores>>): Partial<Record<BenchConfig, QualityScores>> {
        return scores; // already config-based from parseJudgeResponse
    }

    const evalPrompt = buildEvalPrompt(prompt, shuffled, labels);

    // 2 primary judges (frontier, neutral — never generate)
    const judges: JudgeCaller[] = [
        { name: 'Claude', call: (p, t) => callClaude(p, t, judgeModels.claude) },
        { name: 'GPT', call: (p, t) => callOpenAI(p, t, judgeModels.openai) },
    ];

    // ─── Round 1: independent scoring ───
    const round1Results = await Promise.all(
        judges.map(async (judge) => {
            try {
                const r = await judge.call(evalPrompt, timeout);
                if (r.error) return { name: judge.name, scores: null };
                const scores = parseJudgeResponse(r.content, labels, labelMap);
                return { name: judge.name, scores };
            } catch {
                return { name: judge.name, scores: null };
            }
        })
    );

    const round1Valid = round1Results.filter(j => j.scores !== null);
    if (round1Valid.length === 0) return null;

    const r1Status = round1Results.map(j => `${j.name}:${j.scores ? 'OK' : 'FAIL'}`).join(' ');
    process.stdout.write(`[R1 ${r1Status}] `);

    const details: JudgeDetail[] = round1Valid.map(j => ({
        judge: j.name + ' (R1)',
        scores: j.scores!,
    }));

    const allConfigs = [...new Set(valid.map(v => v.config))];

    // Check if only 1 judge responded — no divergence possible
    if (round1Valid.length < 2) {
        const averaged = averageJudgeScores(round1Valid.map(j => j.scores!), allConfigs);
        return { averaged, details, debateTriggered: false, tiebreakerUsed: false };
    }

    // ─── Check divergence ───
    const divergentConfigs: BenchConfig[] = [];
    for (const cfg of allConfigs) {
        const totals = round1Valid.filter(j => j.scores![cfg]).map(j => j.scores![cfg]!.total);
        if (totals.length < 2) continue;
        const diff = Math.abs(totals[0] - totals[1]) / 10;
        if (diff > judgeThreshold) divergentConfigs.push(cfg);
    }

    if (divergentConfigs.length === 0) {
        const averaged = averageJudgeScores(round1Valid.map(j => j.scores!), allConfigs);
        return { averaged, details, debateTriggered: false, tiebreakerUsed: false };
    }

    // ─── Round 2: DEBATE on divergent configs ───
    const divergentLabels = divergentConfigs.map(c => configToLabel.get(c)!).filter(Boolean);
    const divergentNames = divergentConfigs.map(c => CONFIG_SHORT[c]);
    process.stdout.write(`[DEBATE: ${divergentNames.join(',')}] `);

    // Build debate prompt with anonymized justifications
    const debateOutputs: string[] = [];
    for (const cfg of divergentConfigs) {
        const r1scores = round1Valid.map((j, idx) => {
            const s = j.scores![cfg];
            return s ? `Judge ${String.fromCharCode(65 + idx)}: ${s.total}/10 — "${s.justification}"` : '';
        }).filter(Boolean);
        debateOutputs.push(`Config ${configToLabel.get(cfg)}:\n${r1scores.join('\n')}`);
    }

    const debatePrompt = `You are a code quality evaluator in a calibration round.
Two judges disagreed (>20% divergence) on the following configurations.

## Task
${prompt}

## Scores from Round 1 (anonymized):
${debateOutputs.join('\n\n')}

Re-evaluate ONLY these divergent configurations. Consider the other judge's reasoning.
Respond in the same JSON format as before, with scores 1-10 for each metric.`;

    const round2Results = await Promise.all(
        judges.map(async (judge) => {
            try {
                const r = await judge.call(debatePrompt, timeout);
                if (r.error) return { name: judge.name, scores: null };
                return { name: judge.name, scores: parseJudgeResponse(r.content, labels, labelMap) };
            } catch {
                return { name: judge.name, scores: null };
            }
        })
    );

    const round2Valid = round2Results.filter(j => j.scores !== null);
    const r2Status = round2Results.map(j => `${j.name}:${j.scores ? 'OK' : 'FAIL'}`).join(' ');
    process.stdout.write(`[R2 ${r2Status}] `);

    // Add round2 details
    for (const j of round2Valid) {
        details.push({ judge: j.name + ' (R2)', scores: j.scores! });
    }

    // Merge: round2 for divergent configs, round1 for the rest
    const mergedScores = round1Valid.map((r1) => {
        const r2 = round2Valid.find(r => r.name === r1.name);
        const merged: Partial<Record<BenchConfig, QualityScores>> = { ...r1.scores! };
        if (r2) {
            for (const cfg of divergentConfigs) {
                if (r2.scores![cfg]) merged[cfg] = r2.scores![cfg];
            }
        }
        return merged;
    });

    // Check if divergence persists after round 2
    const stillDivergent: BenchConfig[] = [];
    for (const cfg of divergentConfigs) {
        const totals = mergedScores.filter(s => s[cfg]).map(s => s[cfg]!.total);
        if (totals.length >= 2 && Math.abs(totals[0] - totals[1]) / 10 > judgeThreshold) {
            stillDivergent.push(cfg);
        }
    }

    // ─── Tie-breaker: Gemini (only if divergence persists) ───
    let tiebreakerUsed = false;
    if (stillDivergent.length > 0 && tiebreakerModel) {
        process.stdout.write(`[TIE-BREAK: ${stillDivergent.map(c => CONFIG_SHORT[c]).join(',')}] `);
        tiebreakerUsed = true;

        try {
            const tbResult = await callGemini(evalPrompt, timeout, tiebreakerModel);
            if (!tbResult.error) {
                const tbScores = parseJudgeResponse(tbResult.content, labels, labelMap);
                if (tbScores) {
                    details.push({ judge: 'Gemini (TB)', scores: tbScores });
                    mergedScores.push(tbScores);
                }
            }
        } catch { /* tie-breaker failure is non-fatal */ }
    }

    const averaged = averageJudgeScores(mergedScores, allConfigs);
    return { averaged, details, debateTriggered: true, tiebreakerUsed };
}

function clamp(n: number): number {
    return Math.max(1, Math.min(10, Math.round(n)));
}

// ─── Main ────────────────────────────────────────────────────────────

interface ParsedArgs {
    cases: string[];
    categories: Category[];
    complexities: Array<'simple' | 'medium' | 'complex'>;
    configs: BenchConfig[];
    maxIter: number;
    timeout: number;
    gen1Model: string;
    gen2Model: string;
    claudeJudgeModel: string;
    openaiJudgeModel: string;
    tiebreakerModel: string;
    judgeThreshold: number;
    seed: number;
    dryRun: boolean;
}

function parseArgs(): ParsedArgs {
    const args = process.argv.slice(2);
    let cases: string[] = [];
    let categories: Category[] = [];
    let complexities: Array<'simple' | 'medium' | 'complex'> = [];
    let configs: BenchConfig[] = [];
    let maxIter = 2;
    let timeout = 600_000;
    let gen1Model = 'qwen3-coder:480b-cloud';
    let gen2Model = 'minimax-m2:cloud';
    let claudeJudgeModel = 'claude-sonnet-4-5-20250929';
    let openaiJudgeModel = 'gpt-4.1';
    let tiebreakerModel = 'gemini-2.5-pro';
    let judgeThreshold = 0.2;
    let seed = 42;
    let dryRun = false;

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--cases' && args[i + 1]) {
            cases = args[++i].split(',');
        } else if (args[i] === '--category' && args[i + 1]) {
            categories = args[++i].split(',') as Category[];
        } else if (args[i] === '--complexity' && args[i + 1]) {
            complexities = args[++i].split(',') as Array<'simple' | 'medium' | 'complex'>;
        } else if (args[i] === '--configs' && args[i + 1]) {
            configs = args[++i].split(',') as BenchConfig[];
        } else if (args[i] === '--max-iter' && args[i + 1]) {
            maxIter = parseInt(args[++i], 10);
        } else if (args[i] === '--timeout' && args[i + 1]) {
            timeout = parseInt(args[++i], 10);
        } else if (args[i] === '--gen1-model' && args[i + 1]) {
            gen1Model = args[++i];
        } else if (args[i] === '--gen2-model' && args[i + 1]) {
            gen2Model = args[++i];
        } else if (args[i] === '--claude-judge-model' && args[i + 1]) {
            claudeJudgeModel = args[++i];
        } else if (args[i] === '--openai-judge-model' && args[i + 1]) {
            openaiJudgeModel = args[++i];
        } else if (args[i] === '--tiebreaker-model' && args[i + 1]) {
            tiebreakerModel = args[++i];
        } else if (args[i] === '--judge-threshold' && args[i + 1]) {
            judgeThreshold = parseFloat(args[++i]);
        } else if (args[i] === '--seed' && args[i + 1]) {
            seed = parseInt(args[++i], 10);
        } else if (args[i] === '--dry-run') {
            dryRun = true;
        } else if (args[i] === '--help') {
            console.log(`CRISTAL CODE Benchmark — 100 cases, 9 categories, 8 configurations\n`);
            console.log(`Configs: ${ALL_CONFIGS.join(', ')}\n`);
            console.log(`Options:`);
            console.log(`  --cases name1,name2         Run specific cases by name`);
            console.log(`  --category cat1,cat2        Filter by category`);
            console.log(`  --complexity simple,medium   Filter by difficulty`);
            console.log(`  --configs c1,c2             Select configs (default: all 8)`);
            console.log(`  --max-iter N                Max debate iterations (default: 2)`);
            console.log(`  --timeout N                 Timeout per call in ms (default: 600000)`);
            console.log(`  --gen1-model MODEL          Gen1 model (default: qwen3-coder:480b-cloud)`);
            console.log(`  --gen2-model MODEL          Gen2 model (default: minimax-m2:cloud)`);
            console.log(`  --claude-judge-model MODEL  Claude judge (default: claude-sonnet-4-5-20250929)`);
            console.log(`  --openai-judge-model MODEL  GPT judge (default: gpt-4.1)`);
            console.log(`  --tiebreaker-model MODEL    Tie-breaker (default: gemini-2.5-pro)`);
            console.log(`  --judge-threshold N         Divergence threshold (default: 0.2)`);
            console.log(`  --seed N                    PRNG seed (default: 42)`);
            console.log(`  --dry-run                   Simulate run with mock scores, no API calls`);
            console.log(`\nRequires: ANTHROPIC_API_KEY + OPENAI_API_KEY + GEMINI_API_KEY env vars + Ollama.`);
            console.log(`\nCategories: ${ALL_CATEGORIES.join(', ')}`);
            console.log(`\nAll ${ALL_CASES.length} cases:`);
            for (const cat of ALL_CATEGORIES) {
                const inCat = ALL_CASES.filter(c => c.category === cat);
                console.log(`  ${cat} (${inCat.length}): ${inCat.map(c => c.name).join(', ')}`);
            }
            process.exit(0);
        }
    }

    if (configs.length === 0) configs = [...ALL_CONFIGS];

    return {
        cases, categories, complexities, configs, maxIter, timeout,
        gen1Model, gen2Model, claudeJudgeModel, openaiJudgeModel,
        tiebreakerModel, judgeThreshold, seed, dryRun,
    };
}

const ALL_CATEGORIES: Category[] = [
    'algorithms', 'data-structures', 'string-text', 'math',
    'web-api', 'database', 'frontend', 'concurrency', 'security',
];

async function main(): Promise<void> {
    const opts = parseArgs();
    const {
        cases: filterCases, categories, complexities, configs,
        maxIter, timeout, gen1Model, gen2Model,
        claudeJudgeModel, openaiJudgeModel, tiebreakerModel,
        judgeThreshold, seed, dryRun,
    } = opts;
    const cwd = process.cwd();

    const genModels = { gen1: gen1Model, gen2: gen2Model };
    const judgeModels = { claude: claudeJudgeModel, openai: openaiJudgeModel };
    _rng = mulberry32(seed);

    console.log('╔══════════════════════════════════════════════════════════════════════╗');
    console.log('║     CRISTAL CODE — Benchmark Suite (8 configs × 100 cases)          ║');
    console.log('║     Solos · Lead/Consult · Orch/Code · Self-Refine                  ║');
    console.log('║     Mid-tier generators + Frontier judges                           ║');
    if (dryRun) {
    console.log('║     ⚡ DRY-RUN MODE — mock scores, no API calls                     ║');
    }
    console.log('╚══════════════════════════════════════════════════════════════════════╝\n');

    const hasAnthropic = !!process.env.ANTHROPIC_API_KEY;
    const hasOpenAI = !!process.env.OPENAI_API_KEY;
    const hasGemini = !!process.env.GEMINI_API_KEY;

    console.log(`  Generator 1 : ${gen1Model} (Ollama)`);
    console.log(`  Generator 2 : ${gen2Model} (Ollama)`);
    console.log(`  Judge 1     : ${claudeJudgeModel} (Anthropic API) ${hasAnthropic ? '✓' : 'MISSING'}`);
    console.log(`  Judge 2     : ${openaiJudgeModel} (OpenAI API) ${hasOpenAI ? '✓' : 'MISSING'}`);
    console.log(`  Tie-breaker : ${tiebreakerModel} (Google AI) ${hasGemini ? '✓' : 'MISSING'}`);
    console.log(`  Seed        : ${seed}`);
    console.log(`  Configs     : ${configs.map(c => CONFIG_SHORT[c]).join(', ')}`);
    console.log(`  Max iters   : ${maxIter}`);
    console.log(`  Threshold   : ${judgeThreshold}`);
    console.log(`  Timeout     : ${timeout}ms (${(timeout / 1000).toFixed(0)}s)`);
    if (dryRun) { console.log(`  Mode        : DRY-RUN (mock scores via PRNG, zero API calls)`); }
    console.log(`  CWD         : ${cwd}\n`);

    if (!dryRun) {
        if (!hasAnthropic) { console.error('ERROR: Set ANTHROPIC_API_KEY environment variable.'); process.exit(1); }
        if (!hasOpenAI) { console.error('ERROR: Set OPENAI_API_KEY environment variable.'); process.exit(1); }
    }

    // Apply filters
    let selectedCases = ALL_CASES;
    if (filterCases.length > 0) {
        selectedCases = selectedCases.filter(c => filterCases.includes(c.name));
    }
    if (categories.length > 0) {
        selectedCases = selectedCases.filter(c => categories.includes(c.category));
    }
    if (complexities.length > 0) {
        selectedCases = selectedCases.filter(c => complexities.includes(c.complexity));
    }

    if (selectedCases.length === 0) {
        console.error(`No matching cases. Use --help to see available cases.`);
        process.exit(1);
    }

    // Show category/complexity breakdown
    const catCounts = new Map<string, number>();
    const cplxCounts = new Map<string, number>();
    for (const c of selectedCases) {
        catCounts.set(c.category, (catCounts.get(c.category) ?? 0) + 1);
        cplxCounts.set(c.complexity, (cplxCounts.get(c.complexity) ?? 0) + 1);
    }
    console.log(`  Total cases : ${selectedCases.length} × ${configs.length} configs = ${selectedCases.length * configs.length} runs`);
    console.log(`  By category : ${[...catCounts.entries()].map(([k, v]) => `${k}(${v})`).join(', ')}`);
    console.log(`  By difficulty: ${[...cplxCounts.entries()].map(([k, v]) => `${k}(${v})`).join(', ')}\n`);
    console.log('═'.repeat(90));

    const results: BenchmarkResult[] = [];

    for (let ci = 0; ci < selectedCases.length; ci++) {
        const tc = selectedCases[ci];
        console.log(`\n▶ [${ci + 1}/${selectedCases.length}] ${tc.name} (${tc.category} / ${tc.complexity})`);
        console.log(`  ${tc.description}`);
        console.log(`  Prompt: "${tc.prompt.slice(0, 70)}..."\n`);

        const caseConfigs: ConfigResult[] = [];

        if (dryRun) {
            // ─── DRY-RUN: mock scores via PRNG, zero API calls ───
            for (const cfg of configs) {
                const mockDuration = Math.round(_rng() * 3000 + 500);
                const mockChars = Math.round(_rng() * 2000 + 200);
                caseConfigs.push({
                    config: cfg, output: `/* mock ${cfg} */`,
                    outputChars: mockChars, outputLines: Math.round(mockChars / 40),
                    duration: mockDuration, iterations: 1, consensusReached: false,
                });
                console.log(`  ${CONFIG_SHORT[cfg].padEnd(8)} ... ${mockDuration}ms, ${mockChars} chars (mock)`);
            }

            // Mock quality scores (4-9 range) + escalation simulation
            const quality: Partial<Record<BenchConfig, QualityScores>> = {};
            const judgeDetails: JudgeDetail[] = [];

            for (const cfg of configs) {
                const mockScore = (j: string) => {
                    const base = Math.round(_rng() * 5 + 4); // 4-9
                    return { correctness: base, completeness: base, edgeCases: base,
                        codeQuality: base, readability: base, total: base, justification: `mock (${j})` };
                };

                const s1 = mockScore('claude');
                const s2 = mockScore('openai');
                const delta = Math.abs(s1.total - s2.total) / 10;
                const diverged = delta > judgeThreshold;

                if (diverged) {
                    // Simulate round 2 (debate re-score)
                    const s1b = mockScore('claude-r2');
                    const s2b = mockScore('openai-r2');
                    const delta2 = Math.abs(s1b.total - s2b.total) / 10;

                    if (delta2 > judgeThreshold) {
                        // Tie-breaker would fire
                        const sTb = mockScore('gemini-tb');
                        quality[cfg] = sTb;
                        console.log(`  ${CONFIG_SHORT[cfg].padEnd(8)} Judge: ${s1.total}/${s2.total} → DEBATE → ${s1b.total}/${s2b.total} → TIE-BREAKER → ${sTb.total}/10`);
                    } else {
                        const avg = parseFloat(((s1b.total + s2b.total) / 2).toFixed(1));
                        quality[cfg] = { ...s1b, total: avg };
                        console.log(`  ${CONFIG_SHORT[cfg].padEnd(8)} Judge: ${s1.total}/${s2.total} → DEBATE → ${s1b.total}/${s2b.total} → avg=${avg}/10`);
                    }
                } else {
                    const avg = parseFloat(((s1.total + s2.total) / 2).toFixed(1));
                    quality[cfg] = { ...s1, total: avg };
                    console.log(`  ${CONFIG_SHORT[cfg].padEnd(8)} Judge: ${s1.total}/${s2.total} → avg=${avg}/10`);
                }
            }

            results.push({
                caseName: tc.name, category: tc.category, complexity: tc.complexity,
                timestamp: new Date().toISOString(), configs: caseConfigs, quality,
                judgeDetails,
            });
        } else {
            // ─── REAL RUN: call APIs ─────────────────────────────
            for (const cfg of configs) {
                process.stdout.write(`  ${CONFIG_SHORT[cfg].padEnd(8)} ... `);
                const result = await runBenchConfig(cfg, tc.prompt, maxIter, timeout, genModels);
                caseConfigs.push(result);
                console.log(formatResult(result));
            }

            // Quality evaluation — blind (2 judges + escalation)
            let quality: Partial<Record<BenchConfig, QualityScores>> | undefined;
            let judgeDetails: JudgeDetail[] | undefined;
            const withOutput = caseConfigs.filter(c => c.output.length > 0);

            if (withOutput.length > 0) {
                process.stdout.write('  Quality ... ');
                const evalResult = await evaluateQuality(
                    tc.prompt,
                    withOutput.map(c => ({ config: c.config, output: c.output })),
                    timeout,
                    judgeModels,
                    tiebreakerModel,
                    judgeThreshold,
                );

                if (evalResult) {
                    quality = evalResult.averaged;
                    judgeDetails = evalResult.details;
                    const scores = configs
                        .filter(c => quality![c])
                        .map(c => `${CONFIG_SHORT[c]}=${quality![c]!.total}/10`);
                    console.log(scores.join('  '));
                } else {
                    console.log('SKIPPED (eval failed)');
                }
            }

            results.push({
                caseName: tc.name, category: tc.category, complexity: tc.complexity,
                timestamp: new Date().toISOString(), configs: caseConfigs, quality,
                judgeDetails,
            });
        }

        console.log('─'.repeat(70));
    }

    // ─── Performance summary ─────────────────────────────────────
    console.log('\n╔══════════════════════════════════════════════════════════════════════════════════╗');
    console.log('║                              PERFORMANCE                                        ║');
    console.log('╚══════════════════════════════════════════════════════════════════════════════════╝\n');

    // Header
    let hdr = pad('Case', 22);
    for (const cfg of configs) { hdr += pad(CONFIG_SHORT[cfg], 12); }
    console.log(hdr);
    console.log('─'.repeat(22 + configs.length * 12));

    // Per-case rows
    for (const r of results) {
        let row = pad(r.caseName, 22);
        for (const cfg of configs) {
            const cr = r.configs.find(c => c.config === cfg);
            row += pad(cr ? (cr.error ? 'ERROR' : `${cr.duration}ms`) : '-', 12);
        }
        console.log(row);
    }

    // Averages
    const avgMs: Partial<Record<BenchConfig, number>> = {};
    for (const cfg of configs) {
        const durations = results
            .map(r => r.configs.find(c => c.config === cfg))
            .filter((c): c is ConfigResult => !!c && !c.error)
            .map(c => c.duration);
        if (durations.length > 0) {
            avgMs[cfg] = Math.round(durations.reduce((a, b) => a + b, 0) / durations.length);
        }
    }

    console.log('─'.repeat(22 + configs.length * 12));
    let avgRow = pad('AVERAGE', 22);
    for (const cfg of configs) {
        avgRow += pad(avgMs[cfg] ? `${avgMs[cfg]}ms` : '-', 12);
    }
    console.log(avgRow);

    // ─── Quality summary ─────────────────────────────────────────
    const withQuality = results.filter(r => r.quality);
    const avgQuality: Partial<Record<BenchConfig, number>> = {};
    const bestCount: Partial<Record<BenchConfig, number>> = {};

    if (withQuality.length > 0) {
        console.log('\n╔══════════════════════════════════════════════════════════════════════════════════╗');
        console.log('║                   QUALITY (2 judges + Gemini tie-breaker)                          ║');
        console.log('╚══════════════════════════════════════════════════════════════════════════════════╝\n');

        let qHdr = pad('Case', 22);
        for (const cfg of configs) { qHdr += pad(CONFIG_SHORT[cfg], 10); }
        qHdr += pad('Winner', 10);
        console.log(qHdr);
        console.log('─'.repeat(22 + configs.length * 10 + 10));

        for (const r of withQuality) {
            const q = r.quality!;
            let row = pad(r.caseName, 22);
            let bestScore = 0;
            let bestCfg = '';
            for (const cfg of configs) {
                const score = q[cfg]?.total ?? 0;
                row += pad(score ? `${score}/10` : '-', 10);
                if (score > bestScore) { bestScore = score; bestCfg = CONFIG_SHORT[cfg]; }
            }
            row += pad(bestCfg, 10);
            console.log(row);
        }

        // Averages
        for (const cfg of configs) {
            const totals = withQuality
                .filter(r => r.quality![cfg])
                .map(r => r.quality![cfg]!.total);
            if (totals.length > 0) {
                avgQuality[cfg] = parseFloat((totals.reduce((a, b) => a + b, 0) / totals.length).toFixed(1));
            }
            // Count wins
            bestCount[cfg] = 0;
        }

        for (const r of withQuality) {
            const q = r.quality!;
            let bestScore = 0;
            let bestCfg: BenchConfig | null = null;
            for (const cfg of configs) {
                const score = q[cfg]?.total ?? 0;
                if (score > bestScore) { bestScore = score; bestCfg = cfg; }
            }
            if (bestCfg) { bestCount[bestCfg] = (bestCount[bestCfg] ?? 0) + 1; }
        }

        console.log('─'.repeat(22 + configs.length * 10 + 10));
        let qAvgRow = pad('AVERAGE', 22);
        for (const cfg of configs) {
            qAvgRow += pad(avgQuality[cfg] ? `${avgQuality[cfg]}/10` : '-', 10);
        }
        console.log(qAvgRow);

        console.log('\n  Wins per config:');
        for (const cfg of configs) {
            if (bestCount[cfg]) {
                console.log(`    ${CONFIG_SHORT[cfg].padEnd(10)} ${bestCount[cfg]}/${withQuality.length} cases`);
            }
        }
    }

    // ─── Category breakdown ──────────────────────────────────────
    const byCategory: CategoryStats[] = [];
    const usedCategories = [...new Set(results.map(r => r.category))];

    if (usedCategories.length > 1) {
        console.log('\n╔══════════════════════════════════════════════════════════════════════════════════╗');
        console.log('║                         RESULTS BY CATEGORY                                     ║');
        console.log('╚══════════════════════════════════════════════════════════════════════════════════╝\n');

        let catHdr = pad('Category', 18) + pad('N', 4);
        for (const cfg of configs) { catHdr += pad(CONFIG_SHORT[cfg], 10); }
        catHdr += pad('Best', 10);
        console.log(catHdr);
        console.log('─'.repeat(18 + 4 + configs.length * 10 + 10));

        for (const cat of usedCategories) {
            const catResults = results.filter(r => r.category === cat);
            const catQ = catResults.filter(r => r.quality);
            const catAvgMs: Partial<Record<BenchConfig, number>> = {};
            const catAvgQ: Partial<Record<BenchConfig, number>> = {};

            for (const cfg of configs) {
                const durations = catResults
                    .map(r => r.configs.find(c => c.config === cfg))
                    .filter((c): c is ConfigResult => !!c && !c.error)
                    .map(c => c.duration);
                if (durations.length > 0) {
                    catAvgMs[cfg] = Math.round(durations.reduce((a, b) => a + b, 0) / durations.length);
                }
                const quals = catQ.filter(r => r.quality![cfg]).map(r => r.quality![cfg]!.total);
                if (quals.length > 0) {
                    catAvgQ[cfg] = parseFloat((quals.reduce((a, b) => a + b, 0) / quals.length).toFixed(1));
                }
            }

            // Best config by quality
            let bestQ = 0;
            let bestCfg = '-';
            for (const cfg of configs) {
                if ((catAvgQ[cfg] ?? 0) > bestQ) { bestQ = catAvgQ[cfg]!; bestCfg = CONFIG_SHORT[cfg]; }
            }

            let row = pad(cat, 18) + pad(String(catResults.length), 4);
            for (const cfg of configs) {
                const q = catAvgQ[cfg];
                const ms = catAvgMs[cfg];
                row += pad(q ? `${q}` : (ms ? `${ms}ms` : '-'), 10);
            }
            row += pad(bestCfg, 10);
            console.log(row);

            byCategory.push({
                category: cat,
                count: catResults.length,
                avgMs: catAvgMs,
                avgQuality: catAvgQ,
                bestConfig: bestCfg,
            });
        }
    }

    // ─── Save report ─────────────────────────────────────────────
    const report: BenchmarkReport = {
        meta: {
            date: new Date().toISOString(), cwd, maxIterations: maxIter,
            timeoutMs: timeout,
            gen1Model, gen2Model, claudeJudgeModel, openaiJudgeModel, tiebreakerModel, seed,
            totalAvailableCases: ALL_CASES.length, selectedCases: selectedCases.length,
            selectedConfigs: configs,
        },
        cases: results.map(r => ({
            ...r,
            configs: r.configs.map(c => ({ ...c, output: c.output.slice(0, 5000) })),
        })),
        summary: {
            totalCases: results.length,
            successfulCases: results.filter(r => r.configs.some(c => !c.error)).length,
            avgMs,
            avgQuality,
            bestConfigCount: bestCount,
            byCategory,
        },
    };

    const resultsDir = path.join(cwd, 'benchmark-results');
    fs.mkdirSync(resultsDir, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const reportPath = path.join(resultsDir, `bench_${ts}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n  Report saved to: ${reportPath}\n`);
}

// ─── Helpers ─────────────────────────────────────────────────────────

function formatResult(r: ConfigResult): string {
    if (r.error) { return `ERROR (${r.duration}ms) — ${r.error.slice(0, 60)}`; }
    const iter = r.iterations > 1 ? `, ${r.iterations}i` : '';
    const cons = r.consensusReached && r.iterations > 1 ? ', OK' : '';
    return `${r.duration}ms, ${r.outputChars} chars${iter}${cons}`;
}

function pad(s: string, len: number): string {
    return s.padEnd(len);
}

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

// ─── Run ─────────────────────────────────────────────────────────────

main().catch(err => {
    console.error('FATAL:', err);
    process.exit(1);
});
