import type { AssistantMessage, Model } from "@earendil-works/pi-ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { generateBranchSummary } from "../../../src/core/compaction/branch-summarization.ts";
import type { SessionEntry } from "../../../src/core/session-manager.ts";

const { completeSimpleMock } = vi.hoisted(() => ({
	completeSimpleMock: vi.fn(),
}));

vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => {
	const actual = await importOriginal<typeof import("@earendil-works/pi-ai/compat")>();
	return {
		...actual,
		completeSimple: completeSimpleMock,
	};
});

function createModel(reasoning: boolean, maxTokens = 8192): Model<"anthropic-messages"> {
	return {
		id: reasoning ? "reasoning-model" : "non-reasoning-model",
		name: reasoning ? "Reasoning Model" : "Non-reasoning Model",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl: "https://api.anthropic.com",
		reasoning,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens,
	};
}

const mockSummaryResponse: AssistantMessage = {
	role: "assistant",
	content: [{ type: "text", text: "## Goal\nBranch summary" }],
	api: "anthropic-messages",
	provider: "anthropic",
	model: "claude-sonnet-4-5",
	usage: {
		input: 10,
		output: 10,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 20,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	},
	stopReason: "stop",
	timestamp: Date.now(),
};

const entries: SessionEntry[] = [
	{
		type: "message",
		id: "entry-1",
		parentId: null,
		timestamp: new Date().toISOString(),
		message: { role: "user", content: "abandoned branch work", timestamp: Date.now() },
	},
];

describe("generateBranchSummary thinking budget (regression)", () => {
	beforeEach(() => {
		completeSimpleMock.mockReset();
		completeSimpleMock.mockResolvedValue(mockSummaryResponse);
	});

	// Guards against reintroducing the bug this fix is for: threading the session's active
	// thinkingLevel into branch-summary's `reasoning` field competes with the visible summary
	// for the same shared maxTokens budget on the adaptive-thinking path and can bring back
	// empty summaries even with the larger proportional budget below. Branch summarization must
	// never set `reasoning`, regardless of what thinking level the caller/session has active.
	// `thinkingLevel` is cast via `as any` because `GenerateBranchSummaryOptions` deliberately has
	// no such field: this proves the signal is ignored even if a caller (or a future partial
	// revert) tries to smuggle it back in, not just that nobody happens to pass it today.
	it("does not request reasoning for reasoning-capable models even with thinking active in the session", async () => {
		await generateBranchSummary(entries, {
			model: createModel(true),
			apiKey: "test-key",
			signal: new AbortController().signal,
			thinkingLevel: "high",
		} as unknown as Parameters<typeof generateBranchSummary>[1]);

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).not.toHaveProperty("reasoning");
	});

	it("uses a maxTokens budget proportional to reserveTokens instead of the old 2048 hardcode", async () => {
		await generateBranchSummary(entries, {
			model: createModel(true, 16384),
			apiKey: "test-key",
			signal: new AbortController().signal,
			reserveTokens: 16384,
		});

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		// 0.8 * reserveTokens (16384) = 13107, matching compaction.ts's main summarization budget.
		// Prior to the fix this was hardcoded to 2048 regardless of reserveTokens/model.maxTokens.
		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({ maxTokens: 13107 });
		expect(completeSimpleMock.mock.calls[0][2]).not.toHaveProperty("reasoning");
	});

	it("clamps maxTokens to the model output cap when it is below the proportional budget", async () => {
		await generateBranchSummary(entries, {
			model: createModel(false, 4096),
			apiKey: "test-key",
			signal: new AbortController().signal,
			reserveTokens: 16384,
		});

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({ maxTokens: 4096 });
	});
});
