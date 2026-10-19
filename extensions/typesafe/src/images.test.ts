import {
  decisionBatchV1ToV2,
  decisionResultV2ToV1,
  type DecisionBatch,
  type DecisionBatchV2,
} from "openclaw/plugin-sdk/decisions";
import type { LookupFn } from "openclaw/plugin-sdk/ssrf-runtime";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import manifest from "../openclaw.plugin.json" with { type: "json" };
import { evaluate } from "./client.js";
import type { ModelProviderConfig } from "./config.js";
import { requestEvaluation } from "./transport.js";
import {
  choiceAnswer,
  decisionContext,
  mockJsonResponse,
  preparedContext,
  registeredDecisionProvider,
  registeredPreparedDecisionProvider,
  scoreAnswer,
} from "./transport.test-support.js";

vi.mock("openclaw/plugin-sdk/secret-input-runtime", async (importOriginal) => ({
  ...(await importOriginal<typeof import("openclaw/plugin-sdk/secret-input-runtime")>()),
  getPreparedPluginSecretInput: () => ({ revision: 1, value: "synthetic-key" }),
}));
const { lookup } = vi.hoisted(() => ({ lookup: vi.fn<LookupFn>() }));
vi.mock("node:dns/promises", async (importOriginal) => ({
  ...(await importOriginal<typeof import("node:dns/promises")>()),
  lookup,
}));

const HEADER = "TypeSafe-Danger-Zone-Images";
const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const questions = {
  q: { type: "boolean", instructions: "Is a person visible in the photo?" },
  c: { type: "choice", criteria: { indoor: "Indoors", outdoor: "Outdoors" } },
  s: { type: "score", criteria: ["Blurry", "Sharp"] },
} as const;
const response = {
  model: "jev-test",
  answers: {
    q: { type: "noul", noul: 0.82 },
    c: choiceAnswer("outdoor", { indoor: 0.1, outdoor: 0.9 }, 0.8),
    s: scoreAnswer(0.7, { 0: 0.3, 1: 0.7 }, { 0: "Blurry", 1: "Sharp" }, 0.6),
  },
  usage: { input_tokens: 812, output_tokens: 3 },
};
const png = Buffer.from("\x89PNG\r\n\x1a\nsynthetic pixels", "latin1").toString("base64");
const image = (dataUri: string, text?: string): DecisionBatchV2 => ({
  state: { type: "image", dataUri, ...(text === undefined ? {} : { text }) },
  questions,
});

function hosted(config: Record<string, unknown> = {}) {
  return registeredPreparedDecisionProvider(() => ({
    plugins: { entries: { typesafe: { config } } },
  }));
}
function routed(providerConfig: ModelProviderConfig) {
  return registeredPreparedDecisionProvider(() => ({
    models: { providers: { typesafe: providerConfig } },
  }));
}
function sent(fetch: ReturnType<typeof mockJsonResponse>) {
  const init = fetch.mock.lastCall?.[1];
  return {
    body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    headers: new Headers(init?.headers),
  };
}

beforeAll(async () => {
  // Measure translation and dispatch, not the cold client import.
  await import("./client.js");
});
beforeEach(() => {
  lookup.mockReset();
  lookup.mockResolvedValue([{ address: "104.18.0.1", family: 4 }]);
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("V2 registration", () => {
  it("registers a plugin-owned V2 provider over the unchanged manifest catalog", () => {
    expect(manifest.decisionModels.map((model) => model.id)).toEqual([
      "jev-latest",
      "jev-1.13.0",
      "kev-latest",
    ]);
    expect(manifest.modelCatalog.providers.typesafe.authScope).toBe("plugin");
    const provider = hosted();
    expect(provider).toMatchObject({ id: "typesafe", contractVersion: 2 });
    expect(provider.provider).toMatchObject({ label: "TypeSafe AI", auth: [] });
    expect(provider.provider?.resolveSyntheticAuth?.({ provider: "typesafe" })).toEqual({
      apiKey: "typesafe-plugin",
      source: "typesafe plugin",
      mode: "api-key",
    });
    expect(manifest.nonSecretAuthMarkers).toEqual(["typesafe-plugin"]);
  });

  it("reports readiness only while the plugin owns a usable route", () => {
    const missing = routed({
      baseUrl: "https://decision.example.com/v1",
      models: [],
      apiKey: { source: "store", provider: "default", id: "UNAVAILABLE" },
    });
    expect(missing.provider?.resolveSyntheticAuth?.({ provider: "typesafe" })).toBeUndefined();
    const invalid = hosted({ timeoutMs: -1 });
    expect(invalid.provider?.resolveSyntheticAuth?.({ provider: "typesafe" })).toBeUndefined();
    const local = hosted({ baseUrl: "http://127.0.0.1:8009" });
    expect(local.provider?.resolveSyntheticAuth?.({ provider: "typesafe" })).toBeDefined();
  });

  it("declares image input for hosted Jev rows only and keeps enrichment idempotent", async () => {
    const row = (id: string) => ({
      provider: "typesafe",
      id,
      name: id,
      inference: {
        chat: false,
        decision: {
          protocol: "decision-v1",
          input: ["text" as const],
          limits: { maxQuestions: 256 },
        },
      },
    });
    const augment = hosted().provider?.augmentModelCatalog;
    const enriched = await augment?.({
      env: {},
      entries: [row("jev-latest"), row("jev-1.13.0"), row("kev-latest"), row("other")].concat({
        ...row("unrelated"),
        provider: "openrouter",
      }),
    });
    const decision = (id: string) =>
      enriched?.find((entry) => entry.id === id)?.inference?.decision;
    for (const id of ["jev-latest", "jev-1.13.0"]) {
      expect(decision(id)).toEqual({
        protocol: "typesafe-systemone",
        input: ["text", "image"],
        reasoning: { modes: ["auto"] },
        limits: {
          maxQuestions: 256,
          image: {
            maxBytes: 4_194_304,
            mimeTypes: ["image/png", "image/jpeg"],
            remoteUrls: false,
          },
        },
      });
    }
    for (const id of ["kev-latest", "other"]) {
      expect(decision(id)).toEqual({
        protocol: "typesafe-systemone",
        input: ["text"],
        reasoning: { modes: ["auto"] },
        limits: { maxQuestions: 256 },
      });
    }
    expect(enriched?.map((entry) => entry.provider)).not.toContain("openrouter");
    // A route listed under models.providers.typesafe.models becomes a text decision row.
    const configured = await augment?.({
      env: {},
      entries: [{ provider: "typesafe", id: "my-route", name: "My route" }],
    });
    expect(configured).toEqual([
      {
        provider: "typesafe",
        id: "my-route",
        name: "My route",
        inference: {
          chat: false,
          decision: {
            protocol: "typesafe-systemone",
            input: ["text"],
            questions: {
              boolean: { probabilities: "boolean", abstention: false },
              choice: { probabilities: "provider-defined", abstention: false, maxOptions: 255 },
              score: { probabilities: "provider-defined", abstention: false, maxOptions: 10 },
            },
            confidence: "provider-specific",
            reasoning: { modes: ["auto"] },
            limits: { maxQuestions: 256 },
          },
        },
      },
    ]);
    expect(await augment?.({ env: {}, entries: enriched ? [...enriched] : [] })).toEqual(enriched);
  });
});

describe("text decisions", () => {
  const batch: DecisionBatch = { state: { evidence: "synthetic only" }, questions };
  const expectedBody =
    '{"state":{"evidence":"synthetic only"},"questions":{"q":{"type":"noul","instructions":"Is a person visible in the photo?"},"c":{"type":"choice","criteria":{"indoor":"Indoors","outdoor":"Outdoors"}},"s":{"type":"score","criteria":["Blurry","Sharp"]}},"model":"jev-latest"}';

  it("sends the same V1 wire request and returns V1-convertible results", async () => {
    const legacyFetch = mockJsonResponse(response);
    // A V1 caller reaches the V2 provider through the host's lossless conversion.
    const legacy = await registeredDecisionProvider(() => ({
      plugins: { entries: { typesafe: { config: {} } } },
    })).evaluate(batch, decisionContext("jev-latest"));
    const legacyInit = legacyFetch.mock.lastCall?.[1];
    const v2Fetch = mockJsonResponse(response);
    const v2Batch = decisionBatchV1ToV2(batch);
    if (!v2Batch) {
      throw new Error("expected a convertible text batch");
    }
    const outcome = await hosted().evaluate(v2Batch, preparedContext("jev-latest"));
    const v2Init = v2Fetch.mock.lastCall?.[1];
    expect(legacyInit?.body).toBe(expectedBody);
    expect(v2Init?.body).toBe(expectedBody);
    expect([...new Headers(v2Init?.headers)]).toEqual([...new Headers(legacyInit?.headers)]);
    expect(new Headers(v2Init?.headers).has(HEADER)).toBe(false);
    if (legacy.status !== "ok") {
      throw new Error("expected a V1 text decision");
    }
    expect(outcome).toEqual({ status: "ok", result: legacy.result });
    if (outcome.status !== "ok") {
      throw new Error("expected a text decision");
    }
    expect(decisionResultV2ToV1(v2Batch, outcome.result)).toEqual(outcome.result);
  });

  it("keeps V1 reasoning compatibility", async () => {
    const fetch = mockJsonResponse(response);
    const text = { state: { type: "text", text: "synthetic" }, questions } as const;
    const provider = hosted();
    await expect(
      provider.evaluate(text, { ...preparedContext("jev-latest"), reasoning: "auto" }),
    ).resolves.toMatchObject({ status: "ok" });
    await expect(
      provider.evaluate(text, { ...preparedContext("jev-latest"), reasoning: "on" }),
    ).resolves.toEqual({ status: "unavailable", reason: "unsupported-input" });
    expect(fetch).toHaveBeenCalledOnce();
  });
});

describe("tagged image state", () => {
  it.each([
    ["png", `data:image/png;base64,${png}`, "Doorbell camera, 08:14"],
    ["jpg", "data:image/jpeg;base64,/9j/4AAQSkZJRg==", undefined],
  ])("sends one referenced %s asset with the experimental header", async (format, uri, text) => {
    const fetch = mockJsonResponse(response);
    const outcome = await hosted().evaluate(image(uri, text), preparedContext("jev-latest"));
    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch.mock.lastCall?.[0]).toBe("https://api.typesafe.ai/v1/systemone");
    const { body, headers } = sent(fetch);
    expect(headers.get(HEADER)).toBe("true");
    expect(headers.get("authorization")).toBe("Bearer synthetic-key");
    const ref = Object.keys(body.assets)[0] ?? "";
    expect(ref).toMatch(UUID);
    expect(body).toEqual({
      state: text === undefined ? ref : [text, ref],
      questions: {
        q: { ...questions.q, type: "noul" },
        c: questions.c,
        s: questions.s,
      },
      model: "jev-latest",
      assets: { [ref]: { type: "image", format, data_base64: uri.split(",")[1] } },
    });
    expect(JSON.stringify(body).split(ref)).toHaveLength(3);
    expect(outcome).toEqual({
      status: "ok",
      result: {
        model: "jev-test",
        answers: {
          q: { type: "boolean", probabilityTrue: 0.82 },
          c: {
            type: "choice",
            choice: "outdoor",
            confidence: 0.8,
            probabilities: response.answers.c.probabilities,
          },
          s: { type: "score", score: 0.7, confidence: 0.6, probabilities: [0.3, 0.7] },
        },
        usage: { inputTokens: 812, outputTokens: 3 },
      },
    });
  });

  it("generates a fresh reference that does not occur in caller text", async () => {
    const fetch = mockJsonResponse(response);
    const provider = hosted();
    const refs = new Set<string>();
    for (let index = 0; index < 2; index++) {
      await provider.evaluate(image(`data:image/png;base64,${png}`), preparedContext("jev-latest"));
      refs.add(Object.keys(sent(fetch).body.assets)[0] ?? "");
    }
    expect(refs.size).toBe(2);
  });

  it.each([
    ["header", { headers: { [HEADER.toLowerCase()]: "false" } }],
    ["request header", { request: { headers: { [HEADER.toUpperCase()]: "false" } } }],
  ])("keeps the opt-in adapter-owned against a configured %s", async (_kind, override) => {
    const provider = routed({
      baseUrl: "https://api.typesafe.ai/v1",
      models: [],
      apiKey: "synthetic-key",
      ...override,
    });
    const fetch = mockJsonResponse(response);
    await expect(
      provider.evaluate(image(`data:image/png;base64,${png}`), preparedContext("jev-latest")),
    ).resolves.toMatchObject({ status: "ok" });
    expect(sent(fetch).headers.get(HEADER)).toBe("true");
    const text = routed({
      baseUrl: "https://api.typesafe.ai/v1",
      models: [],
      apiKey: "synthetic-key",
      ...JSON.parse(JSON.stringify(override).replaceAll('"false"', '"true"')),
    });
    await expect(
      text.evaluate(
        { state: { type: "text", text: "synthetic" }, questions },
        preparedContext("jev-latest"),
      ),
    ).resolves.toMatchObject({ status: "ok" });
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(sent(fetch).headers.has(HEADER)).toBe(false);
  });

  it.each([
    ["WebP", "data:image/webp;base64,UklGRg=="],
    ["GIF", "data:image/gif;base64,R0lGODlh"],
  ])("refuses %s before dispatch", async (_format, uri) => {
    const fetch = mockJsonResponse(response);
    await expect(hosted().evaluate(image(uri), preparedContext("jev-latest"))).resolves.toEqual({
      status: "unavailable",
      reason: "unsupported-input",
    });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("refuses image questions the native adapter cannot translate", async () => {
    const fetch = mockJsonResponse(response);
    const provider = hosted();
    for (const question of [
      { type: "box", instructions: "Where is the person?" },
      { type: "tags", criteria: { person: "A person" } },
    ] as const) {
      await expect(
        provider.evaluate(
          {
            state: { type: "image", dataUri: `data:image/png;base64,${png}` },
            questions: { x: question },
          },
          preparedContext("jev-latest"),
        ),
      ).resolves.toEqual({ status: "unavailable", reason: "unsupported-input" });
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([
    ["local Kev", () => hosted({ baseUrl: "http://127.0.0.1:8009" }), "kev-latest"],
    [
      "custom endpoint",
      () =>
        routed({ baseUrl: "https://decision.example.com/v1", models: [], apiKey: "server-key" }),
      "jev-latest",
    ],
  ])("refuses images for a %s before any HTTP dispatch", async (_kind, create, model) => {
    const fetch = mockJsonResponse(response);
    await expect(
      create().evaluate(image(`data:image/png;base64,${png}`), preparedContext(model)),
    ).resolves.toEqual({ status: "unavailable", reason: "unsupported-input" });
    expect(fetch).not.toHaveBeenCalled();
    expect(lookup).not.toHaveBeenCalled();
  });

  it("explains the local/custom refusal at the client and transport", async () => {
    const fetch = mockJsonResponse(response);
    const assets = { ref: { type: "image", format: "png", data_base64: png } } as const;
    const input = {
      state: "ref",
      questions: { q: { type: "noul" as const } },
      model: "kev-latest",
    };
    await expect(
      evaluate(
        input,
        { baseUrl: "http://127.0.0.1:8009", timeoutMs: 1000 },
        undefined,
        undefined,
        undefined,
        assets,
      ),
    ).rejects.toMatchObject({
      reason: "unsupported-input",
      message: expect.stringContaining("require hosted Jev"),
    });
    await expect(
      requestEvaluation({
        body: { ...input, model: "jev-latest", assets },
        providerConfig: { baseUrl: "https://decision.example.com/v1", models: [] },
        timeoutMs: 1000,
      }),
    ).rejects.toMatchObject({ reason: "unsupported-input" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("admits exactly 4 MiB of decoded image and refuses one byte more", async () => {
    const fetch = mockJsonResponse(response);
    const provider = hosted();
    const exact = Buffer.alloc(MAX_IMAGE_BYTES, 7).toString("base64");
    await expect(
      provider.evaluate(image(`data:image/png;base64,${exact}`), preparedContext("jev-latest")),
    ).resolves.toMatchObject({ status: "ok" });
    const { body, headers } = sent(fetch);
    expect(headers.get(HEADER)).toBe("true");
    expect(Object.values(body.assets)).toEqual([
      { type: "image", format: "png", data_base64: exact },
    ]);
    const over = Buffer.alloc(MAX_IMAGE_BYTES + 1, 7).toString("base64");
    await expect(
      provider.evaluate(image(`data:image/png;base64,${over}`), preparedContext("jev-latest")),
    ).resolves.toEqual({ status: "unavailable", reason: "unsupported-input" });
    expect(fetch).toHaveBeenCalledOnce();
  });
});
