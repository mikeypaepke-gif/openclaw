import { randomUUID } from "node:crypto";
import type { ModelCatalogEntry } from "openclaw/plugin-sdk/agent-runtime";
import {
  decisionBatchV2ToV1,
  decisionResultV1ToV2,
  validateDecisionBatchV2,
} from "openclaw/plugin-sdk/decisions";
import type {
  DecisionBatch,
  DecisionBatchResult,
  DecisionBatchV2,
  DecisionProviderV1,
  DecisionProviderV2,
  DecisionQuestion,
  ProviderDecisionOutcome,
  ProviderDecisionOutcomeV2,
} from "openclaw/plugin-sdk/decisions";
import { createLazyRuntimeModule } from "openclaw/plugin-sdk/lazy-runtime";
import { hasCredentials, type RuntimeConfig } from "./config.js";
import { decisionFailure } from "./errors.js";
import type { ImageAssets } from "./schema.js";

const loadClient = createLazyRuntimeModule(() => import("./client.js"));

/** The host's tagged-image bound; TypeSafe itself allows up to 16 MiB per image. */
const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
/** Hosted Jev models documented to accept images. Kev and custom routes stay text-only. */
const IMAGE_MODELS = new Set(["jev-latest", "jev-1.13.0"]);
/** Readiness evidence for the shared auth contract; it is never sent to an endpoint. */
const AUTH_MARKER = "typesafe-plugin";

type LegacyContext = Parameters<DecisionProviderV1["evaluate"]>[1];
type LegacyProvider = DecisionProviderV1 & {
  /** Only the V2 image translation supplies assets; V1 callers cannot. */
  evaluate(
    batch: DecisionBatch,
    context: LegacyContext,
    assets?: ImageAssets,
  ): Promise<ProviderDecisionOutcome>;
};

/** Translate the host-selected decision contract through the TypeSafe transport. */
function createDecisionProvider(getConfig: () => RuntimeConfig): LegacyProvider {
  return {
    id: "typesafe",
    contractVersion: 1,
    isReady: () => {
      const config = getConfig();
      return hasCredentials(config);
    },
    async evaluate(batch: DecisionBatch, context: LegacyContext, assets?: ImageAssets) {
      context.signal.throwIfAborted();
      // Registration and readiness do not need the transport or compiled validators.
      // Sample configuration and the remaining deadline after the cold import settles.
      const { evaluate: evaluateTypeSafe } = await loadClient();
      context.signal.throwIfAborted();
      const config = getConfig();
      if (!hasCredentials(config)) {
        return { status: "unavailable", reason: "credentials-unavailable" };
      }
      const remaining = context.deadlineMonotonicMs - performance.now();
      if (remaining <= 0) {
        return { status: "unavailable", reason: "transport" };
      }
      const questions = Object.fromEntries(
        Object.entries(batch.questions).map(([id, q]) => [
          id,
          q.type === "boolean" ? { ...q, type: "noul" } : q,
        ]),
      );
      try {
        const { evaluation } = await evaluateTypeSafe(
          { state: batch.state, questions, model: context.model },
          { ...config, timeoutMs: Math.min(config.timeoutMs, remaining) },
          context.signal,
          context.deadlineMonotonicMs,
          () => {
            if (context.isAdmissible && !context.isAdmissible()) {
              return false;
            }
            // DNS and capture admission may await; only the current owner may authorize egress.
            const current = getConfig();
            return (
              current.baseUrl === config.baseUrl &&
              current.providerConfig === config.providerConfig &&
              current.apiKey === config.apiKey &&
              !current.credentialsUnavailable
            );
          },
          assets,
        );
        context.signal.throwIfAborted();
        const answers: Record<string, DecisionBatchResult["answers"][string]> = {};
        for (const [id, answer] of Object.entries(evaluation.answers)) {
          if (answer.type === "noul") {
            answers[id] = { type: "boolean", probabilityTrue: answer.noul };
          } else if (answer.type === "choice") {
            answers[id] = answer;
          } else {
            const question = batch.questions[id];
            if (!question || question.type !== "score") {
              return { status: "unavailable", reason: "invalid-response" };
            }
            answers[id] = {
              type: "score",
              score: answer.score,
              confidence: answer.confidence,
              probabilities: question.criteria.map((_level, i) => answer.probabilities[String(i)]!),
            };
          }
        }
        return {
          status: "ok",
          result: {
            model: evaluation.model,
            answers,
            usage: {
              inputTokens: evaluation.usage.input_tokens,
              outputTokens: evaluation.usage.output_tokens,
            },
          },
        };
      } catch (error) {
        context.signal.throwIfAborted();
        return decisionFailure(error);
      }
    },
  };
}

/** Text-only facts for a configured server route, matching the declared presets. */
function configuredRouteInference(): NonNullable<ModelCatalogEntry["inference"]> {
  return {
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
  };
}

/** Map tagged image state to one generated TypeSafe reference plus its caption. */
function imageRequest(
  batch: DecisionBatchV2,
): { batch: DecisionBatch; assets: ImageAssets } | undefined {
  const state = batch.state;
  if (state.type !== "image") {
    return undefined;
  }
  // Jev accepts PNG and JPEG only; WebP and other formats are refused before dispatch.
  const match = /^data:image\/(png|jpeg);base64,/.exec(state.dataUri);
  const data = match ? state.dataUri.slice(match[0].length) : "";
  if (!match || Buffer.byteLength(data, "base64") > MAX_IMAGE_BYTES) {
    return undefined;
  }
  const questions: Record<string, DecisionQuestion> = {};
  for (const [id, question] of Object.entries(batch.questions)) {
    if (question.type !== "boolean" && question.type !== "choice" && question.type !== "score") {
      return undefined;
    }
    questions[id] = question;
  }
  // TypeSafe substitutes a reference wherever it occurs in request strings. Base64
  // never contains "-", so only caller text can collide with a fresh UUID.
  const text = JSON.stringify([state.text ?? null, questions]);
  let ref = randomUUID();
  while (text.includes(ref)) {
    ref = randomUUID();
  }
  return {
    batch: { state: state.text === undefined ? ref : [state.text, ref], questions },
    assets: {
      [ref]: { type: "image", format: match[1] === "png" ? "png" : "jpg", data_base64: data },
    },
  };
}

/** Text and JSON keep the V1 wire translation; one tagged image becomes a referenced asset. */
function translate(
  batch: DecisionBatchV2,
): { batch: DecisionBatch; assets?: ImageAssets } | undefined {
  try {
    if (!validateDecisionBatchV2(batch)) {
      return undefined;
    }
    if (batch.state.type === "image") {
      return imageRequest(batch);
    }
    const v1 = decisionBatchV2ToV1(batch);
    return v1 ? { batch: v1 } : undefined;
  } catch {
    // Malformed or accessor-backed input is refused without reading it further.
    return undefined;
  }
}

function resultV2(
  batch: DecisionBatch,
  outcome: ProviderDecisionOutcome,
): ProviderDecisionOutcomeV2 {
  if (outcome.status !== "ok") {
    return outcome;
  }
  const result = decisionResultV1ToV2(batch, outcome.result);
  return result ? { status: "ok", result } : { status: "unavailable", reason: "invalid-response" };
}

/** Shared-catalog provider: text keeps the V1 translation; tagged images reach hosted Jev. */
export function createPreparedDecisionProvider(getConfig: () => RuntimeConfig): DecisionProviderV2 {
  const legacy = createDecisionProvider(getConfig);
  return {
    id: "typesafe",
    contractVersion: 2,
    provider: {
      label: "TypeSafe AI",
      auth: [],
      authScope: "plugin",
      // Credentials stay plugin-owned and are read per request; this only reports readiness.
      resolveSyntheticAuth: () => {
        try {
          return hasCredentials(getConfig())
            ? { apiKey: AUTH_MARKER, source: "typesafe plugin", mode: "api-key" }
            : undefined;
        } catch {
          return undefined;
        }
      },
      augmentModelCatalog: ({ entries }) => {
        const models: typeof entries = [];
        for (const entry of entries) {
          if (entry.provider !== "typesafe") {
            continue;
          }
          const inference = entry.inference;
          if (!inference?.decision) {
            // Configured server routes carry no decision facts, and TypeSafe serves no chat.
            models.push({ ...entry, inference: configuredRouteInference() });
            continue;
          }
          // Enrich only the documented V1 manifest input; canonical declarations win.
          if (inference.decision.protocol !== "decision-v1") {
            models.push(entry);
            continue;
          }
          const decision = inference.decision;
          models.push({
            ...entry,
            inference: {
              ...inference,
              decision: {
                ...decision,
                protocol: "typesafe-systemone",
                reasoning: { modes: ["auto"] },
                ...(IMAGE_MODELS.has(entry.id)
                  ? {
                      input: ["text", "image"],
                      limits: {
                        ...decision.limits,
                        image: {
                          maxBytes: MAX_IMAGE_BYTES,
                          mimeTypes: ["image/png", "image/jpeg"],
                          remoteUrls: false,
                        },
                      },
                    }
                  : {}),
              },
            },
          });
        }
        return models;
      },
    },
    async evaluate(batch, context) {
      context.signal.throwIfAborted();
      if (context.reasoning !== undefined && context.reasoning !== "auto") {
        return { status: "unavailable", reason: "unsupported-input" };
      }
      const legacyContext: LegacyContext = {
        model: context.model.id,
        ...(context.agentId ? { agentId: context.agentId } : {}),
        signal: context.signal,
        deadlineMonotonicMs: context.deadlineMonotonicMs,
        isAdmissible: context.isAdmissible,
      };
      const request = translate(batch);
      if (!request) {
        return { status: "unavailable", reason: "unsupported-input" };
      }
      return resultV2(
        request.batch,
        await legacy.evaluate(request.batch, legacyContext, request.assets),
      );
    },
  };
}
