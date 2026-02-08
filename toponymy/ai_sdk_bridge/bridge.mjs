import { readFileSync } from 'node:fs';
import { embedMany, generateText } from 'ai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createCohere } from '@ai-sdk/cohere';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createHuggingFace } from '@ai-sdk/huggingface';
import { createMistral } from '@ai-sdk/mistral';
import { createOpenAI } from '@ai-sdk/openai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { createTogetherAI } from '@ai-sdk/togetherai';
import { createAzure as createAzureInference } from '@quail-ai/azure-ai-provider';
import { createOllama } from 'ollama-ai-provider';

function pruneUndefined(obj) {
  return Object.fromEntries(
    Object.entries(obj).filter(([, value]) => value !== undefined && value !== null),
  );
}

function getProvider(request) {
  const providerKey = request.provider;

  switch (providerKey) {
    case 'openai':
      return createOpenAI(
        pruneUndefined({
          apiKey: request.apiKey,
          baseURL: request.baseURL,
          organization: request.organization,
          headers: request.headers,
        }),
      );

    case 'anthropic':
      return createAnthropic(
        pruneUndefined({
          apiKey: request.apiKey,
          baseURL: request.baseURL,
          headers: request.headers,
        }),
      );

    case 'cohere':
      return createCohere(
        pruneUndefined({
          apiKey: request.apiKey,
          baseURL: request.baseURL,
          headers: request.headers,
        }),
      );

    case 'google':
      return createGoogleGenerativeAI(
        pruneUndefined({
          apiKey: request.apiKey,
          baseURL: request.baseURL,
          headers: request.headers,
        }),
      );

    case 'azure':
    case 'azure_inference':
      return createAzureInference(
        pruneUndefined({
          apiKey: request.apiKey,
          endpoint: request.endpoint,
          apiVersion: request.apiVersion,
          headers: request.headers,
        }),
      );

    case 'together':
      return createTogetherAI(
        pruneUndefined({
          apiKey: request.apiKey,
          baseURL: request.baseURL,
          headers: request.headers,
        }),
      );

    case 'ollama':
      return createOllama(
        pruneUndefined({
          baseURL: request.host ?? request.baseURL,
          headers: request.headers,
        }),
      );

    case 'huggingface':
      return createHuggingFace(
        pruneUndefined({
          apiKey: request.apiKey,
          baseURL: request.baseURL,
          headers: request.headers,
        }),
      );

    case 'mistral':
      return createMistral(
        pruneUndefined({
          apiKey: request.apiKey,
          baseURL: request.baseURL,
          headers: request.headers,
        }),
      );

    case 'replicate':
      return createOpenAICompatible(
        pruneUndefined({
          name: 'replicate',
          apiKey: request.apiKey,
          baseURL: request.baseURL ?? 'https://api.replicate.com/v1',
          headers: request.headers,
        }),
      );

    case 'voyage':
      return createOpenAICompatible(
        pruneUndefined({
          name: 'voyage',
          apiKey: request.apiKey,
          baseURL: request.baseURL ?? 'https://api.voyageai.com/v1',
          headers: request.headers,
        }),
      );

    case 'openai_compatible':
      return createOpenAICompatible(
        pruneUndefined({
          name: request.name ?? 'compatible',
          apiKey: request.apiKey,
          baseURL: request.baseURL,
          headers: request.headers,
        }),
      );

    default:
      throw new Error(`Unsupported provider: ${providerKey}`);
  }
}

function getLanguageModel(provider, model) {
  if (typeof provider === 'function') {
    return provider(model);
  }

  const candidates = [
    'languageModel',
    'chat',
    'messages',
    'responses',
    'chatModel',
    'completionModel',
  ];

  for (const method of candidates) {
    if (typeof provider[method] === 'function') {
      return provider[method](model);
    }
  }

  throw new Error('Provider does not expose a language model factory');
}

function getEmbeddingModel(provider, model) {
  const candidates = [
    'embeddingModel',
    'embedding',
    'textEmbeddingModel',
    'textEmbedding',
  ];

  for (const method of candidates) {
    if (typeof provider[method] === 'function') {
      return provider[method](model);
    }
  }

  throw new Error('Provider does not expose an embedding model factory');
}

async function runGenerateText(request) {
  const provider = getProvider(request);
  const model = getLanguageModel(provider, request.model);

  const opts = {
    model,
    temperature: request.temperature,
    maxOutputTokens: request.maxOutputTokens ?? request.maxTokens,
    topP: request.topP,
    presencePenalty: request.presencePenalty,
    frequencyPenalty: request.frequencyPenalty,
    stopSequences: request.stopSequences,
    seed: request.seed,
    providerOptions: request.providerOptions,
  };

  if (request.prompt?.type === 'chat') {
    const messages = [];
    if (request.prompt.system) {
      messages.push({ role: 'system', content: request.prompt.system });
    }
    messages.push({ role: 'user', content: request.prompt.user });
    opts.messages = messages;
  } else if (request.prompt?.type === 'text') {
    opts.prompt = request.prompt.prompt;
  } else {
    throw new Error('Invalid prompt payload');
  }

  const result = await generateText(opts);
  return { text: result.text ?? '' };
}

async function runEmbedMany(request) {
  const texts = Array.isArray(request.texts) ? request.texts : [];
  if (texts.length === 0) {
    return { embeddings: [] };
  }

  const provider = getProvider(request);
  const model = getEmbeddingModel(provider, request.model);

  const result = await embedMany({
    model,
    values: texts,
    providerOptions: request.providerOptions,
  });

  return { embeddings: result.embeddings ?? [] };
}

async function main() {
  try {
    const input = readFileSync(0, 'utf8');
    const request = JSON.parse(input);

    let data;
    switch (request.action) {
      case 'generate_text':
        data = await runGenerateText(request);
        break;
      case 'embed_many':
        data = await runEmbedMany(request);
        break;
      default:
        throw new Error(`Unsupported action: ${request.action}`);
    }

    process.stdout.write(`${JSON.stringify({ ok: true, ...data })}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stdout.write(`${JSON.stringify({ ok: false, error: message })}\n`);
    process.exitCode = 1;
  }
}

await main();
