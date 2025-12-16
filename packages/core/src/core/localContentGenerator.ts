/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import type { Config } from '../config/config.js';
import type { ContentGenerator } from './contentGenerator.js';
import { getErrorMessage } from '../utils/errors.js';
import { fetchWithTimeout } from '../utils/fetch.js';

// Basic type for the expected JSON response from streamGenerateContent
interface StreamGenerateContentResponse {
  candidates: Array<{
    content: {
      parts: Array<{ text: string }>;
      role: string;
    };
    finishReason: string;
    index: number;
  }>;
}

export class LocalContentGenerator implements ContentGenerator {
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly timeoutMs = 60000; // 60 seconds

  constructor(private readonly config: Config) {
    this.baseUrl = this.config.getLocalLlmBaseUrl();
    this.model = this.config.getLocalLlmModel();
  }

  private getEndpoint(stream: boolean): string {
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    return `${this.baseUrl}/models/${this.model}:${method}`;
  }

  async generateContent(
    request: GenerateContentParameters,
    // userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const endpoint = this.getEndpoint(false);
    const payload = {
      contents: request.contents,
      safetySettings: request.config?.safetySettings,
      generationConfig: {
        temperature: request.config?.temperature,
        topP: request.config?.topP,
        topK: request.config?.topK,
        candidateCount: request.config?.candidateCount,
        maxOutputTokens: request.config?.maxOutputTokens,
        stopSequences: request.config?.stopSequences,
      },
    };

    try {
      const response = await fetchWithTimeout(
        endpoint,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        },
        this.timeoutMs,
      );

      if (!response.ok) {
        throw new Error(
          `HTTP error! status: ${response.status} ${response.statusText}`,
        );
      }
      return (await response.json()) as GenerateContentResponse;
    } catch (error) {
      throw new Error(
        `Failed to generate content from local LLM: ${getErrorMessage(error)}`,
      );
    }
  }

  async *generateContentStream(
    request: GenerateContentParameters,
    // userPromptId: string,
  ): AsyncGenerator<GenerateContentResponse> {
    const endpoint = this.getEndpoint(true);
    const payload = {
      contents: request.contents,
      safetySettings: request.config?.safetySettings,
      generationConfig: {
        temperature: request.config?.temperature,
        topP: request.config?.topP,
        topK: request.config?.topK,
        candidateCount: request.config?.candidateCount,
        maxOutputTokens: request.config?.maxOutputTokens,
        stopSequences: request.config?.stopSequences,
      },
    };

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(
          `HTTP error! status: ${response.status} ${response.statusText}`,
        );
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });

        // gemma_api_server returns newline-delimited JSON chunks
        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
          const line = buffer.substring(0, newlineIndex);
          buffer = buffer.substring(newlineIndex + 1);

          if (line.startsWith('data: ')) {
            const jsonStr = line.substring(5);
            try {
              const json = JSON.parse(
                jsonStr,
              ) as StreamGenerateContentResponse[];
              // The server wraps the response in an array
              if (Array.isArray(json) && json.length > 0) {
                yield json[0] as GenerateContentResponse;
              } else {
                console.warn('Unexpected stream format:', json);
              }
            } catch (e) {
              console.error('Failed to parse JSON stream chunk:', e, jsonStr);
            }
          }
        }
      }
    } catch (error) {
      throw new Error(
        `Failed to generate content stream from local LLM: ${getErrorMessage(
          error,
        )}`,
      );
    }
  }

  async countTokens(
    _request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    throw new Error('countTokens is not supported by the local LLM server.');
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('embedContent is not supported by the local LLM server.');
  }
}
