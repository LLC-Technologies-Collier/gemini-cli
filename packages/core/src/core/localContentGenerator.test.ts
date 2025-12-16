/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Config } from '../config/config.js';
import { LocalContentGenerator } from './localContentGenerator.js';
import type * as fetchUtils from '../utils/fetch.js';
import type {
  GenerateContentParameters,
  GenerateContentResponse,
  CountTokensParameters,
  EmbedContentParameters,
} from '@google/genai';

// Mock fetchWithTimeout
const mockFetchWithTimeout = vi.fn();
vi.mock('../utils/fetch.js', async (importOriginal) => {
  const actual = await importOriginal<typeof fetchUtils>();
  return {
    ...actual,
    fetchWithTimeout: mockFetchWithTimeout,
  };
});

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('LocalContentGenerator', () => {
  let config: Config;
  let generator: LocalContentGenerator;

  beforeEach(() => {
    config = {
      getLocalLlmEnabled: vi.fn().mockReturnValue(true),
      getLocalLlmBaseUrl: vi
        .fn()
        .mockReturnValue('http://localhost:8000/v1beta'),
      getLocalLlmModel: vi.fn().mockReturnValue('gemini-1.5-flash'),
    } as unknown as Config;
    generator = new LocalContentGenerator(config);
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('generateContent', () => {
    it('should call fetchWithTimeout with the correct arguments', async () => {
      const request: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };
      const mockResponse: GenerateContentResponse = {
        candidates: [
          {
            index: 0,
            content: { role: 'model', parts: [{ text: 'Hi there!' }] },
          },
        ],
      };
      mockFetchWithTimeout.mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });

      await generator.generateContent(request, 'prompt-id');

      expect(mockFetchWithTimeout).toHaveBeenCalledWith(
        'http://localhost:8000/v1beta/models/gemini-1.5-flash:generateContent',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            contents: request.contents,
            safetySettings: undefined,
            generationConfig: {
              temperature: undefined,
              topP: undefined,
              topK: undefined,
              candidateCount: undefined,
              maxOutputTokens: undefined,
              stopSequences: undefined,
            },
          }),
        },
        60000,
      );
    });

    it('should return the response from fetchWithTimeout', async () => {
      const request: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };
      const mockResponse: GenerateContentResponse = {
        candidates: [
          {
            index: 0,
            content: { role: 'model', parts: [{ text: 'Hi there!' }] },
          },
        ],
      };
      mockFetchWithTimeout.mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await generator.generateContent(request, 'prompt-id');
      expect(response).toEqual(mockResponse);
    });

    it('should throw an error if fetchWithTimeout fails', async () => {
      const request: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };
      mockFetchWithTimeout.mockRejectedValue(new Error('Network error'));

      await expect(
        generator.generateContent(request, 'prompt-id'),
      ).rejects.toThrow(
        'Failed to generate content from local LLM: Network error',
      );
    });

    it('should throw an error if response is not ok', async () => {
      const request: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };
      mockFetchWithTimeout.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
      });

      await expect(
        generator.generateContent(request, 'prompt-id'),
      ).rejects.toThrow(
        'Failed to generate content from local LLM: HTTP error! status: 500 Internal Server Error',
      );
    });
  });

  describe('generateContentStream', () => {
    // Helper to create a ReadableStream from an array of strings
    const createMockStream = (chunks: string[]) => {
      const encoder = new TextEncoder();
      const readableStream = new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(encoder.encode(chunk));
          }
          controller.close();
        },
      });
      return { ok: true, body: readableStream };
    };

    it('should call fetch with the correct arguments', async () => {
      const request: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };
      mockFetch.mockResolvedValue(createMockStream([]));

      const stream = await generator.generateContentStream(
        request,
        'prompt-id',
      );
      await stream.next(); // Consume the stream

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/v1beta/models/gemini-1.5-flash:streamGenerateContent',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            contents: request.contents,
            safetySettings: undefined,
            generationConfig: {
              temperature: undefined,
              topP: undefined,
              topK: undefined,
              candidateCount: undefined,
              maxOutputTokens: undefined,
              stopSequences: undefined,
            },
          }),
        },
      );
    });

    it('should yield stream responses', async () => {
      const request: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Stream test' }] }],
      };
      const mockResponses: GenerateContentResponse[] = [
        {
          candidates: [
            {
              index: 0,
              content: { role: 'model', parts: [{ text: 'Chunk 1' }] },
            },
          ],
        },
        {
          candidates: [
            {
              index: 0,
              content: { role: 'model', parts: [{ text: ' Chunk 2' }] },
            },
          ],
        },
      ];
      const streamChunks = mockResponses.map(
        (r) => `data: ${JSON.stringify([r])}\n`,
      );
      mockFetch.mockResolvedValue(createMockStream(streamChunks));

      const stream = await generator.generateContentStream(
        request,
        'prompt-id',
      );
      const receivedResponses: GenerateContentResponse[] = [];
      for await (const chunk of stream) {
        receivedResponses.push(chunk);
      }

      expect(receivedResponses).toEqual(mockResponses);
    });

    it('should handle stream errors', async () => {
      const request: GenerateContentParameters = {
        contents: [{ role: 'user', parts: [{ text: 'Stream error' }] }],
      };
      mockFetch.mockRejectedValue(new Error('Stream failed'));

      try {
        const stream = await generator.generateContentStream(
          request,
          'prompt-id',
        );
        await stream.next();
        expect.fail('Stream should have thrown an error');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        expect((error as Error).message).toContain(
          'Failed to generate content stream from local LLM: Stream failed',
        );
      }
    });
  });

  describe('unsupported methods', () => {
    it('countTokens should throw an error', async () => {
      await expect(
        generator.countTokens({ contents: [] } as CountTokensParameters),
      ).rejects.toThrow(
        'countTokens is not supported by the local LLM server.',
      );
    });

    it('embedContent should throw an error', async () => {
      await expect(
        generator.embedContent({
          content: { parts: [] },
        } as EmbedContentParameters),
      ).rejects.toThrow(
        'embedContent is not supported by the local LLM server.',
      );
    });
  });
});
