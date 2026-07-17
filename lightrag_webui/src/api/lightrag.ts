import axios, { AxiosError } from 'axios'
import { backendBaseUrl, popularLabelsDefaultLimit, searchLabelsDefaultLimit } from '@/lib/constants'
import { errorMessage } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore } from '@/stores/state'
import { provenanceFromHeaders, useProvenanceStore } from '@/stores/provenance'
import { navigationService } from '@/services/navigation'
import type { GraphPlane } from '@/stores/settings'

// Types
export type LightragNodeType = {
  id: string
  labels: string[]
  properties: Record<string, any>
}

export type LightragEdgeType = {
  id: string
  source: string
  target: string
  type: string
  properties: Record<string, any>
}

export type LightragGraphType = {
  nodes: LightragNodeType[]
  edges: LightragEdgeType[]
}

export type LightragStatus = {
  status: 'ready'
  generation_runtime: 'ready'
  core_version: string
  api_version: string
  webui_available: boolean
}

/**
 * Specifies the retrieval mode:
 * - "naive": Performs a basic search without advanced techniques.
 * - "local": Focuses on context-dependent information.
 * - "global": Utilizes global knowledge.
 * - "hybrid": Combines local and global retrieval methods.
 * - "mix": Integrates knowledge graph and vector retrieval.
 * - "bypass": Bypasses knowledge retrieval and directly uses the LLM.
 */
export type QueryMode = 'naive' | 'local' | 'global' | 'hybrid' | 'mix' | 'bypass'

export type Message = {
  role: 'user' | 'assistant' | 'system'
  content: string
  thinkingContent?: string
  displayContent?: string
  thinkingTime?: number | null
}

export type QueryRequest = {
  query: string
  /** Specifies the retrieval mode. */
  mode: QueryMode
  /** If True, only returns the retrieved context without generating a response. */
  only_need_context?: boolean
  /** If True, only returns the generated prompt without producing a response. */
  only_need_prompt?: boolean
  /** Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'. */
  response_type?: string
  /** If True, enables streaming output for real-time responses. */
  stream?: boolean
  /** Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode. */
  top_k?: number
  /** Maximum number of text chunks to retrieve and keep after reranking. */
  chunk_top_k?: number
  /** Maximum number of tokens allocated for entity context in unified token control system. */
  max_entity_tokens?: number
  /** Maximum number of tokens allocated for relationship context in unified token control system. */
  max_relation_tokens?: number
  /** Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt). */
  max_total_tokens?: number
  /**
   * Stores past conversation history to maintain context.
   * Format: [{"role": "user/assistant", "content": "message"}].
   */
  conversation_history?: Message[]
  /** Number of complete conversation turns (user-assistant pairs) to consider in the response context. */
  history_turns?: number
  /** User-provided prompt for the query. If provided, this will be used instead of the default value from prompt template. */
  user_prompt?: string
  /** Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued. Default is True. */
  enable_rerank?: boolean
}

export type Citation = {
  citation_id: string
  chunk_id: string
  source_key: string
  source_revision: string
  content?: string | null
}

export type QueryResponse = {
  response: string
  citations?: Citation[]
}

export type AuthStatusResponse = {
  auth_configured: boolean
  access_token?: string
  token_type?: string
  auth_mode?: 'enabled' | 'disabled'
  message?: string
  core_version?: string
  api_version?: string
  webui_title?: string
  webui_description?: string
}

export type LoginResponse = {
  access_token: string
  token_type: string
  auth_mode?: 'enabled' | 'disabled'  // Authentication mode identifier
  message?: string                    // Optional message
  core_version?: string
  api_version?: string
  webui_title?: string
  webui_description?: string
}

export const InvalidApiKeyError = 'Invalid API Key'
export const RequireApiKeError = 'API Key required'

// Axios instance
const axiosInstance = axios.create({
  baseURL: backendBaseUrl,
  headers: {
    'Content-Type': 'application/json'
  }
})

// ========== Token Management ==========
// Prevent multiple requests from triggering token refresh simultaneously
let isRefreshingGuestToken = false;
let refreshTokenPromise: Promise<string> | null = null;

// Silent refresh for guest token
const silentRefreshGuestToken = async (): Promise<string> => {
  // If already refreshing, return the same Promise
  if (isRefreshingGuestToken && refreshTokenPromise) {
    return refreshTokenPromise;
  }

  isRefreshingGuestToken = true;
  refreshTokenPromise = (async () => {
    try {
      // Call /auth-status to get new guest token
      const response = await axios.get('/auth-status', {
        baseURL: backendBaseUrl,
        // This request must skip the interceptor to avoid adding expired token
        headers: { 'X-Skip-Interceptor': 'true' }
      });

      if (response.data.access_token && !response.data.auth_configured) {
        const newToken = response.data.access_token;
        // Update localStorage
        localStorage.setItem('LIGHTRAG-API-TOKEN', newToken);
        // Update auth state
        useAuthStore.getState().login(
          newToken,
          true,
          response.data.core_version,
          response.data.api_version,
          response.data.webui_title || null,
          response.data.webui_description || null
        );
        return newToken;
      } else {
        throw new Error('Failed to get guest token');
      }
    } finally {
      isRefreshingGuestToken = false;
      refreshTokenPromise = null;
    }
  })();

  return refreshTokenPromise;
};

// Interceptor: add api key and check authentication
axiosInstance.interceptors.request.use((config) => {
  // Skip interceptor for token refresh requests
  if (config.headers['X-Skip-Interceptor']) {
    delete config.headers['X-Skip-Interceptor'];
    return config;
  }

  const apiKey = useSettingsStore.getState().apiKey
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN');

  // Always include token if it exists, regardless of path
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`
  }
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey
  }
  return config
})

// Interceptor：handle token renewal and authentication errors
axiosInstance.interceptors.response.use(
  (response) => {
    // ========== Check for new token from backend ==========
    const newToken = response.headers['x-new-token'];
    if (newToken) {
      localStorage.setItem('LIGHTRAG-API-TOKEN', newToken);

      // Optional: log in development mode
      if (import.meta.env.DEV) {
        console.log('[Auth] Token auto-renewed by backend');
      }

      // Update auth state with renewal tracking
      try {
        const payload = JSON.parse(atob(newToken.split('.')[1]));
        const authStore = useAuthStore.getState();
        if (authStore.isAuthenticated) {
          // Track token renewal time and expiration
          const renewalTime = Date.now();
          const expiresAt = payload.exp ? payload.exp * 1000 : 0;
          authStore.setTokenRenewal(renewalTime, expiresAt);

          // Update username (usually unchanged, but just in case)
          const newUsername = payload.sub;
          if (newUsername && newUsername !== authStore.username) {
            // Need to add setUsername method or just update via login
            // For now, we'll skip username update as it's rare
          }
        }
      } catch (error) {
        console.warn('[Auth] Failed to parse renewed token:', error);
      }
    }
    // ========== End of token renewal check ==========

    return response;
  },
  async (error: AxiosError) => {
    if (error.response) {
      if (error.response?.status === 401) {
        const originalRequest = error.config;

        // 1. For login API, throw error directly
        if (originalRequest?.url?.includes('/login')) {
          throw error;
        }

        // 2. Prevent infinite retry
        if (originalRequest && (originalRequest as any)._retry) {
          navigationService.navigateToLogin();
          return Promise.reject(new Error('Authentication required'));
        }

        // 3. Check if in guest mode
        const authStore = useAuthStore.getState();
        const currentToken = localStorage.getItem('LIGHTRAG-API-TOKEN');
        const isGuest = currentToken && authStore.isGuestMode;

        // 4. Guest mode: silent refresh and retry
        if (isGuest && originalRequest) {
          try {
            const newToken = await silentRefreshGuestToken();

            // Mark as retried to prevent infinite loop
            (originalRequest as any)._retry = true;

            // Update token in request headers
            originalRequest.headers['Authorization'] = `Bearer ${newToken}`;

            // Retry original request
            return axiosInstance(originalRequest);
          } catch (refreshError) {
            console.error('Failed to refresh guest token:', refreshError);
            // Refresh failed, navigate to login
            navigationService.navigateToLogin();
            return Promise.reject(new Error('Failed to refresh authentication'));
          }
        }

        // 5. Non-guest mode: navigate to login page
        navigationService.navigateToLogin();
        return Promise.reject(new Error('Authentication required'));
      }
      throw new Error(
        `${error.response.status} ${error.response.statusText}\n${JSON.stringify(
          error.response.data
        )}\n${error.config?.url}`
      )
    }
    throw error
  }
)

/**
 * Record the generation identity a plane response actually came from
 * (X-LightRAG-* headers) so the UI can surface build provenance.
 */
const captureAxiosProvenance = (
  plane: GraphPlane,
  headers: Record<string, unknown>
): void => {
  const provenance = provenanceFromHeaders(plane, (name) => {
    const value = headers[name]
    return typeof value === 'string' ? value : null
  })
  if (provenance) {
    useProvenanceStore.getState().setPlaneProvenance(provenance)
  }
}

// API methods
export const queryGraphs = async (
  plane: GraphPlane,
  label: string,
  maxDepth: number,
  maxNodes: number
): Promise<LightragGraphType> => {
  const response = await axiosInstance.get(`/planes/${plane}/graphs?label=${encodeURIComponent(label)}&max_depth=${maxDepth}&max_nodes=${maxNodes}`)
  captureAxiosProvenance(plane, response.headers)
  return response.data
}

export const getGraphLabels = async (plane: GraphPlane): Promise<string[]> => {
  const response = await axiosInstance.get(`/planes/${plane}/graph/label/list`)
  return response.data
}

export const getPopularLabels = async (
  plane: GraphPlane,
  limit: number = popularLabelsDefaultLimit
): Promise<string[]> => {
  const response = await axiosInstance.get(`/planes/${plane}/graph/label/popular?limit=${limit}`)
  return response.data
}

export const searchLabels = async (
  plane: GraphPlane,
  query: string,
  limit: number = searchLabelsDefaultLimit
): Promise<string[]> => {
  const response = await axiosInstance.get(`/planes/${plane}/graph/label/search?q=${encodeURIComponent(query)}&limit=${limit}`)
  return response.data
}

export const checkHealth = async (): Promise<
  LightragStatus | { status: 'error'; message: string }
> => {
  try {
    const response = await axiosInstance.get('/health')
    return response.data
  } catch (error) {
    return {
      status: 'error',
      message: errorMessage(error)
    }
  }
}

export const queryText = async (
  plane: GraphPlane,
  request: QueryRequest,
  signal?: AbortSignal
): Promise<QueryResponse> => {
  const response = await axiosInstance.post(`/planes/${plane}/query`, request, { signal })
  captureAxiosProvenance(plane, response.headers)
  return response.data
}

/**
 * True when an error originates from the user aborting the request (Stop
 * button) rather than a real failure. Used to suppress error rendering and any
 * auth-failure side effects (e.g. redirecting to login) on user cancellation.
 */
export const isUserAbortError = (
  signal: AbortSignal | undefined,
  error: unknown
): boolean => Boolean(signal?.aborted) || (error as Error)?.name === 'AbortError'

/**
 * Read an NDJSON (application/x-ndjson) stream from a fetch Response body
 * and dispatch each parsed line to ``onChunk`` / ``onError``.
 *
 * Extracted from ``queryTextStream`` so the normal path and the guest-token
 * retry path share the same parsing logic without duplication.
 */
async function _readNdjsonStream(
  response: Response,
  onChunk: (chunk: string) => void,
  onError: ((error: string) => void) | undefined,
  onCitations?: (citations: Citation[]) => void
): Promise<void> {
  if (!response.body) {
    throw new Error('Response body is null');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const dispatchLine = (parsed: Record<string, unknown>): void => {
    if (typeof parsed.response === 'string' && parsed.response) {
      onChunk(parsed.response);
    } else if (typeof parsed.error === 'string' && parsed.error) {
      onError?.(parsed.error);
    } else if (Array.isArray(parsed.citations)) {
      onCitations?.(parsed.citations as Citation[]);
    }
    // generation-frame lines are covered by the X-LightRAG-* headers.
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete lines (NDJSON)
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        try {
          dispatchLine(JSON.parse(trimmed));
        } catch {
          // Truncated or malformed JSON — log and skip the line so one
          // bad line does not kill the whole stream.
          console.warn('Failed to parse NDJSON line:', trimmed.substring(0, 120));
        }
      }
    }
  } finally {
    // Always release the reader lock, even on abort
    try {
      reader.releaseLock();
    } catch {
      // Already released or never acquired
    }
  }

  // Process any remaining data in the buffer after the stream ends
  if (buffer.trim()) {
    try {
      dispatchLine(JSON.parse(buffer));
    } catch {
      console.warn('Failed to parse final NDJSON buffer:', buffer.substring(0, 120));
      onError?.(
        'Response stream ended with incomplete data — the response may be truncated.'
      );
    }
  }
}

/**
 * Build auth headers for the streaming fetch request.
 */
function _buildStreamHeaders(): HeadersInit {
  const apiKey = useSettingsStore.getState().apiKey;
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN');
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    Accept: 'application/x-ndjson',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  return headers;
}

/**
 * Classify a fetch error and produce a user-friendly message for
 * ``onError``, or ``null`` when the error should be silently swallowed
 * (e.g. user-initiated abort).
 */
function _classifyStreamError(
  error: unknown,
  signal: AbortSignal | undefined
): string | null {
  if (isUserAbortError(signal, error)) {
    return null; // Stop button — exit silently
  }

  const message = errorMessage(error);

  if (message === 'Authentication required') {
    return 'Authentication required';
  }

  const statusCodeMatch = message.match(/^(\d{3})\s/);
  if (statusCodeMatch) {
    const statusCode = parseInt(statusCodeMatch[1], 10);
    switch (statusCode) {
      case 403:
        return 'You do not have permission to access this resource (403 Forbidden)';
      case 404:
        return 'The requested resource does not exist (404 Not Found)';
      case 429:
        return 'Too many requests, please try again later (429 Too Many Requests)';
      case 500:
      case 502:
      case 503:
      case 504:
        return `Server error, please try again later (${statusCode})`;
      default:
        return message;
    }
  }

  if (
    message.includes('NetworkError') ||
    message.includes('Failed to fetch') ||
    message.includes('Network request failed')
  ) {
    return 'Network connection error, please check your internet connection';
  }

  if (message.includes('Error parsing') || message.includes('SyntaxError')) {
    return 'Error processing response data';
  }

  return message;
}

/**
 * Format a non-ok streaming ``Response`` into the canonical error string that
 * ``_classifyStreamError`` understands (``"<status> <statusText>\n{...}\n<url>"``)
 * and throw it. Always throws — the return type is ``never``.
 *
 * Shared by the first response and the refreshed-retry response so that an
 * HTTP error (403/429/5xx, …) is classified identically on both paths.
 */
async function _throwStreamHttpError(response: Response, requestUrl: string): Promise<never> {
  let errorBody = 'Unknown error';
  try {
    errorBody = await response.text();
  } catch {
    /* ignore */
  }
  throw new Error(
    `${response.status} ${response.statusText}\n${JSON.stringify({ error: errorBody })}\n${requestUrl}`
  );
}

export const queryTextStream = async (
  plane: GraphPlane,
  request: QueryRequest,
  onChunk: (chunk: string) => void,
  onError?: (error: string) => void,
  signal?: AbortSignal,
  onCitations?: (citations: Citation[]) => void
) => {
  const headers = _buildStreamHeaders();
  const requestUrl = `${backendBaseUrl}/planes/${plane}/query/stream`

  try {
    const response = await fetch(requestUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
      signal,
    });

    // The response whose body we ultimately read — replaced by the retry
    // response when a guest token is silently refreshed below.
    let activeResponse = response;

    if (!response.ok) {
      // --- 401 guest-token retry -------------------------------------------
      if (response.status === 401) {
        const currentToken = localStorage.getItem('LIGHTRAG-API-TOKEN');
        const isGuest =
          currentToken && useAuthStore.getState().isGuestMode;

        if (isGuest) {
          // Only the token refresh + retry fetch are guarded here: a failure
          // of the refresh itself (or a user abort) is an auth problem and
          // routes to login. The retried HTTP *response*, however, is handled
          // with the same logic as the first response below, so a 403/429/5xx
          // on retry is classified by the outer catch rather than mislabelled
          // as an auth-refresh failure.
          let retryResponse: Response;
          try {
            const newToken = await silentRefreshGuestToken();
            const retryHeaders: Record<string, string> = { ...(headers as Record<string, string>) };
            retryHeaders['Authorization'] = `Bearer ${newToken}`;

            retryResponse = await fetch(requestUrl, {
              method: 'POST',
              headers: retryHeaders,
              body: JSON.stringify(request),
              signal,
            });
          } catch (refreshError) {
            if (isUserAbortError(signal, refreshError)) {
              return;
            }
            console.error(
              'Failed to refresh guest token for streaming:',
              refreshError
            );
            navigationService.navigateToLogin();
            throw new Error('Failed to refresh authentication', {
              cause: refreshError,
            });
          }

          if (!retryResponse.ok) {
            if (retryResponse.status === 401) {
              // Refreshed token still rejected → genuine auth failure
              navigationService.navigateToLogin();
              throw new Error('Authentication required');
            }
            // Non-auth HTTP error on retry → classify like the first response
            await _throwStreamHttpError(retryResponse, requestUrl);
          }

          activeResponse = retryResponse;
        } else {
          // Non-guest 401 → login
          navigationService.navigateToLogin();
          throw new Error('Authentication required');
        }
      } else {
        // --- Other HTTP errors ---------------------------------------------
        await _throwStreamHttpError(response, requestUrl);
      }
    }

    // --- Read the NDJSON stream (happy path or refreshed retry) ------------
    const provenance = provenanceFromHeaders(plane, (name) =>
      activeResponse.headers.get(name)
    );
    if (provenance) {
      useProvenanceStore.getState().setPlaneProvenance(provenance);
    }
    await _readNdjsonStream(activeResponse, onChunk, onError, onCitations);
  } catch (error) {
    const classified = _classifyStreamError(error, signal);
    if (classified === null) {
      return; // User abort — silent exit
    }
    console.error('Stream request error:', classified);
    onError?.(classified);
  }
};

export const getAuthStatus = async (): Promise<AuthStatusResponse> => {
  try {
    // Add a timeout to the request to prevent hanging
    const response = await axiosInstance.get('/auth-status', {
      timeout: 5000, // 5 second timeout
      headers: {
        'Accept': 'application/json' // Explicitly request JSON
      }
    });

    // Check if response is HTML (which indicates a redirect or wrong endpoint)
    const contentTypeHeader = response.headers['content-type'];
    const contentType = typeof contentTypeHeader === 'string' ? contentTypeHeader : '';
    if (contentType.includes('text/html')) {
      console.warn('Received HTML response instead of JSON for auth-status endpoint');
      return {
        auth_configured: true,
        auth_mode: 'enabled'
      };
    }

    // Strict validation of the response data
    if (response.data &&
        typeof response.data === 'object' &&
        'auth_configured' in response.data &&
        typeof response.data.auth_configured === 'boolean') {

      // For unconfigured auth, ensure we have an access token
      if (!response.data.auth_configured) {
        if (response.data.access_token && typeof response.data.access_token === 'string') {
          return response.data;
        } else {
          console.warn('Auth not configured but no valid access token provided');
        }
      } else {
        // For configured auth, just return the data
        return response.data;
      }
    }

    // If response data is invalid but we got a response, log it
    console.warn('Received invalid auth status response:', response.data);

    // Default to auth configured if response is invalid
    return {
      auth_configured: true,
      auth_mode: 'enabled'
    };
  } catch (error) {
    // If the request fails, assume authentication is configured
    console.error('Failed to get auth status:', errorMessage(error));
    return {
      auth_configured: true,
      auth_mode: 'enabled'
    };
  }
}

export const loginToServer = async (username: string, password: string): Promise<LoginResponse> => {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);
  formData.append('grant_type', 'password');

  const response = await axiosInstance.post('/login', formData, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
  });

  return response.data;
}
