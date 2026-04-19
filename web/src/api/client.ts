const API_BASE = '/api/v1'

export class ApiError extends Error {
  readonly status: number
  readonly body: unknown
  readonly code?: string

  constructor(status: number, message: string, body?: unknown) {
    super(message)
    this.status = status
    this.body = body
    if (body && typeof body === 'object' && 'code' in body) {
      const code = (body as { code?: unknown }).code
      if (typeof code === 'string') this.code = code
    }
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { query?: Record<string, string | number | boolean | undefined> } = {},
): Promise<T> {
  const { query, ...rest } = init
  const url = new URL(API_BASE + path, window.location.origin)
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined) continue
      url.searchParams.set(key, String(value))
    }
  }

  const response = await fetch(url.toString(), {
    headers: { 'Content-Type': 'application/json', ...(rest.headers ?? {}) },
    ...rest,
  })
  const text = await response.text()
  const body = text ? (JSON.parse(text) as unknown) : undefined
  if (!response.ok) {
    const detail =
      (body && typeof body === 'object' && 'detail' in body && typeof (body as { detail?: unknown }).detail === 'string')
        ? ((body as { detail: string }).detail)
        : response.statusText
    throw new ApiError(response.status, detail, body)
  }
  return body as T
}

export const api = {
  get: <T>(path: string, query?: Record<string, string | number | boolean | undefined>) =>
    request<T>(path, { method: 'GET', query }),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'POST',
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'PUT',
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  del: <T>(path: string, query?: Record<string, string | number | boolean | undefined>) =>
    request<T>(path, { method: 'DELETE', query }),
}

export function wsUrl(path: string): string {
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.host
  return `${scheme}//${host}${API_BASE}${path}`
}
