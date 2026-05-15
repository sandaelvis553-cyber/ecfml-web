/**
 * Shared helpers for normalizing API envelopes returned by Next.js proxy routes.
 */

export function unwrapApiData<T>(payload: unknown): T {
  if (
    payload &&
    typeof payload === "object" &&
    "data" in (payload as Record<string, unknown>)
  ) {
    return (payload as { data: T }).data;
  }

  return payload as T;
}

export function extractApiErrorMessage(
  payload: unknown,
  fallback: string,
): string {
  if (!payload || typeof payload !== "object") {
    return fallback;
  }

  const maybeError = (payload as { error?: unknown }).error;
  if (typeof maybeError === "string") {
    return maybeError;
  }

  if (maybeError && typeof maybeError === "object") {
    const errorObject = maybeError as { message?: unknown };
    if (typeof errorObject.message === "string" && errorObject.message) {
      return errorObject.message;
    }
  }

  return fallback;
}
