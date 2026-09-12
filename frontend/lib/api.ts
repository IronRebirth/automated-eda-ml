const API_BASE_URL =
  typeof window === "undefined"
    ? process.env.INTERNAL_API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
    : process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function apiRequest<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    options,
  );

  if (!response.ok) {
    let message = `API request failed with status ${response.status}.`;

    try {
      const data = (await response.json()) as {
        detail?: string;
      };

      if (data.detail) {
        message = data.detail;
      }
    } catch {
      // Keep the default error message when the response is not JSON.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export { API_BASE_URL };