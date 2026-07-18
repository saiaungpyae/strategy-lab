export async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url)
  return r.json() as Promise<T>
}

export async function postJSON<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    body: body == null ? undefined : JSON.stringify(body),
  })
  return r.json() as Promise<T>
}
