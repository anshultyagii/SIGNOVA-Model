/**
 * Splits and trims a raw user input string into an ordered array of non-empty gloss tokens.
 * Supports delimiters: whitespace (spaces, tabs, newlines) and commas.
 */
export function parseGlossInput(raw: string): string[] {
  if (!raw) return [];
  return raw
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}
