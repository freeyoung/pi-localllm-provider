// ─── macOS Keychain integration ────────────────────────────────────
//
// Stores raw API keys in the macOS keychain instead of settings.json,
// referencing them afterward via Pi's generic "!command" API key format
// (the same shell-resolved-credential mechanism Pi already supports for
// any provider's apiKey field).
//
// Uses execFile (not exec/a shell string) so the raw key is passed as a
// single argv element to the `security` binary — it can never be
// interpreted as shell syntax, regardless of what characters it contains.

import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const KEYCHAIN_SERVICE = "pi-localllm-provider";

export function isDirectApiKey(key: string): boolean {
  return key.length > 0 && !key.startsWith("!") && !key.startsWith("$");
}

export function keychainCommand(account: string): string {
  return `!security find-generic-password -s '${KEYCHAIN_SERVICE}' -a '${account}' -w`;
}

export async function storeInKeychain(account: string, rawKey: string): Promise<void> {
  await deleteFromKeychain(account);
  await execFileAsync("security", [
    "add-generic-password",
    "-s", KEYCHAIN_SERVICE,
    "-a", account,
    "-w", rawKey,
  ]);
}

export async function deleteFromKeychain(account: string): Promise<void> {
  try {
    await execFileAsync("security", [
      "delete-generic-password",
      "-s", KEYCHAIN_SERVICE,
      "-a", account,
    ]);
  } catch {
    // No existing entry for this account — fine.
  }
}
