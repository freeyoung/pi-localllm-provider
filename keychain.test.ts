import { afterEach, describe, expect, it, vi } from "vitest";

const execFileCalls: Array<{ file: string; args: string[] }> = [];
let nextShouldFail = false;

vi.mock("node:child_process", () => ({
  execFile: (
    file: string,
    args: string[],
    cb: (err: Error | null, stdout?: string, stderr?: string) => void,
  ) => {
    execFileCalls.push({ file, args });
    if (nextShouldFail) {
      cb(new Error("no such keychain item"));
    } else {
      cb(null, "", "");
    }
  },
}));

const { isDirectApiKey, keychainCommand, storeInKeychain, deleteFromKeychain } = await import(
  "./keychain.ts"
);

afterEach(() => {
  execFileCalls.length = 0;
  nextShouldFail = false;
});

describe("isDirectApiKey", () => {
  it("treats a plain string as a direct key", () => {
    expect(isDirectApiKey("sk-abc123")).toBe(true);
  });

  it("rejects empty, !command, and $ENV_VAR forms", () => {
    expect(isDirectApiKey("")).toBe(false);
    expect(isDirectApiKey("!security find-generic-password -w")).toBe(false);
    expect(isDirectApiKey("$MY_API_KEY")).toBe(false);
  });
});

describe("keychainCommand", () => {
  it("formats a !security find-generic-password command for the given account", () => {
    expect(keychainCommand("abc123")).toBe(
      "!security find-generic-password -s 'pi-localllm-provider' -a 'abc123' -w",
    );
  });
});

describe("storeInKeychain", () => {
  it("deletes any existing entry, then adds the key as a single argv element", async () => {
    await storeInKeychain("abc123", "sk-with-a-'-quote-and-$(dangerous)-chars");

    expect(execFileCalls).toHaveLength(2);
    expect(execFileCalls[0]).toEqual({
      file: "security",
      args: ["delete-generic-password", "-s", "pi-localllm-provider", "-a", "abc123"],
    });
    expect(execFileCalls[1]).toEqual({
      file: "security",
      args: [
        "add-generic-password",
        "-s",
        "pi-localllm-provider",
        "-a",
        "abc123",
        "-w",
        "sk-with-a-'-quote-and-$(dangerous)-chars",
      ],
    });
  });
});

describe("deleteFromKeychain", () => {
  it("swallows errors when there is no existing entry", async () => {
    nextShouldFail = true;
    await expect(deleteFromKeychain("abc123")).resolves.toBeUndefined();
  });
});
