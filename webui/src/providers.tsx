export interface ProviderInfo {
  id: string;
  label: string;
  color: string;
  keyName: string | null;
  placeholder: string;
  optionalKey?: boolean;
}

export const PROVIDERS: ProviderInfo[] = [
  {
    id: "anthropic",
    label: "Anthropic",
    color: "#d97757",
    keyName: "anthropic",
    placeholder: "claude-sonnet-5",
  },
  {
    id: "openai",
    label: "OpenAI",
    color: "#10a37f",
    keyName: "openai",
    placeholder: "gpt-5.2",
  },
  {
    id: "google",
    label: "Google",
    color: "#4285f4",
    keyName: "google",
    placeholder: "gemini-3-pro",
  },
  {
    id: "xai",
    label: "xAI",
    color: "#43474f",
    keyName: "xai",
    placeholder: "grok-4",
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    color: "#4d6bfe",
    keyName: "deepseek",
    placeholder: "deepseek-chat",
  },
  {
    id: "meta",
    label: "Meta",
    color: "#0866ff",
    keyName: "meta",
    placeholder: "muse-spark-1.1",
  },
  {
    id: "kimi",
    label: "Kimi",
    color: "#6a51e6",
    keyName: "kimi",
    placeholder: "kimi-k2.6",
  },
  {
    id: "zai",
    label: "Z.ai",
    color: "#3d63e8",
    keyName: "zai",
    placeholder: "glm-5.2",
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    color: "#7c6ff0",
    keyName: "openrouter",
    placeholder: "openai/gpt-5.5",
  },
  {
    id: "cerebras",
    label: "Cerebras",
    color: "#f15a24",
    keyName: "cerebras",
    placeholder: "gpt-oss-120b",
  },
  {
    id: "local",
    label: "Local",
    color: "#6b7684",
    keyName: "compat",
    placeholder: "model",
    optionalKey: true,
  },
  {
    id: "random",
    label: "Random",
    color: "#8d97a5",
    keyName: null,
    placeholder: "",
  },
  {
    id: "human",
    label: "Human",
    color: "#d9a531",
    keyName: null,
    placeholder: "",
  },
];

export function providerOf(spec: string): ProviderInfo {
  const id = spec.split(":", 1)[0];
  return PROVIDERS.find((p) => p.id === id) ?? byId("random");
}

export function providerOfName(name: string): ProviderInfo | null {
  const lowered = name.toLowerCase();
  if (lowered.startsWith("human")) return byId("human");
  if (lowered.startsWith("random")) return byId("random");
  if (lowered.startsWith("claude")) return byId("anthropic");
  if (lowered.startsWith("gpt") || lowered.startsWith("o1") || lowered.startsWith("o3") || lowered.startsWith("o4"))
    return byId("openai");
  if (lowered.startsWith("gemini")) return byId("google");
  if (lowered.startsWith("grok")) return byId("xai");
  if (lowered.startsWith("deepseek")) return byId("deepseek");
  if (lowered.startsWith("muse")) return byId("meta");
  if (lowered.startsWith("kimi")) return byId("kimi");
  if (lowered.startsWith("glm")) return byId("zai");
  return null;
}

function byId(id: string): ProviderInfo {
  return PROVIDERS.find((p) => p.id === id)!;
}

/* Every provider mark is a hanko-style stamp chip: brand color square,
   bone glyph. Keeps the set cohesive however many providers are added. */
const STAMP = "#fdf6e3";

function stampMark(id: string, c: string) {
  switch (id) {
    case "anthropic":
      return (
        <>
          <path d="M6.4 3.9h2.1l3.7 8.2h-2l-.75-1.85H6L5.25 12.1h-2L6.4 3.9zm.25 4.75h2.15L7.7 6.1 6.65 8.65z" fill={STAMP} />
          <path d="M9.6 3.9h1.9l1.4 3.2-1.5 1.2-1.8-4.4z" fill={STAMP} opacity="0.62" />
        </>
      );
    case "openai":
      return (
        <>
          <path d="M8 2.9l4.4 2.55v5.1L8 13.1l-4.4-2.55v-5.1L8 2.9z" fill="none" stroke={STAMP} stroke-width="1.5" />
          <circle cx="8" cy="8" r="1.5" fill={STAMP} />
        </>
      );
    case "google":
      return (
        <path
          d="M8 2.4c.38 3 2.2 4.82 5.2 5.2v.8c-3 .38-4.82 2.2-5.2 5.2h-.8c-.38-3-2.2-4.82-5.2-5.2v-.8c3-.38 4.82-2.2 5.2-5.2h.8z"
          fill={STAMP}
        />
      );
    case "xai":
      return (
        <>
          <path d="M3.7 3.6h2.3l6.3 8.8h-2.3L3.7 3.6z" fill={STAMP} />
          <path d="M12.3 3.6h-2.2L7.9 6.7l1.15 1.6 3.25-4.7zM3.7 12.4h2.2l2.2-3.1-1.15-1.6-3.25 4.7z" fill={STAMP} opacity="0.62" />
        </>
      );
    case "deepseek":
      return (
        <>
          <path d="M2.6 10.2C4 6.3 7.2 4.1 12.2 4.5c.8.07 1.3.47 1.3 1.1 0 2.5-2.9 6.4-7.6 6.4-1.25 0-2.35-.6-3.3-1.8z" fill={STAMP} />
          <circle cx="11.2" cy="6.1" r="0.85" fill={c} />
        </>
      );
    case "meta":
      return (
        <path
          d="M8 8c-1.1 1.5-2 2.4-3.2 2.4a2.4 2.4 0 0 1 0-4.8c1.2 0 2.1.9 3.2 2.4 1.1-1.5 2-2.4 3.2-2.4a2.4 2.4 0 0 1 0 4.8C10 10.4 9.1 9.5 8 8z"
          fill="none"
          stroke={STAMP}
          stroke-width="1.5"
        />
      );
    case "kimi":
      return <path d="M10.3 2.8a5.5 5.5 0 1 0 2.9 7.2 4.4 4.4 0 0 1-2.9-7.2z" fill={STAMP} />;
    case "zai":
      return <path d="M4.2 3.9h7.6v1.9L7.4 10h4.5v2.1H4V10.2L8.4 6H4.2V3.9z" fill={STAMP} />;
    case "openrouter":
      return (
        <>
          <path d="M2.8 5.5h2c3.3 0 4.6 5 7.5 5" fill="none" stroke={STAMP} stroke-width="1.6" />
          <path d="M2.8 10.5h2c3.3 0 4.6-5 7.5-5" fill="none" stroke={STAMP} stroke-width="1.6" />
          <path d="M11.4 3.3l3 2.2-3 2.2z" fill={STAMP} />
          <path d="M11.4 8.3l3 2.2-3 2.2z" fill={STAMP} />
        </>
      );
    case "cerebras":
      return (
        <>
          <circle cx="8" cy="8" r="4.7" fill={STAMP} />
          <path d="M3.3 6.2h9.4M3.3 9.8h9.4M6.2 3.3v9.4M9.8 3.3v9.4" stroke={c} stroke-width="0.9" />
        </>
      );
    case "local":
      return (
        <>
          <path d="M3.8 5l3.1 3-3.1 3" fill="none" stroke={STAMP} stroke-width="1.8" />
          <rect x="8.4" y="10.4" width="4" height="1.8" fill={STAMP} />
        </>
      );
    case "random":
      return (
        <>
          <circle cx="4.6" cy="4.6" r="1.35" fill={STAMP} />
          <circle cx="11.4" cy="4.6" r="1.35" fill={STAMP} />
          <circle cx="8" cy="8" r="1.45" fill="#c2372e" />
          <circle cx="4.6" cy="11.4" r="1.35" fill={STAMP} />
          <circle cx="11.4" cy="11.4" r="1.35" fill={STAMP} />
        </>
      );
    case "human":
      return (
        <>
          <circle cx="8" cy="5.2" r="2.3" fill={STAMP} />
          <path d="M3.4 12.9c.55-2.7 2.5-4 4.6-4s4.05 1.3 4.6 4H3.4z" fill={STAMP} />
        </>
      );
    default:
      return <circle cx="8" cy="8" r="2.2" fill={STAMP} />;
  }
}

export function ProviderIcon({ provider, size = 16 }: { provider: ProviderInfo; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" style="flex:none">
      <rect x="0.5" y="0.5" width="15" height="15" rx="4.2" fill={provider.color} stroke="rgba(253,246,227,0.35)" />
      {stampMark(provider.id, provider.color)}
    </svg>
  );
}
