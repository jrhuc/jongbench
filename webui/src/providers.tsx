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
    color: "#74aa9c",
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
    color: "#b8bcc4",
    keyName: "xai",
    placeholder: "grok-4",
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    color: "#536dfe",
    keyName: "deepseek",
    placeholder: "deepseek-chat",
  },
  {
    id: "meta",
    label: "Meta",
    color: "#4f7cff",
    keyName: "meta",
    placeholder: "muse-spark-1.1",
  },
  {
    id: "kimi",
    label: "Kimi",
    color: "#7957d5",
    keyName: "kimi",
    placeholder: "kimi-k2.6",
  },
  {
    id: "zai",
    label: "Z.ai",
    color: "#5b8def",
    keyName: "zai",
    placeholder: "glm-5.2",
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    color: "#8b7cf6",
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
    color: "#8d97a5",
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
    color: "#d4a94e",
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

export function ProviderIcon({ provider, size = 16 }: { provider: ProviderInfo; size?: number }) {
  const s = size;
  const c = provider.color;
  switch (provider.id) {
    case "anthropic":
      return (
        <svg width={s} height={s} viewBox="0 0 16 16" aria-hidden="true">
          <path d="M6.1 3h2.4l4.4 10h-2.3L9.7 10.7H5.9L5 13H2.7L6.1 3zm.6 5.8h2.4L7.9 5.6 6.7 8.8z" fill={c} />
          <path d="M9.9 3h2.2l1.6 3.8-1.7 1.4L9.9 3z" fill={c} opacity="0.55" />
        </svg>
      );
    case "openai":
      return (
        <svg width={s} height={s} viewBox="0 0 16 16" aria-hidden="true">
          <path
            d="M8 1.8l5.4 3.1v6.2L8 14.2l-5.4-3.1V4.9L8 1.8zm0 2L4.6 5.9v4.2L8 12.2l3.4-2.1V5.9L8 3.8z"
            fill={c}
          />
          <circle cx="8" cy="8" r="1.7" fill={c} />
        </svg>
      );
    case "google":
      return (
        <svg width={s} height={s} viewBox="0 0 16 16" aria-hidden="true">
          <circle cx="5.5" cy="5.5" r="2.6" fill="#4285f4" />
          <circle cx="10.5" cy="5.5" r="2.6" fill="#ea4335" />
          <circle cx="5.5" cy="10.5" r="2.6" fill="#34a853" />
          <circle cx="10.5" cy="10.5" r="2.6" fill="#fbbc05" />
        </svg>
      );
    case "xai":
      return (
        <svg width={s} height={s} viewBox="0 0 16 16" aria-hidden="true">
          <path d="M3 3h2.6L13 13h-2.6L3 3z" fill={c} />
          <path d="M13 3h-2.6L7.9 6.4l1.3 1.7L13 3zM3 13h2.6l2.5-3.4-1.3-1.7L3 13z" fill={c} opacity="0.6" />
        </svg>
      );
    case "deepseek":
      return (
        <svg width={s} height={s} viewBox="0 0 16 16" aria-hidden="true">
          <path
            d="M2 10.5C3.5 6 7 3.5 12.5 4c.9.1 1.5.5 1.5 1.2 0 2.8-3.2 7.3-8.5 7.3-1.4 0-2.6-.7-3.5-2z"
            fill={c}
          />
          <circle cx="11.4" cy="6" r="0.9" fill="#0b1512" />
        </svg>
      );
    case "human":
      return (
        <svg width={s} height={s} viewBox="0 0 16 16" aria-hidden="true">
          <circle cx="8" cy="5" r="2.6" fill={c} />
          <path d="M2.8 13.6c.6-3 2.8-4.4 5.2-4.4s4.6 1.4 5.2 4.4H2.8z" fill={c} />
        </svg>
      );
    default:
      return (
        <svg width={s} height={s} viewBox="0 0 16 16" aria-hidden="true">
          <rect x="2.2" y="2.2" width="11.6" height="11.6" rx="2.4" fill="none" stroke={c} stroke-width="1.6" />
          <circle cx="5.6" cy="5.6" r="1.1" fill={c} />
          <circle cx="10.4" cy="10.4" r="1.1" fill={c} />
          <circle cx="8" cy="8" r="1.1" fill={c} />
        </svg>
      );
  }
}
