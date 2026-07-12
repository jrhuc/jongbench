import { useEffect, useId, useRef, useState } from "preact/hooks";
import * as api from "../api";
import type { ModelEntry } from "../api";
import type { ProviderInfo } from "../providers";

const modelCache = new Map<string, Promise<ModelEntry[]>>();

/** Model id input that lists the provider's available models once its API key
    is entered, while always allowing free text. */
export function ModelCombobox({ provider, apiKey, baseUrl, value, onChange, onModelsChange, label }: {
  provider: ProviderInfo;
  apiKey: string;
  baseUrl?: string;
  value: string;
  onChange(value: string): void;
  onModelsChange(models: ModelEntry[] | null): void;
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [models, setModels] = useState<ModelEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const onModelsChangeRef = useRef(onModelsChange);
  const menuId = useId();

  onModelsChangeRef.current = onModelsChange;

  useEffect(() => {
    setModels(null);
    onModelsChangeRef.current(null);
  }, [provider.id, apiKey.trim(), baseUrl]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  useEffect(() => {
    const key = apiKey.trim();
    if ((!key && provider.id !== "local") || (provider.id === "local" && !baseUrl)) {
      setModels(null);
      onModelsChangeRef.current(null);
      setLoading(false);
      setError(null);
      return;
    }
    const cacheKey = `${provider.id}\0${baseUrl ?? ""}\0${key}`;
    let cancelled = false;
    setLoading(true);
    setError(null);
    const timer = window.setTimeout(() => {
      let pending = modelCache.get(cacheKey);
      if (!pending) {
        pending = api.listModels(provider.id === "local" ? "compat" : provider.id, key, baseUrl).catch((exc: Error) => {
          modelCache.delete(cacheKey);
          throw exc;
        });
        modelCache.set(cacheKey, pending);
      }
      pending.then((entries) => {
        if (cancelled) return;
        setModels(entries);
        onModelsChangeRef.current(entries);
        setLoading(false);
        setError(null);
      }).catch((exc: Error) => {
        if (cancelled) return;
        setModels(null);
        onModelsChangeRef.current(null);
        setLoading(false);
        setError(exc.message);
      });
    }, 350);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [provider.id, apiKey.trim(), baseUrl]);

  const openMenu = () => {
    setQuery("");
    setActive(0);
    setOpen(true);
  };

  const filtered = (() => {
    if (!models) return [] as ModelEntry[];
    const needle = query.trim().toLowerCase();
    if (!needle) return models;
    return models.filter((entry) => entry.id.toLowerCase().includes(needle));
  })();
  const visible = filtered.slice(0, 12);
  const moreCount = filtered.length - visible.length;
  const hasKey = apiKey.trim().length > 0 || provider.id === "local";
  const showOptions = hasKey && !loading && error === null && models !== null;

  const move = (delta: number) => {
    if (visible.length === 0) return;
    setActive((current) => (current + delta + visible.length) % visible.length);
  };

  const choose = (index: number) => {
    const entry = visible[index];
    if (entry === undefined) return;
    onChange(entry.id);
    setOpen(false);
  };

  const onInput = (event: Event) => {
    const next = (event.currentTarget as HTMLInputElement).value;
    onChange(next);
    if (open) {
      setQuery(next);
      setActive(0);
    }
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (!open) {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        openMenu();
      }
      return;
    }
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        move(1);
        break;
      case "ArrowUp":
        event.preventDefault();
        move(-1);
        break;
      case "Enter":
        event.preventDefault();
        if (showOptions && visible.length > 0) choose(active);
        else setOpen(false);
        break;
      case "Escape":
        event.preventDefault();
        setOpen(false);
        break;
      case "Tab":
        setOpen(false);
        break;
    }
  };

  return (
    <div class={`dropdown${open ? " dropdown-open" : ""}`} ref={rootRef}>
      <input
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-activedescendant={open && showOptions && visible[active] ? `${menuId}-${active}` : undefined}
        aria-label={label}
        value={value}
        placeholder={provider.placeholder}
        onFocus={openMenu}
        onClick={() => { if (!open) openMenu(); }}
        onInput={onInput}
        onKeyDown={onKeyDown}
      />
      {open && (
        <div class="dropdown-menu" role="listbox" id={menuId}>
          {!hasKey && (
            <div class="dropdown-note">
              Enter the {provider.label} API key below to browse models
            </div>
          )}
          {hasKey && loading && <div class="dropdown-note">Loading models…</div>}
          {hasKey && error !== null && (
            <div class="dropdown-note dropdown-note-error">{error}</div>
          )}
          {showOptions && visible.length === 0 && (
            <div class="dropdown-note">
              No matching models — press Enter to use "{query}"
            </div>
          )}
          {showOptions && visible.map((entry, index) => (
            <div
              key={entry.id}
              id={`${menuId}-${index}`}
              role="option"
              aria-selected={entry.id === value}
              class={`dropdown-option${index === active ? " dropdown-option-active" : ""}`}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => choose(index)}
              onMouseEnter={() => setActive(index)}
            >
              <span>{entry.id}</span>
            </div>
          ))}
          {showOptions && moreCount > 0 && (
            <div class="dropdown-note">…{moreCount} more — type to filter</div>
          )}
        </div>
      )}
    </div>
  );
}
