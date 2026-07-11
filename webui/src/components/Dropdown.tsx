import type { ComponentChildren } from "preact";
import { useEffect, useId, useRef, useState } from "preact/hooks";

export interface DropdownOption {
  value: string;
  label: string;
  disabled?: boolean;
  icon?: ComponentChildren;
}

/** Themed replacement for <select>: the OS-native popup can't be styled and
    overlays the control instead of extending below it. */
export function Dropdown({ value, options, onChange, label, up = false }: {
  value: string;
  options: DropdownOption[];
  onChange(value: string): void;
  label?: string;
  up?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const menuId = useId();
  const selected = options.find((option) => option.value === value);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  const openMenu = () => {
    const index = options.findIndex((option) => option.value === value);
    setActive(index === -1 ? 0 : index);
    setOpen(true);
  };

  const move = (delta: number) => {
    setActive((current) => {
      let next = current;
      for (let step = 0; step < options.length; step += 1) {
        next = (next + delta + options.length) % options.length;
        if (!options[next].disabled) return next;
      }
      return current;
    });
  };

  const choose = (index: number) => {
    const option = options[index];
    if (option === undefined || option.disabled) return;
    setOpen(false);
    if (option.value !== value) onChange(option.value);
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (!open) {
      if (event.key === "Enter" || event.key === " " || event.key === "ArrowDown" || event.key === "ArrowUp") {
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
      case " ":
        event.preventDefault();
        choose(active);
        break;
      case "Escape":
      case "Tab":
        setOpen(false);
        break;
    }
  };

  return (
    <div class={`dropdown${open ? " dropdown-open" : ""}`} ref={rootRef}>
      <button
        type="button"
        class="dropdown-button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={label}
        aria-controls={open ? menuId : undefined}
        aria-activedescendant={open ? `${menuId}-${active}` : undefined}
        onClick={() => (open ? setOpen(false) : openMenu())}
        onKeyDown={onKeyDown}
      >
        {selected?.icon}
        <span class="dropdown-label">{selected?.label ?? value}</span>
        <svg class="dropdown-caret" width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
          <path d="M2 3.5l3 3 3-3" fill="none" stroke="currentColor" stroke-width="1.5" />
        </svg>
      </button>
      {open && (
        <div class={`dropdown-menu${up ? " dropdown-menu-up" : ""}`} role="listbox" id={menuId}>
          {options.map((option, index) => (
            <div
              key={option.value}
              id={`${menuId}-${index}`}
              role="option"
              aria-selected={option.value === value}
              aria-disabled={option.disabled || undefined}
              class={`dropdown-option${index === active ? " dropdown-option-active" : ""}${option.disabled ? " dropdown-option-disabled" : ""}`}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => choose(index)}
              onMouseEnter={() => setActive(index)}
            >
              {option.icon}
              <span>{option.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
