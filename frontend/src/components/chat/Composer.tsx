"use client";

import React, { useRef, useEffect, useState, type KeyboardEvent } from "react";
import { Paperclip, ArrowUp, Database, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ComposerProps {
  onSubmit: (question: string) => void;
  disabled?: boolean;
  placeholder?: string;
  tableCount?: number;
}

export function Composer({
  onSubmit,
  disabled = false,
  placeholder = "Ask anything about your data...",
  tableCount = 25,
}: ComposerProps): React.JSX.Element {
  const [value, setValue] = useState<string>("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const LINE_HEIGHT = 24;
  const MAX_LINES = 8;
  const MAX_HEIGHT = LINE_HEIGHT * MAX_LINES;

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [value]);

  const canSubmit = value.trim().length > 0 && !disabled;
  const isTyping = value.length > 0;

  function handleSubmit(): void {
    if (!canSubmit) return;
    onSubmit(value.trim());
    setValue("");
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>): void {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div
      className={cn(
        "w-full max-w-3xl mx-auto flex flex-col gap-2",
        disabled && "opacity-50 pointer-events-none"
      )}
    >
      {/* The pill */}
      <div
        className={cn(
          "flex items-center gap-2 w-full",
          "bg-card border border-border rounded-2xl",
          "pl-3 pr-1.5 py-1.5",
          "transition-[border-color,box-shadow] duration-200 ease-out",
          "focus-within:border-brand focus-within:ring-2 focus-within:ring-brand/20"
        )}
      >
        {/* Paperclip */}
        <Button
          variant="ghost"
          size="icon"
          tabIndex={-1}
          className={cn(
            "size-9 shrink-0 rounded-lg",
            "text-muted-foreground hover:text-foreground",
            "transition-colors duration-150 ease-out",
            "hover:bg-transparent focus-visible:bg-transparent"
          )}
          aria-label="Attach file"
        >
          <Paperclip className="size-4" />
        </Button>

        {/* Textarea — invisible scrollbar, wheel/arrow keys still work */}
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className={cn(
            "flex-1 min-w-0",
            "bg-transparent border-none outline-none ring-0",
            "resize-none overflow-y-auto",
            "text-base text-foreground placeholder:text-muted-foreground",
            "leading-6",
            "py-1.5",
            "[&::-webkit-scrollbar]:hidden",
            "[scrollbar-width:none]"
          )}
          style={{ maxHeight: `${MAX_HEIGHT}px` }}
          aria-label="Ask a question"
        />

        {/* Tables chip — hidden while typing so the textarea can claim full width */}
        {!isTyping && (
          <div
            className={cn(
              "flex items-center gap-1.5 shrink-0",
              "bg-muted text-muted-foreground",
              "px-2.5 py-1 rounded-full",
              "text-xs select-none"
            )}
            aria-label={`${tableCount} tables`}
          >
            <Database className="size-3.5 shrink-0" />
            <span className="font-medium">Tables</span>
            <span className="text-muted-foreground/60">·</span>
            <span>{tableCount}</span>
          </div>
        )}

        {/* Privacy chip — hidden while typing */}
        {!isTyping && (
          <div
            className={cn(
              "flex items-center gap-1.5 shrink-0",
              "bg-muted text-muted-foreground",
              "px-2.5 py-1 rounded-full",
              "text-xs select-none"
            )}
            aria-label="Privacy active"
          >
            <span className="size-1.5 rounded-full bg-success shrink-0" />
            <Shield className="size-3.5 shrink-0" />
            <span className="font-medium">Privacy</span>
          </div>
        )}

        {/* Submit button */}
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!canSubmit}
          aria-label="Submit question"
          className={cn(
            "size-9 shrink-0 rounded-full flex items-center justify-center",
            "transition-colors duration-150 ease-out",
            canSubmit
              ? "bg-brand text-brand-foreground hover:bg-brand/90 cursor-pointer"
              : "bg-muted text-muted-foreground cursor-default"
          )}
        >
          <ArrowUp className="size-4" />
        </button>
      </div>

      {/* Helper text */}
      <p className="text-center text-xs text-muted-foreground select-none">
        Press Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}

export default function ComposerPreview(): React.JSX.Element {
  return (
    <div className="dark min-h-screen bg-background flex items-end justify-center p-6">
      <div className="w-full max-w-3xl pb-8">
        <Composer onSubmit={(q) => console.log("Submitted:", q)} />
      </div>
    </div>
  );
}