"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";
import { BarChart3, Layers, TrendingUp, Users } from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SuggestedChip {
  /** The full question text. This is what gets passed to onSelect when clicked. */
  question: string;
  /** A short display label for the chip. Optional — if omitted, renders the full question. */
  label?: string;
  /** A lucide icon component to render as the leading affordance. */
  icon: LucideIcon;
}

export interface SuggestedChipsProps {
  /** The chips to display. */
  chips: SuggestedChip[];
  /** Called when a chip is clicked, with the full question text. */
  onSelect: (question: string) => void;
  /** Extra classes for the outer wrapper */
  className?: string;
}

// ─── Component ────────────────────────────────────────────────────────────────

export function SuggestedChips({ chips, onSelect, className }: SuggestedChipsProps) {
  return (
    <div
      className={cn(
        "flex flex-wrap justify-center gap-2",
        className,
      )}
    >
      {chips.map((chip) => {
        const Icon = chip.icon;
        const displayText = chip.label ?? chip.question;

        return (
          <motion.button
            key={chip.question}
            type="button"
            onClick={() => onSelect(chip.question)}
            whileTap={{ scale: 0.95 }}
            transition={{ duration: 0.12, ease: "easeOut" }}
            className={cn(
              "group inline-flex items-center gap-2",
              "px-4 py-2.5 rounded-full",
              "card-surface-subtle",
              "text-sm font-sans font-normal text-muted-foreground",
              "transition-all duration-200 ease-out",
              "hover:text-foreground",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background",
              "active:scale-95",
            )}
          >
            <Icon
              className={cn(
                "size-3.5 shrink-0",
                "text-muted-foreground transition-colors duration-200 ease-out",
                "group-hover:text-foreground",
              )}
              aria-hidden="true"
            />
            <span>{displayText}</span>
          </motion.button>
        );
      })}
    </div>
  );
}

// ─── v0 preview wrapper ──────────────────────────────────────────────────────

export default function SuggestedChipsPreview() {
  const sampleChips: SuggestedChip[] = [
    {
      question: "What's our total balance across all customer accounts?",
      label: "Total balance across customer accounts",
      icon: BarChart3,
    },
    {
      question: "Show me the top 10 customers by trade volume in Q1 this year",
      label: "Top 10 customers by trade volume",
      icon: Users,
    },
    {
      question: "How has cash inflow changed over the last 6 months?",
      label: "Cash inflow trend over 6 months",
      icon: TrendingUp,
    },
    {
      question: "Compare account balances by region",
      label: "Account balances by region",
      icon: Layers,
    },
  ];

  return (
    <div className="min-h-screen bg-background pt-32 px-6">
      <div className="max-w-2xl mx-auto w-full">
        <SuggestedChips
          chips={sampleChips}
          onSelect={(q) => console.log("Selected:", q)}
        />
      </div>
    </div>
  );
}