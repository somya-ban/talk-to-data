'use client'

import React from 'react'
import { Sparkles, History, Settings2, Sun } from 'lucide-react'
import { cn } from '@/lib/utils'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface AppShellProps {
  /** Content to render inside the main zone (typically ConversationView). */
  children: React.ReactNode
  /** Called when the wordmark is clicked. Optional. */
  onWordmarkClick?: () => void
  /** Called when the History icon is clicked. Optional. */
  onHistoryClick?: () => void
  /** Called when the Settings icon is clicked. Optional. */
  onSettingsClick?: () => void
  /** Called when the theme toggle is clicked. Optional. */
  onThemeToggle?: () => void
  /** Extra classes for the outermost wrapper. */
  className?: string
}

// ─── Component ────────────────────────────────────────────────────────────────

export function AppShell({
  children,
  onWordmarkClick,
  onHistoryClick,
  onSettingsClick,
  onThemeToggle,
  className,
}: AppShellProps) {
  return (
    <div
      className={cn(
        'relative min-h-screen w-full bg-background overflow-x-hidden flex flex-col',
        className
      )}
    >
      {/* ── Atmosphere layer 1 — primary warm halo ─────────────────────────── */}
      <div
        aria-hidden="true"
        className={cn(
            'absolute top-0 left-0 right-0 h-[800px] pointer-events-none z-0',
            'bg-[radial-gradient(ellipse_70%_80%_at_50%_-10%,rgba(99,102,241,0.32),transparent_75%)]'
        )}
      />

      {/* ── Atmosphere layer 2 — subtle foreground warmth bleed ────────────── */}
      <div
        aria-hidden="true"
        className="absolute top-0 left-0 right-0 h-[200px] pointer-events-none z-0 bg-gradient-to-b from-foreground/[0.015] to-transparent"
      />

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header
        className={cn(
          'relative z-10 w-full',
          'border-b border-border/40',
          'bg-background/60 backdrop-blur-md'
        )}
      >
        <div className="flex items-center justify-between max-w-4xl mx-auto w-full px-6 py-4">

          {/* Left zone — wordmark */}
          <button
            type="button"
            onClick={onWordmarkClick}
            className="flex items-center gap-2.5 cursor-pointer rounded focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            aria-label="Talk to Data home"
          >
            <Sparkles
              className="size-4 text-brand shrink-0"
              aria-hidden="true"
            />
            <span className="font-sans text-base font-semibold text-foreground tracking-tight">
              Talk to Data
            </span>
          </button>

          {/* Right zone — header actions */}
          <nav
            className="flex items-center gap-1"
            aria-label="Header actions"
          >
            <button
              type="button"
              onClick={onHistoryClick}
              aria-label="View conversation history"
              className={cn(
                'size-9 rounded-full flex items-center justify-center',
                'text-muted-foreground hover:text-foreground',
                'hover:bg-muted/40 transition-colors duration-200 ease-out',
                'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'
              )}
            >
              <History className="size-4" aria-hidden="true" />
            </button>

            <button
              type="button"
              onClick={onSettingsClick}
              disabled={!onSettingsClick}
              aria-label="Manage metrics"
              className={cn(
                'size-9 rounded-full flex items-center justify-center',
                'transition-colors duration-200 ease-out',
                'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                onSettingsClick
                  ? 'text-muted-foreground hover:text-foreground hover:bg-muted/40 cursor-pointer'
                  : 'text-muted-foreground/30 cursor-default',
              )}
            >
              <Settings2 className="size-4" aria-hidden="true" />
            </button>

            <button
              type="button"
              onClick={onThemeToggle}
              aria-label="Toggle theme"
              className={cn(
                'size-9 rounded-full flex items-center justify-center',
                'text-muted-foreground hover:text-foreground',
                'hover:bg-muted/40 transition-colors duration-200 ease-out',
                'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'
              )}
            >
              <Sun className="size-4" aria-hidden="true" />
            </button>
          </nav>
        </div>
      </header>

      {/* ── Main content zone ───────────────────────────────────────────────── */}
      <main className="relative z-10 flex-1 w-full">
        {children}
      </main>
    </div>
  )
}

// ─── Preview (default export for v0 isolation) ────────────────────────────────

export default function AppShellPreview() {
  return (
    <AppShell
      onWordmarkClick={() => console.log('Wordmark')}
      onHistoryClick={() => console.log('History')}
      onSettingsClick={() => console.log('Settings')}
      onThemeToggle={() => console.log('Theme')}
    >
      <div className="max-w-2xl mx-auto w-full px-6 py-24 flex flex-col gap-8">
        <h1 className="text-4xl font-medium tracking-tight text-foreground text-center text-balance">
          Atmosphere preview
        </h1>
        <p className="text-base text-muted-foreground leading-relaxed text-center max-w-xl mx-auto text-pretty">
          This is sample content rendered inside the shell so you can see how
          the warm halo, the header chrome, and the content zone interact.
          Replace this with ConversationView in production.
        </p>
      </div>
    </AppShell>
  )
}