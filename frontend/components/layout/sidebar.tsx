"use client";

import {
  BarChart3,
  BrainCircuit,
  Database,
  FileBarChart,
  History,
  LayoutDashboard,
  Settings,
  Sparkles,
  X,
} from "lucide-react";

interface SidebarProps {
  mobileOpen: boolean;
  onClose: () => void;
}

const primaryNavigation = [
  {
    label: "Dashboard",
    icon: LayoutDashboard,
    active: true,
  },
  {
    label: "Datasets",
    icon: Database,
    active: false,
  },
  {
    label: "Analyses",
    icon: BarChart3,
    active: false,
  },
  {
    label: "Models",
    icon: BrainCircuit,
    active: false,
  },
];

const secondaryNavigation = [
  {
    label: "Run history",
    icon: History,
  },
  {
    label: "Reports",
    icon: FileBarChart,
  },
];

export function Sidebar({
  mobileOpen,
  onClose,
}: SidebarProps) {
  return (
    <>
      {mobileOpen && (
        <button
          type="button"
          aria-label="Close navigation"
          onClick={onClose}
          className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm lg:hidden"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-[248px] flex-col border-r border-[var(--border)] bg-[var(--surface)] transition-transform duration-200 lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-16 items-center justify-between border-b border-[var(--border)] px-5">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--accent)] text-[var(--accent-foreground)]">
              <Sparkles size={16} strokeWidth={2} />
            </div>

            <div>
              <div className="text-[13px] font-semibold tracking-[-0.01em]">
                Automated EDA
              </div>
              <div className="text-[11px] text-[var(--muted)]">
                & ML
              </div>
            </div>
          </div>

          <button
            type="button"
            aria-label="Close navigation"
            onClick={onClose}
            className="focus-ring flex h-8 w-8 items-center justify-center rounded-lg text-[var(--muted)] hover:bg-[var(--surface-muted)] lg:hidden"
          >
            <X size={17} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-5">
          <div className="mb-3 px-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--muted-foreground)]">
            Workspace
          </div>

          <nav aria-label="Primary navigation" className="space-y-1">
            {primaryNavigation.map((item) => {
              const Icon = item.icon;

              return (
                <button
                  key={item.label}
                  type="button"
                  onClick={onClose}
                  className={`focus-ring flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors ${
                    item.active
                      ? "bg-[var(--surface-muted)] font-medium text-[var(--foreground)]"
                      : "text-[var(--muted)] hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)]"
                  }`}
                >
                  <Icon size={17} strokeWidth={1.8} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="mb-3 mt-8 px-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--muted-foreground)]">
            Insights
          </div>

          <nav aria-label="Insight navigation" className="space-y-1">
            {secondaryNavigation.map((item) => {
              const Icon = item.icon;

              return (
                <button
                  key={item.label}
                  type="button"
                  onClick={onClose}
                  className="focus-ring flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)]"
                >
                  <Icon size={17} strokeWidth={1.8} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        <div className="border-t border-[var(--border)] p-3">
          <button
            type="button"
            className="focus-ring flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)]"
          >
            <Settings size={17} strokeWidth={1.8} />
            <span>Settings</span>
          </button>

          <div className="mt-3 rounded-xl bg-[var(--surface-muted)] p-3">
            <div className="mb-2 flex items-center gap-2">
              <div className="h-1.5 w-1.5 rounded-full bg-[var(--success)]" />
              <span className="text-[11px] font-medium text-[var(--muted)]">
                System operational
              </span>
            </div>

            <p className="text-[11px] leading-4 text-[var(--muted-foreground)]">
              Your analysis environment is ready.
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}