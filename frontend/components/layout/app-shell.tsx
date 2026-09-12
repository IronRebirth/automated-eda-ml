"use client";

import { useEffect, useState } from "react";
import { Menu, Moon, Sun } from "lucide-react";
import { usePathname } from "next/navigation";

import { Sidebar } from "@/components/layout/sidebar";

interface AppShellProps {
  children: React.ReactNode;
}

function getInitialDarkMode() {
  if (typeof window === "undefined") {
    return false;
  }

  const storedTheme =
    window.localStorage.getItem("theme");

  if (storedTheme === "dark") {
    return true;
  }

  if (storedTheme === "light") {
    return false;
  }

  return window.matchMedia(
    "(prefers-color-scheme: dark)",
  ).matches;
}

function getPageLabel(pathname: string): string {
  if (pathname.startsWith("/runs")) {
    return "Run history";
  }

  if (pathname.startsWith("/datasets")) {
    return "Datasets";
  }

  return "Dashboard";
}

export function AppShell({
  children,
}: AppShellProps) {
  const pathname = usePathname();
  const [darkMode, setDarkMode] = useState(
    getInitialDarkMode,
  );
  const [mobileMenuOpen, setMobileMenuOpen] =
    useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle(
      "dark",
      darkMode,
    );
  }, [darkMode]);

  function toggleTheme() {
    const nextDarkMode = !darkMode;

    window.localStorage.setItem(
      "theme",
      nextDarkMode ? "dark" : "light",
    );

    setDarkMode(nextDarkMode);
  }

  return (
    <div className="min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      <Sidebar
        mobileOpen={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
      />

      <div className="lg:pl-[248px]">
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-[var(--border)] bg-[var(--background)]/90 px-4 backdrop-blur-xl sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <button
              type="button"
              aria-label="Open navigation"
              onClick={() =>
                setMobileMenuOpen(true)
              }
              className="focus-ring flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border)] text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)] lg:hidden"
            >
              <Menu
                size={18}
                strokeWidth={1.8}
              />
            </button>

            <div className="hidden items-center gap-2 text-sm text-[var(--muted)] sm:flex">
              <span>Workspace</span>

              <span className="text-[var(--border-strong)]">
                /
              </span>

              <span className="font-medium text-[var(--foreground)]">
                {getPageLabel(pathname)}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              aria-label="Toggle color theme"
              onClick={toggleTheme}
              className="focus-ring flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border)] text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)]"
            >
              <Moon
                size={17}
                strokeWidth={1.8}
                className="dark:hidden"
                aria-hidden="true"
              />

              <Sun
                size={17}
                strokeWidth={1.8}
                className="hidden dark:block"
                aria-hidden="true"
              />
            </button>

            <div className="ml-1 flex h-9 w-9 items-center justify-center rounded-full bg-[var(--accent)] text-xs font-semibold text-[var(--accent-foreground)]">
              T
            </div>
          </div>
        </header>

        <main>{children}</main>
      </div>
    </div>
  );
}