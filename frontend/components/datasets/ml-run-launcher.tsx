"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertTriangle,
  BrainCircuit,
  ChevronDown,
  Cloud,
  Monitor,
  Loader2,
  Target,
} from "lucide-react";
import { apiRequest } from "@/lib/api";

interface Dataset {
  dataset_id: number;
  filename: string;
  rows: number;
  columns: number;
  profile: {
    column_names: string[];
  };
}

interface MLRunLauncherProps {
  dataset: Dataset;
}

interface ComputeOptionProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  badge?: string;
  selected?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}

function ComputeOption({
  icon,
  title,
  description,
  badge,
  selected,
  disabled,
  onClick,
}: ComputeOptionProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`w-full rounded-xl border p-4 text-left transition-colors ${
        disabled
          ? "cursor-not-allowed border-[var(--border)] bg-[var(--surface-muted)] opacity-60"
          : selected
            ? "border-[var(--accent)] bg-[var(--surface-muted)] ring-1 ring-[var(--accent)]"
            : "border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-muted)]"
      }`}
    >
      <div className="flex items-start gap-3">
        <div
          className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
            selected
              ? "bg-[var(--accent)] text-[var(--accent-foreground)]"
              : "bg-[var(--surface-muted)] text-[var(--muted)]"
          }`}
        >
          {icon}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">{title}</span>

            {badge && (
              <span className="rounded-full bg-[var(--surface-muted)] px-2 py-0.5 text-[10px] font-medium text-[var(--muted-foreground)]">
                {badge}
              </span>
            )}
          </div>

          <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
            {description}
          </p>
        </div>
      </div>
    </button>
  );
}

export function MLRunLauncher({
  dataset,
}: MLRunLauncherProps) {
  const router = useRouter();
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [computeMode, setComputeMode] = useState<"local" | "cloud">("local");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!targetColumn) {
      setError("Please select a target column.");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const run = await apiRequest<{ run_id: number }>(
        `/datasets/${dataset.dataset_id}/run`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            target_column: targetColumn,
            compute_mode: computeMode,
          }),
        },
      );

      router.push(`/runs/${run.run_id}`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to start ML analysis",
      );
      setIsSubmitting(false);
    }
  };

  return (
    <section className="mt-7">
      <div className="surface overflow-hidden rounded-xl">
        <div className="border-b border-[var(--border)] px-5 py-5 sm:px-6">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--surface-muted)] text-[var(--muted)]">
              <BrainCircuit size={19} strokeWidth={1.8} />
            </div>

            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold">
                  Automated ML
                </h2>
                <span className="inline-flex items-center rounded-full bg-[var(--surface-muted)] px-2 py-0.5 text-[10px] font-medium text-[var(--muted)]">
                  Pipeline
                </span>
              </div>

              <p className="mt-1 max-w-2xl text-xs leading-5 text-[var(--muted)]">
                Automatically train, evaluate, and benchmark machine-learning
                models on this dataset.
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-6 px-5 py-5 sm:px-6">
          {/* 2-Column Balanced Setup: Dataset on Left, Target Column on Right */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* Source Dataset Card */}
            <div className="flex flex-col justify-between rounded-xl border border-[var(--border)] bg-[var(--surface-muted)]/60 p-4">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Source Dataset
                  </span>
                  <span className="text-[11px] text-[var(--muted)]">
                    ID #{dataset.dataset_id}
                  </span>
                </div>

                <div
                  className="mt-2 truncate text-xs font-semibold text-[var(--foreground)]"
                  title={dataset.filename}
                >
                  {dataset.filename}
                </div>
              </div>

              <div className="mt-4 flex items-center gap-3 border-t border-[var(--border)] pt-3 text-[11px] text-[var(--muted)]">
                <span>{dataset.rows.toLocaleString()} rows</span>
                <span>·</span>
                <span>{dataset.columns.toLocaleString()} columns</span>
              </div>
            </div>

            {/* Target Column Selector Card */}
            <div className="flex flex-col justify-between rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4">
              <div>
                <div className="flex items-center justify-between">
                  <label
                    htmlFor="target-column-select"
                    className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]"
                  >
                    <Target size={12} className="text-[var(--accent)]" />
                    Target Column
                  </label>

                  {targetColumn && (
                    <span className="rounded bg-[var(--surface-muted)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--accent)]">
                      Selected
                    </span>
                  )}
                </div>

                <div className="relative mt-2">
                  <select
                    id="target-column-select"
                    value={targetColumn}
                    onChange={(e) => {
                      setTargetColumn(e.target.value);
                      setError(null);
                    }}
                    className="focus-ring w-full cursor-pointer appearance-none rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-2.5 pr-9 text-xs font-medium text-[var(--foreground)] transition-colors hover:border-[var(--muted)] focus:border-[var(--accent)]"
                  >
                    <option value="">Select target column to predict…</option>
                    {dataset.profile?.column_names?.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>

                  <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[var(--muted)]">
                    <ChevronDown size={14} />
                  </div>
                </div>
              </div>

              <div className="mt-3 border-t border-[var(--border)] pt-3 text-[11px] text-[var(--muted)]">
                {targetColumn ? (
                  <span>
                    Target:{" "}
                    <strong className="font-semibold text-[var(--foreground)]">
                      {targetColumn}
                    </strong>
                  </span>
                ) : (
                  <span>Choose the feature to train models to predict</span>
                )}
              </div>
            </div>
          </div>

          <div>
            <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
              Compute mode
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <ComputeOption
                icon={<Monitor size={16} />}
                title="Local"
                description="Run the analysis using your local machine."
                selected={computeMode === "local"}
                onClick={() => setComputeMode("local")}
              />

              <ComputeOption
                icon={<Cloud size={16} />}
                title="Cloud"
                description="Run the analysis using cloud compute."
                badge="Coming soon"
                disabled
              />
            </div>
          </div>

          {error && (
            <div className="flex items-start gap-3 rounded-lg border border-[var(--border)] bg-[var(--danger-surface)] p-3">
              <AlertTriangle
                size={15}
                className="mt-0.5 shrink-0 text-[var(--danger)]"
              />

              <p className="text-xs leading-5 text-[var(--danger)]">
                {error}
              </p>
            </div>
          )}

          <div className="flex flex-col gap-3 border-t border-[var(--border)] pt-4 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs text-[var(--muted)]">
              {targetColumn
                ? `Ready to launch ML analysis targeting "${targetColumn}".`
                : "Select a target column above to begin model training."}
            </p>

            <button
              type="button"
              disabled={isSubmitting || !targetColumn}
              onClick={handleSubmit}
              className="focus-ring inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-5 py-2.5 text-xs font-semibold text-[var(--accent-foreground)] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto"
            >
              {isSubmitting && (
                <Loader2
                  size={14}
                  className="animate-spin"
                />
              )}
              Start ML analysis
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}