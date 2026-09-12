"use client";

import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Cpu,
  History,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { apiRequest } from "@/lib/api";

interface Run {
  run_id: number;
  dataset_id: number;
  dataset_filename: string | null;
  status: string;
  compute_mode: string;
  current_stage: string;
  progress: number;
  stage_message: string;
  target_column: string | null;
  test_size: number;
  random_state: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  result: Record<string, unknown> | null;
}

interface RunsResponse {
  runs: Run[];
  limit: number;
  offset: number;
  count: number;
  total_count: number;
}

const PAGE_SIZE = 20;

export default function RunsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [totalCount, setTotalCount] = useState(0);

  const loadRuns = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await apiRequest<RunsResponse>(
        `/runs?limit=${PAGE_SIZE}&offset=${offset}`,
        {
          cache: "no-store",
        },
      );

      setRuns(data.runs);
      setTotalCount(data.total_count);
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "Unable to load run history.",
      );
    } finally {
      setLoading(false);
    }
  }, [offset]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadRuns();
    }, 0);

    return () => {
      window.clearTimeout(timer);
    };
  }, [loadRuns]);

  const hasPreviousPage = offset > 0;
  const hasNextPage = offset + runs.length < totalCount;

  const pageNumber = useMemo(
    () => Math.floor(offset / PAGE_SIZE) + 1,
    [offset],
  );

  const totalPages = Math.max(
    1,
    Math.ceil(totalCount / PAGE_SIZE),
  );

  function goToPreviousPage() {
    if (!hasPreviousPage) {
      return;
    }

    setOffset((currentOffset) =>
      Math.max(0, currentOffset - PAGE_SIZE),
    );
  }

  function goToNextPage() {
    if (!hasNextPage) {
      return;
    }

    setOffset(
      (currentOffset) => currentOffset + PAGE_SIZE,
    );
  }

  function refresh() {
    void loadRuns();
  }

  return (
    <div className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
      <section>
        <div className="flex flex-col gap-5 border-b border-[var(--border)] pb-7 sm:flex-row sm:items-end sm:justify-between">
          <div className="flex min-w-0 items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[var(--surface-muted)] text-[var(--muted)]">
              <History size={20} strokeWidth={1.8} />
            </div>

            <div className="min-w-0">
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Insights
              </div>

              <h1 className="mt-1 text-2xl font-semibold tracking-[-0.025em] sm:text-3xl">
                Run history
              </h1>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">
                Review previous machine learning analyses,
                their status, progress, and the datasets they
                belong to.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={refresh}
            disabled={loading}
            className="focus-ring inline-flex h-9 shrink-0 items-center justify-center gap-2 rounded-lg border border-[var(--border)] px-3 text-xs font-medium text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RefreshCw
              size={14}
              className={loading ? "animate-spin" : ""}
            />
            Refresh
          </button>
        </div>
      </section>

      <section className="mt-7">
        {loading && runs.length === 0 ? (
          <LoadingState />
        ) : error ? (
          <ErrorState
            message={error}
            onRetry={refresh}
          />
        ) : runs.length === 0 ? (
          <EmptyState />
        ) : (
          <>
            <div className="surface overflow-hidden rounded-xl">
              <div className="border-b border-[var(--border)] px-5 py-4 sm:px-6">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <div className="text-sm font-semibold">
                      Analyses
                    </div>

                    <div className="mt-1 text-xs text-[var(--muted)]">
                      Page {pageNumber} of {totalPages}
                    </div>
                  </div>

                  <div className="text-xs text-[var(--muted)]">
                    {totalCount} total run
                    {totalCount === 1 ? "" : "s"}
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full min-w-[980px] text-left">
                  <thead>
                    <tr className="border-b border-[var(--border)]">
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)] sm:px-6">
                        Run
                      </th>

                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                        Dataset
                      </th>

                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                        Target
                      </th>

                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                        Status
                      </th>

                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                        Compute
                      </th>

                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                        Created
                      </th>

                      <th className="px-5 py-3 text-right text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                        Details
                      </th>
                    </tr>
                  </thead>

                  <tbody className="divide-y divide-[var(--border)]">
                    {runs.map((run) => (
                      <tr
                        key={run.run_id}
                        className="transition-colors hover:bg-[var(--surface-muted)]"
                      >
                        <td className="px-5 py-4 sm:px-6">
                          <div className="text-sm font-semibold">
                            #{run.run_id}
                          </div>

                          <div className="mt-1 text-[11px] text-[var(--muted)]">
                            {run.status === "running" ||
                            run.status === "pending" ||
                            run.status === "queued"
                              ? `${run.progress}% complete`
                              : run.completed_at
                                ? formatDuration(
                                    run.created_at,
                                    run.completed_at,
                                  )
                                : "Analysis run"}
                          </div>
                        </td>

                        <td className="px-5 py-4">
                          <div className="max-w-[220px]">
                            <div
                              className="truncate text-xs font-medium"
                              title={
                                run.dataset_filename ??
                                `Dataset #${run.dataset_id}`
                              }
                            >
                              {run.dataset_filename ??
                                `Dataset #${run.dataset_id}`}
                            </div>

                            <div className="mt-1 text-[10px] text-[var(--muted)]">
                              Dataset #{run.dataset_id}
                            </div>
                          </div>
                        </td>

                        <td className="px-5 py-4">
                          <span className="text-xs text-[var(--muted)]">
                            {run.target_column ?? "—"}
                          </span>
                        </td>

                        <td className="px-5 py-4">
                          <StatusCell run={run} />
                        </td>

                        <td className="px-5 py-4">
                          <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
                            <Cpu size={13} />

                            <span>
                              {formatLabel(
                                run.compute_mode,
                              )}
                            </span>
                          </div>
                        </td>

                        <td className="px-5 py-4">
                          <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
                            <Clock3 size={13} />
                            {formatDate(run.created_at)}
                          </div>
                        </td>

                        <td className="px-5 py-4 text-right">
                          <a
                            href={`/runs/${run.run_id}`}
                            className="focus-ring inline-flex items-center rounded-md px-2.5 py-1.5 text-xs font-medium text-[var(--foreground)] hover:bg-[var(--surface)]"
                          >
                            View run
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-xs text-[var(--muted)]">
                Showing {offset + 1}–
                {Math.min(
                  offset + runs.length,
                  totalCount,
                )}{" "}
                of {totalCount}
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={goToPreviousPage}
                  disabled={!hasPreviousPage || loading}
                  className="focus-ring inline-flex h-9 items-center gap-1 rounded-lg border border-[var(--border)] px-3 text-xs font-medium text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)] disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <ChevronLeft size={14} />
                  Previous
                </button>

                <button
                  type="button"
                  onClick={goToNextPage}
                  disabled={!hasNextPage || loading}
                  className="focus-ring inline-flex h-9 items-center gap-1 rounded-lg border border-[var(--border)] px-3 text-xs font-medium text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)] disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Next
                  <ChevronRight size={14} />
                </button>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function StatusCell({
  run,
}: {
  run: Run;
}) {
  const normalizedStatus = run.status.toLowerCase();

  if (
    normalizedStatus === "running" ||
    normalizedStatus === "pending" ||
    normalizedStatus === "queued"
  ) {
    return (
      <div className="min-w-[150px]">
        <StatusBadge status={run.status} />

        <div className="mt-2">
          <div className="h-1.5 overflow-hidden rounded-full bg-[var(--surface-muted)]">
            <div
              className="h-full rounded-full bg-[var(--accent)] transition-[width]"
              style={{
                width: `${Math.min(
                  100,
                  Math.max(0, run.progress),
                )}%`,
              }}
            />
          </div>

          <div className="mt-1 text-[10px] text-[var(--muted)]">
            {run.progress}% ·{" "}
            {formatLabel(run.current_stage)}
          </div>
        </div>
      </div>
    );
  }

  return <StatusBadge status={run.status} />;
}

function StatusBadge({
  status,
}: {
  status: string;
}) {
  const normalizedStatus = status.toLowerCase();

  if (normalizedStatus === "completed") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--success-surface)] px-2.5 py-1 text-[10px] font-semibold text-[var(--success)]">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--success)]" />
        Completed
      </span>
    );
  }

  if (normalizedStatus === "failed") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--danger-surface)] px-2.5 py-1 text-[10px] font-semibold text-[var(--danger)]">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--danger)]" />
        Failed
      </span>
    );
  }

  if (normalizedStatus === "cancelled") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--surface-muted)] px-2.5 py-1 text-[10px] font-semibold text-[var(--muted)]">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--muted)]" />
        Cancelled
      </span>
    );
  }

  if (
    normalizedStatus === "running" ||
    normalizedStatus === "pending" ||
    normalizedStatus === "queued"
  ) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--warning-surface)] px-2.5 py-1 text-[10px] font-semibold text-[var(--warning)]">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--warning)]" />
        {formatLabel(status)}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--surface-muted)] px-2.5 py-1 text-[10px] font-semibold text-[var(--muted)]">
      {formatLabel(status)}
    </span>
  );
}

function LoadingState() {
  return (
    <div className="surface rounded-xl p-10 text-center">
      <Loader2
        size={20}
        className="mx-auto animate-spin text-[var(--muted)]"
      />

      <p className="mt-3 text-xs text-[var(--muted)]">
        Loading run history…
      </p>
    </div>
  );
}

function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="surface rounded-xl p-6 sm:p-8">
      <div className="flex items-start gap-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--danger-surface)] text-[var(--danger)]">
          <AlertTriangle size={18} />
        </div>

        <div className="min-w-0">
          <h2 className="text-sm font-semibold">
            Unable to load run history
          </h2>

          <p className="mt-1.5 text-xs leading-5 text-[var(--muted)]">
            {message}
          </p>

          <button
            type="button"
            onClick={onRetry}
            className="focus-ring mt-4 inline-flex h-8 items-center gap-2 rounded-lg border border-[var(--border)] px-3 text-xs font-medium hover:bg-[var(--surface-muted)]"
          >
            <RefreshCw size={13} />
            Try again
          </button>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="surface rounded-xl p-10 text-center sm:p-14">
      <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--surface-muted)] text-[var(--muted)]">
        <History size={20} />
      </div>

      <h2 className="mt-4 text-sm font-semibold">
        No analysis runs yet
      </h2>

      <p className="mx-auto mt-2 max-w-md text-xs leading-5 text-[var(--muted)]">
        Once you run an automated ML analysis, it will
        appear here with its status and creation time.
      </p>
    </div>
  );
}

function formatLabel(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) =>
      character.toUpperCase(),
    );
}

function formatDate(value: string): string {
  const normalizedValue =
    /(?:Z|[+-]\d{2}:\d{2})$/.test(value)
      ? value
      : `${value}Z`;

  const date = new Date(normalizedValue);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function formatDuration(
  startedAt: string,
  completedAt: string,
): string {
  const startValue =
    /(?:Z|[+-]\d{2}:\d{2})$/.test(startedAt)
      ? startedAt
      : `${startedAt}Z`;

  const endValue =
    /(?:Z|[+-]\d{2}:\d{2})$/.test(completedAt)
      ? completedAt
      : `${completedAt}Z`;

  const start = new Date(startValue);
  const end = new Date(endValue);

  if (
    Number.isNaN(start.getTime()) ||
    Number.isNaN(end.getTime())
  ) {
    return "Analysis run";
  }

  const durationSeconds = Math.max(
    0,
    Math.round(
      (end.getTime() - start.getTime()) / 1000,
    ),
  );

  if (durationSeconds < 60) {
    return `${durationSeconds}s duration`;
  }

  const minutes = Math.floor(
    durationSeconds / 60,
  );
  const seconds = durationSeconds % 60;

  if (minutes < 60) {
    return `${minutes}m ${seconds}s duration`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  return `${hours}h ${remainingMinutes}m duration`;
}