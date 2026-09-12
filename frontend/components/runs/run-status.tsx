"use client";

import {
  AlertTriangle,
  CheckCircle2,
  CheckCircle,
  Clock3,
  Cloud,
  Loader2,
  Monitor,
  RefreshCw,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { apiRequest } from "@/lib/api";

interface RunStatusData {
  run_id: number;
  dataset_id: number;
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
  result: unknown | null;
}

interface RunStatusProps {
  run: RunStatusData;
  datasetName?: string | null;
}

interface PipelineStage {
  key: string;
  title: string;
}

const POLL_INTERVAL_MS = 3000;

const PIPELINE_STAGES: PipelineStage[] = [
  {
    key: "loading",
    title: "Dataset loaded",
  },
  {
    key: "profiling",
    title: "Dataset profiled",
  },
  {
    key: "preprocessing",
    title: "Preprocessing complete",
  },
  {
    key: "training",
    title: "Models trained",
  },
  {
    key: "cross_validation",
    title: "Cross-validation complete",
  },
  {
    key: "optimization",
    title: "Hyperparameters optimized",
  },
  {
    key: "evaluation",
    title: "Models evaluated",
  },
  {
    key: "explainability",
    title: "SHAP explainability complete",
  },
  {
    key: "saving",
    title: "Results being saved",
  },
];

const STAGE_ORDER = PIPELINE_STAGES.map(
  (stage) => stage.key,
);

export function RunStatus({
  run: initialRun,
  datasetName,
}: RunStatusProps) {
  const [run, setRun] = useState(initialRun);
  const [now, setNow] = useState(() => Date.now());
  const [pollError, setPollError] = useState<string | null>(
    null,
  );

  const isActive =
    run.status === "pending" ||
    run.status === "queued" ||
    run.status === "running";

  const refreshRun = useCallback(async () => {
    try {
      const latestRun = await apiRequest<RunStatusData>(
        `/runs/${initialRun.run_id}`,
        {
          cache: "no-store",
        },
      );

      setRun(latestRun);
      setPollError(null);

      if (
        latestRun.status === "completed" &&
        latestRun.result
      ) {
        window.location.reload();
      }
    } catch (error) {
      setPollError(
        error instanceof Error
          ? error.message
          : "Unable to refresh run status.",
      );
    }
  }, [initialRun.run_id]);

  useEffect(() => {
    if (!isActive) {
      return;
    }

    const interval = window.setInterval(
      refreshRun,
      POLL_INTERVAL_MS,
    );

    return () => {
      window.clearInterval(interval);
    };
  }, [isActive, refreshRun]);

  useEffect(() => {
    if (!isActive || run.started_at === null) {
      return;
    }

    const interval = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);

    return () => {
      window.clearInterval(interval);
    };
  }, [isActive, run.started_at]);

  const elapsedSeconds = useMemo(
    () => calculateElapsedSeconds(run, now),
    [run, now],
  );

  const statusCopy = useMemo(
    () => getStatusCopy(run.status),
    [run.status],
  );

  if (run.status === "completed" && run.result) {
    return null;
  }

  if (run.status === "failed") {
    return (
      <div className="mx-auto max-w-[900px] px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
        <div className="surface rounded-2xl p-6 sm:p-8">
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--danger-surface)] text-[var(--danger)]">
              <AlertTriangle size={19} />
            </div>

            <div className="min-w-0">
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Machine learning run
              </div>

              <h1 className="mt-1 text-2xl font-semibold tracking-[-0.025em]">
                Analysis failed
              </h1>

              <p className="mt-2 max-w-xl text-sm leading-6 text-[var(--muted)]">
                Run #{run.run_id}
                {datasetName
                  ? ` for ${datasetName}`
                  : ""}{" "}
                did not complete successfully.
              </p>

              <div className="mt-5 inline-flex items-center rounded-full bg-[var(--danger-surface)] px-3 py-1.5 text-xs font-medium text-[var(--danger)]">
                Status: Failed
              </div>

              {run.stage_message && (
                <p className="mt-4 text-xs leading-5 text-[var(--muted)]">
                  {run.stage_message}
                </p>
              )}

              {pollError && (
                <p className="mt-4 text-xs text-[var(--muted)]">
                  {pollError}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (run.status === "cancelled") {
    return (
      <div className="mx-auto max-w-[900px] px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
        <div className="surface rounded-2xl p-6 sm:p-8">
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--surface-muted)] text-[var(--muted)]">
              <Clock3 size={19} />
            </div>

            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Machine learning run
              </div>

              <h1 className="mt-1 text-2xl font-semibold">
                Analysis cancelled
              </h1>

              <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                Run #{run.run_id} was cancelled before
                completion.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[900px] px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
      <div className="surface overflow-hidden rounded-2xl">
        <div className="border-b border-[var(--border)] p-6 sm:p-8">
          <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex min-w-0 items-start gap-4">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[var(--surface-muted)] text-[var(--muted)]">
                {run.compute_mode === "cloud" ? (
                  <Cloud size={20} />
                ) : (
                  <Monitor size={20} />
                )}
              </div>

              <div className="min-w-0">
                <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                  Machine learning run
                </div>

                <h1 className="mt-1 text-2xl font-semibold tracking-[-0.025em] sm:text-3xl">
                  {statusCopy.title}
                </h1>

                <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                  Run #{run.run_id}
                  {datasetName
                    ? ` · ${datasetName}`
                    : ""}
                </p>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-2 self-start">
              <div className="inline-flex items-center gap-2 rounded-full bg-[var(--warning-surface)] px-3 py-1.5 text-xs font-medium text-[var(--warning)]">
                <span className="h-2 w-2 animate-pulse rounded-full bg-[var(--warning)]" />
                {statusCopy.label}
              </div>

              <div className="inline-flex items-center gap-1.5 rounded-full bg-[var(--surface-muted)] px-3 py-1.5 text-xs font-medium text-[var(--muted)]">
                {run.compute_mode === "cloud" ? (
                  <Cloud size={12} />
                ) : (
                  <Monitor size={12} />
                )}
                {run.compute_mode === "cloud"
                  ? "Cloud"
                  : "Local"}
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 sm:p-8">
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-muted)] p-5 sm:p-6">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="text-sm font-semibold">
                  {getStageTitle(run.current_stage)}
                </div>

                <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
                  {run.stage_message}
                </p>
              </div>

              <div className="shrink-0 text-right">
                <div className="text-lg font-semibold tabular-nums">
                  {run.progress}%
                </div>

                <div className="text-[10px] text-[var(--muted-foreground)]">
                  complete
                </div>
              </div>
            </div>

            <div className="mt-5 h-2 overflow-hidden rounded-full bg-[var(--border)]">
              <div
                className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-700 ease-out"
                style={{
                  width: `${Math.max(
                    0,
                    Math.min(100, run.progress),
                  )}%`,
                }}
              />
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
              <div>
                <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                  Running time
                </div>

                <div className="mt-1 text-sm font-semibold tabular-nums">
                  {run.started_at
                    ? formatElapsedTime(elapsedSeconds)
                    : "Not started"}
                </div>
              </div>

              <div>
                <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                  {run.started_at
                    ? "Started"
                    : "Requested"}
                </div>

                <div className="mt-1 text-sm font-medium">
                  {formatLocalDate(
                    run.started_at ?? run.created_at,
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-xs font-semibold">
                Pipeline
              </div>

              <div className="text-[10px] text-[var(--muted)]">
                Live status
              </div>
            </div>

            <div className="overflow-hidden rounded-xl border border-[var(--border)]">
              {PIPELINE_STAGES.map(
                (stage, index) => {
                  const stageState =
                    getStageState(
                      stage.key,
                      run.current_stage,
                      run.status,
                    );

                  return (
                    <div
                      key={stage.key}
                      className={`flex items-center gap-3 px-4 py-3 ${
                        index < PIPELINE_STAGES.length - 1
                          ? "border-b border-[var(--border)]"
                          : ""
                      }`}
                    >
                      <div className="flex h-6 w-6 shrink-0 items-center justify-center">
                        {stageState === "completed" ? (
                          <CheckCircle
                            size={16}
                            className="text-[var(--success)]"
                          />
                        ) : stageState === "active" ? (
                          <Loader2
                            size={16}
                            className="animate-spin text-[var(--accent)]"
                          />
                        ) : (
                          <span className="h-1.5 w-1.5 rounded-full bg-[var(--border-strong)]" />
                        )}
                      </div>

                      <div
                        className={`text-xs ${
                          stageState === "pending"
                            ? "text-[var(--muted)]"
                            : "font-medium"
                        }`}
                      >
                        {stage.title}
                      </div>

                      {stageState === "active" && (
                        <span className="ml-auto text-[10px] font-medium text-[var(--accent)]">
                          In progress
                        </span>
                      )}
                    </div>
                  );
                },
              )}
            </div>
          </div>

          <div className="mt-6 flex items-start gap-3 rounded-xl border border-[var(--border)] p-4">
            <CheckCircle2
              size={17}
              className="mt-0.5 shrink-0 text-[var(--success)]"
            />

            <div>
              <div className="text-xs font-medium">
                You can safely leave this page.
              </div>

              <p className="mt-1 text-[11px] leading-5 text-[var(--muted)]">
                The worker continues the analysis
                independently. This page checks for updates
                automatically.
              </p>
            </div>
          </div>

          {pollError && (
            <div className="mt-5 flex items-start gap-3 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-3.5">
              <RefreshCw
                size={16}
                className="mt-0.5 shrink-0 text-[var(--muted)]"
              />

              <div>
                <div className="text-xs font-medium">
                  Waiting to reconnect
                </div>

                <p className="mt-1 text-[11px] leading-5 text-[var(--muted)]">
                  We could not refresh the run status. We&apos;ll
                  automatically try again.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function getStageTitle(stage: string): string {
  if (stage === "queued") {
    return "Waiting to start";
  }

  const matchingStage = PIPELINE_STAGES.find(
    (item) => item.key === stage,
  );

  return matchingStage?.title ?? "Analysis in progress";
}

function getStageState(
  stage: string,
  currentStage: string,
  status: string,
): "completed" | "active" | "pending" {
  if (status === "completed") {
    return "completed";
  }

  if (currentStage === stage) {
    return "active";
  }

  if (currentStage === "completed") {
    return "completed";
  }

  const currentIndex =
    STAGE_ORDER.indexOf(currentStage);

  const stageIndex =
    STAGE_ORDER.indexOf(stage);

  if (
    currentIndex >= 0 &&
    stageIndex >= 0 &&
    stageIndex < currentIndex
  ) {
    return "completed";
  }

  return "pending";
}

function getStatusCopy(status: string): {
  title: string;
  label: string;
  description: string;
} {
  switch (status) {
    case "pending":
      return {
        title: "Preparing analysis",
        label: "Queued",
        description:
          "Your analysis is waiting for an available ML worker.",
      };

    case "queued":
      return {
        title: "Preparing analysis",
        label: "Queued",
        description:
          "Your analysis is waiting for an available ML worker.",
      };

    case "running":
      return {
        title: "Analysis in progress",
        label: "Running",
        description:
          "The ML worker is processing your dataset.",
      };

    default:
      return {
        title: "Analysis in progress",
        label: "Running",
        description:
          "The ML worker is processing your dataset.",
      };
  }
}

/**
 * Ensure a backend timestamp is parsed as UTC.
 *
 * The API returns naive ISO strings (no Z suffix) that represent
 * UTC values.  Without the suffix, `new Date()` interprets them as
 * local time, which shifts elapsed-time calculations.
 */
function parseUTC(value: string): number {
  const normalized = /(?:Z|[+-]\d{2}:\d{2})$/.test(value)
    ? value
    : `${value}Z`;

  return new Date(normalized).getTime();
}

function calculateElapsedSeconds(
  run: RunStatusData,
  now: number,
): number {
  if (run.started_at === null) {
    return 0;
  }

  const start = parseUTC(run.started_at);

  if (Number.isNaN(start)) {
    return 0;
  }

  const end =
    run.completed_at !== null
      ? parseUTC(run.completed_at)
      : now;

  if (Number.isNaN(end)) {
    return 0;
  }

  return Math.max(
    0,
    Math.floor((end - start) / 1000),
  );
}

function formatElapsedTime(
  totalSeconds: number,
): string {
  const hours = Math.floor(
    totalSeconds / 3600,
  );

  const minutes = Math.floor(
    (totalSeconds % 3600) / 60,
  );

  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(
      2,
      "0",
    )}m`;
  }

  if (minutes > 0) {
    return `${minutes}m ${String(seconds).padStart(
      2,
      "0",
    )}s`;
  }

  return `${seconds}s`;
}

function formatLocalDate(
  value: string,
): string {
  const ts = parseUTC(value);

  if (Number.isNaN(ts)) {
    return "—";
  }

  return new Date(ts).toLocaleString(
    undefined,
    {
      dateStyle: "medium",
      timeStyle: "short",
    },
  );
}