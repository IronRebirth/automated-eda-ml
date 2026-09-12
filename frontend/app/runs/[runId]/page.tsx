import { notFound } from "next/navigation";
import { RunStatus } from "@/components/runs/run-status";
import {
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Database,
  Gauge,
  Lightbulb,
  Target,
  TrendingUp,
} from "lucide-react";

import { apiRequest } from "@/lib/api";

interface RunResult {
  task_type: string;
  target_column: string;
  evaluation: Record<string, Record<string, unknown>>;
  cross_validation: Record<
    string,
    Record<string, unknown>
  >;
  leaderboard: LeaderboardRow[];
  optimized_evaluation: Record<string, Record<string, unknown>>;
  best_model: {
    model_name: string;
    metrics: Record<string, unknown>;
  };
  explainability: {
    summary: Record<string, unknown>;
    insights: unknown[];
    metadata: Record<string, unknown>;
  };
  artifact_path: string | null;
}

interface ExplainabilityFeature {
  rank: number;
  feature: string;
  importance: number;
  relative_importance: number;
  impact: string;
}

interface LeaderboardRow {
  model: string;
  score: number;
  metrics: Record<string, unknown>;
  cv_score?: number;
  cv_std?: number;
  cv_metrics?: Record<string, unknown>;
  rank: number;
}

interface Run {
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
  result: RunResult | null;
}

interface Dataset {
  dataset_id: number;
  filename: string;
  rows: number;
  columns: number;
  created_at: string;
}

interface RunPageProps {
  params: Promise<{
    runId: string;
  }>;
}

export default async function RunPage({
  params,
}: RunPageProps) {
  const { runId } = await params;

  let run: Run;

  try {
    run = await apiRequest<Run>(
      `/runs/${runId}`,
      {
        cache: "no-store",
      },
    );
  } catch (error) {
    if (
      error instanceof Error &&
      error.message.toLowerCase().includes("not found")
    ) {
      notFound();
    }

    return <RunLoadError />;
  }

  let dataset: Dataset | null = null;

  try {
    dataset = await apiRequest<Dataset>(
      `/datasets/${run.dataset_id}`,
      {
        cache: "no-store",
      },
    );
  } catch {
    dataset = null;
  }

  if (run.status === "failed") {
    return (
      <RunFailedState
        run={run}
        dataset={dataset}
      />
    );
  }

  if (!run.result) {
    return (
      <RunStatus
        run={run}
        datasetName={dataset?.filename}
      />
    );
  }

  return (
    <RunResults
      run={run}
      dataset={dataset}
      result={run.result}
    />
  );
}

function RunResults({
  run,
  dataset,
  result,
}: {
  run: Run;
  dataset: Dataset | null;
  result: RunResult;
}) {
  const evaluationEntries = Object.entries(
    result.evaluation ?? {},
  );

  const bestModelEvaluation =
    result.evaluation?.[result.best_model.model_name] ??
    result.best_model.metrics;

  const optimizedEntries = Object.entries(
    result.optimized_evaluation ?? {},
  );

  return (
    <div className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
      <section>
        <div className="flex flex-col gap-6 border-b border-[var(--border)] pb-7 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[var(--surface-muted)] text-[var(--muted)]">
              <BrainCircuit
                size={20}
                strokeWidth={1.8}
              />
            </div>

            <div className="min-w-0">
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Machine learning results
              </div>

              <h1 className="mt-1 truncate text-2xl font-semibold tracking-[-0.025em] sm:text-3xl">
                {dataset?.filename ?? `Run #${run.run_id}`}
              </h1>

              <p className="mt-2 text-sm text-[var(--muted)]">
                Run #{run.run_id} · Dataset #{run.dataset_id}
                {" · "}
                {formatDate(run.created_at)}
              </p>
            </div>
          </div>

          <div className="inline-flex shrink-0 items-center gap-2 self-start rounded-full bg-[var(--success-surface)] px-3 py-1.5 text-xs font-medium text-[var(--success)]">
            <CheckCircle2 size={13} />
            Analysis completed
          </div>
        </div>
      </section>

      <section className="mt-7">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            icon={Target}
            label="Task"
            value={formatLabel(result.task_type)}
            description="Detected prediction problem"
          />

          <MetricCard
            icon={Database}
            label="Target"
            value={result.target_column}
            description="Prediction target"
          />

          <MetricCard
            icon={BrainCircuit}
            label="Best model"
            value={formatModelName(
              result.best_model.model_name,
            )}
            description="Top-performing model"
          />

          <MetricCard
            icon={Gauge}
            label="Test size"
            value={`${formatPercentage(run.test_size)}`}
            description="Holdout evaluation split"
          />
        </div>
      </section>

      <section className="mt-7">
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1.5fr)_minmax(320px,0.8fr)]">
          <section className="surface overflow-hidden rounded-xl">
            <SectionHeader
              icon={BrainCircuit}
              title="Best model"
              description="The model selected by the automated evaluation pipeline."
            />

            <div className="p-5 sm:p-6">
              <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Selected model
                  </div>

                  <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em]">
                    {formatModelName(
                      result.best_model.model_name,
                    )}
                  </h2>

                  <p className="mt-1.5 text-xs text-[var(--muted)]">
                    Ranked highest using the pipeline&apos;s
                    evaluation criterion.
                  </p>
                </div>

                <div className="rounded-lg bg-[var(--surface-muted)] px-4 py-3 sm:min-w-[150px] sm:text-right">
                  <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Primary score
                  </div>

                  <div className="mt-1 text-2xl font-semibold tabular-nums">
                    {formatPrimaryScore(
                      result.best_model.metrics,
                      result.task_type,
                    )}
                  </div>
                </div>
              </div>

              <div className="mt-6 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                {Object.entries(bestModelEvaluation).map(
                  ([metric, value]) => (
                    <MetricValue
                      key={metric}
                      label={metric}
                      value={value}
                    />
                  ),
                )}
              </div>
            </div>
          </section>

          <section className="surface overflow-hidden rounded-xl">
            <SectionHeader
              icon={Target}
              title="Run configuration"
              description="Parameters used for this analysis."
            />

            <div className="divide-y divide-[var(--border)]">
              <InfoRow
                label="Target column"
                value={run.target_column ?? "Not specified"}
              />

              <InfoRow
                label="Task type"
                value={formatLabel(result.task_type)}
              />

              <InfoRow
                label="Test size"
                value={formatPercentage(run.test_size)}
              />

              <InfoRow
                label="Random state"
                value={String(run.random_state)}
              />

              <InfoRow
                label="Completed"
                value={
                  run.completed_at
                    ? formatDate(run.completed_at)
                    : "Not available"
                }
              />
            </div>
          </section>
        </div>
      </section>

      <section className="mt-7">
        <div className="surface overflow-hidden rounded-xl">
          <SectionHeader
            icon={BarChart3}
            title="Model leaderboard"
            description="All models trained and evaluated by the automated pipeline."
          />

          {result.leaderboard.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left">
                <thead>
                  <tr className="border-b border-[var(--border)] bg-[var(--surface-muted)]">
                    <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                      Rank
                    </th>
                    <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                      Model
                    </th>
                    <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                      Score
                    </th>
                    <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                      CV score
                    </th>
                    <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                      CV std
                    </th>
                    <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                      Evaluation
                    </th>
                  </tr>
                </thead>

                <tbody className="divide-y divide-[var(--border)]">
                  {result.leaderboard.map((row) => {
                    const isBest =
                      row.model ===
                      result.best_model.model_name;

                    return (
                      <tr
                        key={row.model}
                        className={
                          isBest
                            ? "bg-[var(--surface-muted)]"
                            : undefined
                        }
                      >
                        <td className="px-5 py-4">
                          <span
                            className={`inline-flex h-7 min-w-7 items-center justify-center rounded-md px-2 text-xs font-semibold ${
                              isBest
                                ? "bg-[var(--accent)] text-[var(--accent-foreground)]"
                                : "bg-[var(--surface-muted)] text-[var(--muted)]"
                            }`}
                          >
                            {row.rank}
                          </span>
                        </td>

                        <td className="px-5 py-4">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">
                              {formatModelName(row.model)}
                            </span>

                            {isBest && (
                              <span className="rounded-full bg-[var(--success-surface)] px-2 py-1 text-[9px] font-semibold text-[var(--success)]">
                                Best
                              </span>
                            )}
                          </div>
                        </td>

                        <td className="px-5 py-4 text-sm font-semibold tabular-nums">
                          {formatNumberValue(row.score)}
                        </td>

                        <td className="px-5 py-4 text-sm text-[var(--muted)] tabular-nums">
                          {formatOptionalNumber(row.cv_score)}
                        </td>

                        <td className="px-5 py-4 text-sm text-[var(--muted)] tabular-nums">
                          {formatOptionalNumber(row.cv_std)}
                        </td>

                        <td className="px-5 py-4">
                          <div className="flex flex-wrap gap-x-4 gap-y-1">
                            {Object.entries(row.metrics).map(
                              ([metric, value]) => (
                                <span
                                  key={metric}
                                  className="text-xs text-[var(--muted)]"
                                >
                                  <span className="font-medium text-[var(--foreground)]">
                                    {formatLabel(metric)}
                                  </span>{" "}
                                  {formatNumberValue(value)}
                                </span>
                              ),
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState message="No leaderboard results were returned for this run." />
          )}
        </div>
      </section>

      <section className="mt-7">
        <div className="grid gap-3 lg:grid-cols-2">
          <section className="surface overflow-hidden rounded-xl">
            <SectionHeader
              icon={BarChart3}
              title="Evaluation metrics"
              description="Holdout performance for each trained model."
            />

            {evaluationEntries.length > 0 ? (
              <div className="divide-y divide-[var(--border)]">
                {evaluationEntries.map(
                  ([modelName, metrics]) => (
                    <div
                      key={modelName}
                      className="px-5 py-4 sm:px-6"
                    >
                      <div className="flex items-center justify-between gap-4">
                        <div className="text-sm font-medium">
                          {formatModelName(modelName)}
                        </div>

                        {modelName ===
                          result.best_model.model_name && (
                          <span className="rounded-full bg-[var(--success-surface)] px-2 py-1 text-[9px] font-semibold text-[var(--success)]">
                            Best
                          </span>
                        )}
                      </div>

                      <div className="mt-3 grid gap-2 sm:grid-cols-2">
                        {Object.entries(metrics).map(
                          ([metric, value]) => (
                            <MetricValue
                              key={metric}
                              label={metric}
                              value={value}
                            />
                          ),
                        )}
                      </div>
                    </div>
                  ),
                )}
              </div>
            ) : (
              <EmptyState message="No evaluation metrics were returned." />
            )}
          </section>

          <section className="surface overflow-hidden rounded-xl">
            <SectionHeader
              icon={Gauge}
              title="Cross-validation"
              description="Validation performance used to assess model stability."
            />

            <div className="p-5 sm:p-6">
              {Object.keys(result.cross_validation ?? {})
                .length > 0 ? (
                <div className="grid gap-3 sm:grid-cols-2">
                  {Object.entries(result.cross_validation).map(
                    ([modelName, metrics]) => (
                      <CrossValidationCard
                        key={modelName}
                        modelName={modelName}
                        metrics={metrics}
                      />
                    ),
                  )}
                </div>
              ) : (
                <EmptyState message="No cross-validation details were returned." />
              )}
            </div>
          </section>
        </div>
      </section>

      <section className="mt-7">
        <div className="surface overflow-hidden rounded-xl">
          <SectionHeader
            icon={TrendingUp}
            title="Optimization"
            description="Performance after automated model optimization."
          />

          {optimizedEntries.length > 0 ? (
            <div className="grid gap-3 p-5 sm:grid-cols-2 sm:p-6 lg:grid-cols-3">
              {optimizedEntries.map(
                ([modelName, metrics]) => (
                  <div
                    key={modelName}
                    className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-4"
                  >
                    <div className="text-sm font-medium">
                      {formatModelName(modelName)}
                    </div>

                    <div className="mt-3 grid gap-2">
                      {Object.entries(metrics).map(
                        ([metric, value]) => (
                          <div
                            key={metric}
                            className="flex items-center justify-between gap-3 text-xs"
                          >
                            <span className="text-[var(--muted)]">
                              {formatLabel(metric)}
                            </span>

                            <span className="font-semibold tabular-nums">
                              {formatNumberValue(value)}
                            </span>
                          </div>
                        ),
                      )}
                    </div>
                  </div>
                ),
              )}
            </div>
          ) : (
            <EmptyState message="No optimized evaluation results were returned." />
          )}
        </div>
      </section>

      <section className="mt-7">
        <div className="surface overflow-hidden rounded-xl">
          <SectionHeader
            icon={Lightbulb}
            title="Explainability"
            description="What the model behavior suggests about the dataset."
          />

          <div className="grid gap-6 p-5 sm:p-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(260px,0.6fr)]">
            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Key insights
              </div>

              {result.explainability.insights.length >
              0 ? (
                <div className="mt-3 space-y-2">
                  {result.explainability.insights.map(
                    (insight, index) => (
                      <InsightCard
                        key={index}
                        insight={insight}
                      />
                    ),
                  )}
                </div>
              ) : (
                <div className="mt-3">
                  <EmptyState message="No explainability insights were returned." />
                </div>
              )}
            </div>

            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Explainability summary
              </div>

              <div className="mt-3 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-4">
                {Object.keys(
                  result.explainability.summary ?? {},
                ).length > 0 ? (
                  <ExplainabilitySummary
                    summary={result.explainability.summary}
                  />
                ) : (
                  <p className="text-xs text-[var(--muted)]">
                    No summary details were returned.
                  </p>
                )}
              </div>

              {Object.keys(
                result.explainability.metadata ?? {},
              ).length > 0 && (
                <div className="mt-4">
                  <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Metadata
                  </div>

                  <div className="mt-2 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-4">
                    {renderMetricObject(
                      result.explainability.metadata,
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="mt-7">
        <div className="flex flex-col gap-3 rounded-xl border border-[var(--border)] bg-[var(--surface-muted)] p-4 sm:flex-row sm:items-center sm:justify-between sm:px-5">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--surface)] text-[var(--muted)]">
              <Clock3 size={16} />
            </div>

            <div>
              <div className="text-xs font-medium">
                Run completed
              </div>

              <div className="mt-0.5 text-[11px] text-[var(--muted)]">
                {run.completed_at
                  ? formatDate(run.completed_at)
                  : "Completion time unavailable"}
              </div>
            </div>
          </div>

          <div className="text-[11px] text-[var(--muted)]">
            Random state {run.random_state}
          </div>
        </div>
      </section>
    </div>
  );
}

function RunFailedState({
  run,
  dataset,
}: {
  run: Run;
  dataset: Dataset | null;
}) {
  return (
    <div className="mx-auto max-w-[900px] px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
      <div className="surface rounded-2xl p-6 sm:p-8">
        <div className="flex items-start gap-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--danger-surface)] text-[var(--danger)]">
            <AlertTriangle size={19} />
          </div>

          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
              Machine learning run
            </div>

            <h1 className="mt-1 text-2xl font-semibold tracking-[-0.025em]">
              Analysis failed
            </h1>

            <p className="mt-2 max-w-xl text-sm leading-6 text-[var(--muted)]">
              Run #{run.run_id}
              {dataset ? ` for ${dataset.filename}` : ""}
              {" "}
              did not complete successfully. No model results
              are available for this run.
            </p>

            <div className="mt-5 inline-flex items-center rounded-full bg-[var(--danger-surface)] px-3 py-1.5 text-xs font-medium text-[var(--danger)]">
              Status: {formatLabel(run.status)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function RunLoadError() {
  return (
    <div className="mx-auto max-w-[900px] px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
      <div className="surface rounded-2xl p-6 sm:p-8">
        <div className="flex items-start gap-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--danger-surface)] text-[var(--danger)]">
            <AlertTriangle size={19} />
          </div>

          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
              Machine learning results
            </div>

            <h1 className="mt-1 text-2xl font-semibold tracking-[-0.025em]">
              Unable to load run
            </h1>

            <p className="mt-2 max-w-xl text-sm leading-6 text-[var(--muted)]">
              We could not retrieve this analysis run right now.
              Make sure the backend is running and try again.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  description,
}: {
  icon: typeof Target;
  label: string;
  value: string;
  description: string;
}) {
  return (
    <div className="surface rounded-xl p-4 sm:p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--surface-muted)] text-[var(--muted)]">
          <Icon size={16} strokeWidth={1.8} />
        </div>

        <span className="text-[10px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
          {label}
        </span>
      </div>

      <div className="mt-4 truncate text-lg font-semibold tracking-[-0.015em]">
        {value}
      </div>

      <div className="mt-1 text-[11px] leading-4 text-[var(--muted)]">
        {description}
      </div>
    </div>
  );
}

function SectionHeader({
  icon: Icon,
  title,
  description,
}: {
  icon: typeof Target;
  title: string;
  description: string;
}) {
  return (
    <div className="border-b border-[var(--border)] px-5 py-5 sm:px-6">
      <div className="flex items-start gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--surface-muted)] text-[var(--muted)]">
          <Icon size={16} strokeWidth={1.8} />
        </div>

        <div>
          <h2 className="text-sm font-semibold">
            {title}
          </h2>

          <p className="mt-1 max-w-2xl text-xs leading-5 text-[var(--muted)]">
            {description}
          </p>
        </div>
      </div>
    </div>
  );
}

function CrossValidationCard({
  modelName,
  metrics,
}: {
  modelName: string;
  metrics: Record<string, unknown>;
}) {
  const metricEntries = buildCrossValidationMetrics(metrics);

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-medium">
            {formatModelName(modelName)}
          </div>

          <div className="mt-1 text-[11px] text-[var(--muted)]">
            Mean ± standard deviation across validation folds
          </div>
        </div>

        <div className="shrink-0 rounded-full bg-[var(--surface)] px-2 py-1 text-[9px] font-medium text-[var(--muted)]">
          CV
        </div>
      </div>

      {metricEntries.length > 0 ? (
        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          {metricEntries.map(
            ({ metric, value, std }) => (
              <div
                key={metric}
                className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2.5"
              >
                <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                  {formatMetricBaseLabel(metric)}
                </div>

                <div className="mt-1 text-sm font-semibold tabular-nums">
                  {formatNumberValue(value)}
                  {std !== undefined && (
                    <span className="font-normal text-[var(--muted)]">
                      {" ± "}
                      {formatNumberValue(std)}
                    </span>
                  )}
                </div>
              </div>
            ),
          )}
        </div>
      ) : (
        <div className="mt-4 text-xs text-[var(--muted)]">
          No numeric validation metrics were returned.
        </div>
      )}
    </div>
  );
}

function ExplainabilitySummary({
  summary,
}: {
  summary: Record<string, unknown>;
}) {
  const topN = summary.top_n;
  const features = parseExplainabilityFeatures(
    summary.features,
  );

  const otherEntries = Object.entries(summary).filter(
    ([key]) => key !== "top_n" && key !== "features",
  );

  return (
    <div className="space-y-5">
      {topN !== undefined && (
        <div className="flex items-center justify-between gap-4 border-b border-[var(--border)] pb-3">
          <span className="text-xs text-[var(--muted)]">
            Top features
          </span>

          <span className="text-sm font-semibold tabular-nums">
            {formatIntegerValue(topN)}
          </span>
        </div>
      )}

      {features.length > 0 && (
        <div>
          <div className="mb-3 text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
            Feature importance
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[620px] text-left">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="pb-2 pr-3 text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                    Rank
                  </th>

                  <th className="pb-2 pr-3 text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                    Feature
                  </th>

                  <th className="pb-2 pr-3 text-right text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                    Importance
                  </th>

                  <th className="pb-2 pr-3 text-right text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                    Relative
                  </th>

                  <th className="pb-2 text-right text-[10px] font-semibold uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
                    Impact
                  </th>
                </tr>
              </thead>

              <tbody className="divide-y divide-[var(--border)]">
                {features.map((item) => (
                  <tr
                    key={`${item.rank}-${item.feature}`}
                  >
                    <td className="py-3 pr-3">
                      <span className="inline-flex h-6 min-w-6 items-center justify-center rounded-md bg-[var(--surface)] px-1.5 text-[10px] font-semibold text-[var(--muted)]">
                        {item.rank}
                      </span>
                    </td>

                    <td className="max-w-[240px] py-3 pr-3">
                      <span
                        className="block truncate text-xs font-medium"
                        title={formatFeatureName(
                          item.feature,
                        )}
                      >
                        {formatFeatureName(item.feature)}
                      </span>
                    </td>

                    <td className="py-3 pr-3 text-right text-xs font-semibold tabular-nums">
                      {formatNumberValue(item.importance)}
                    </td>

                    <td className="py-3 pr-3 text-right text-xs text-[var(--muted)] tabular-nums">
                      {formatRelativeImportance(
                        item.relative_importance,
                      )}
                    </td>

                    <td className="py-3 text-right">
                      <ImpactBadge
                        impact={item.impact}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {otherEntries.length > 0 && (
        <div className="space-y-3 border-t border-[var(--border)] pt-4">
          {otherEntries.map(([key, value]) => (
            <div
              key={key}
              className="flex items-start justify-between gap-4 text-xs"
            >
              <span className="min-w-0 text-[var(--muted)]">
                {formatLabel(key)}
              </span>

              <span className="max-w-[65%] break-words text-right font-medium">
                {formatNumberValue(value)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ImpactBadge({
  impact,
}: {
  impact: string;
}) {
  const normalizedImpact = impact.toLowerCase();
  const isHigh = normalizedImpact === "high";
  const isMedium = normalizedImpact === "medium";

  const className = isHigh
    ? "bg-[var(--danger-surface)] text-[var(--danger)]"
    : isMedium
      ? "bg-[var(--surface)] text-[var(--foreground)]"
      : "bg-[var(--surface)] text-[var(--muted)]";

  return (
    <span
      className={`inline-flex rounded-full px-2 py-1 text-[9px] font-semibold capitalize ${className}`}
    >
      {impact || "Unknown"}
    </span>
  );
}

function parseExplainabilityFeatures(
  value: unknown,
): ExplainabilityFeature[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (
      typeof item !== "object" ||
      item === null
    ) {
      return [];
    }

    const record = item as Record<string, unknown>;

    const rank = Number(record.rank);

    const feature =
      typeof record.feature === "string"
        ? record.feature
        : "";

    const importance = Number(record.importance);

    const relativeImportance = Number(
      record.relative_importance,
    );

    const impact =
      typeof record.impact === "string"
        ? record.impact
        : "Unknown";

    if (
      !Number.isFinite(rank) ||
      !feature ||
      !Number.isFinite(importance) ||
      !Number.isFinite(relativeImportance)
    ) {
      return [];
    }

    return [
      {
        rank,
        feature,
        importance,
        relative_importance: relativeImportance,
        impact,
      },
    ];
  });
}

function buildCrossValidationMetrics(
  metrics: Record<string, unknown>,
): Array<{
  metric: string;
  value: unknown;
  std?: number;
}> {
  const metricMap = new Map<
    string,
    {
      metric: string;
      value: unknown;
      std?: number;
    }
  >();

  for (const [metric, value] of Object.entries(
    metrics,
  )) {
    if (metric.endsWith("_std")) {
      continue;
    }

    const stdValue = metrics[`${metric}_std`];

    const std =
      typeof stdValue === "number" &&
      Number.isFinite(stdValue)
        ? stdValue
        : undefined;

    metricMap.set(metric, {
      metric,
      value,
      std,
    });
  }

  for (const [metric, value] of Object.entries(
    metrics,
  )) {
    if (!metric.endsWith("_std")) {
      continue;
    }

    const baseMetric = metric.replace(
      /_std$/,
      "",
    );

    if (metricMap.has(baseMetric)) {
      continue;
    }

    const std =
      typeof value === "number" &&
      Number.isFinite(value)
        ? value
        : undefined;

    metricMap.set(baseMetric, {
      metric: baseMetric,
      value: "—",
      std,
    });
  }

  return Array.from(metricMap.values());
}

function formatMetricBaseLabel(metric: string): string {
  return formatLabel(
    metric
      .replace(/_std$/, "")
      .replace(/_mean$/, ""),
  );
}

function formatRelativeImportance(value: number): string {
  if (!Number.isFinite(value)) {
    return "—";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function formatFeatureName(value: string): string {
  return value
    .replace(/^(numerical|categorical|text)__/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) =>
      character.toUpperCase(),
    );
}

function MetricValue({
  label,
  value,
}: {
  label: string;
  value: unknown;
}) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-2.5">
      <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-[var(--muted-foreground)]">
        {formatLabel(label)}
      </div>

      <div className="mt-1 text-sm font-semibold tabular-nums">
        {formatNumberValue(value)}
      </div>
    </div>
  );
}

function InfoRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 px-5 py-3.5 sm:px-6">
      <span className="text-xs text-[var(--muted)]">
        {label}
      </span>

      <span className="max-w-[60%] truncate text-right text-xs font-medium">
        {value}
      </span>
    </div>
  );
}

function EmptyState({
  message,
}: {
  message: string;
}) {
  return (
    <div className="px-5 py-10 text-center sm:px-6">
      <div className="mx-auto flex h-9 w-9 items-center justify-center rounded-full bg-[var(--surface-muted)] text-[var(--muted)]">
        <BarChart3 size={16} />
      </div>

      <p className="mx-auto mt-3 max-w-md text-xs leading-5 text-[var(--muted)]">
        {message}
      </p>
    </div>
  );
}

function InsightCard({
  insight,
}: {
  insight: unknown;
}) {
  const text = formatInsight(insight);

  return (
    <div className="flex items-start gap-3 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-3.5">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[var(--surface)] text-[var(--muted)]">
        <Lightbulb size={14} />
      </div>

      <p className="text-xs leading-5 text-[var(--muted)]">
        {text}
      </p>
    </div>
  );
}

function renderMetricObject(
  value: Record<string, unknown>,
) {
  return Object.entries(value).map(
    ([key, itemValue]) => (
      <div
        key={key}
        className="flex items-start justify-between gap-4 text-xs"
      >
        <span className="min-w-0 text-[var(--muted)]">
          {formatLabel(key)}
        </span>

        <span className="max-w-[65%] break-words text-right font-medium">
          {formatNumberValue(itemValue)}
        </span>
      </div>
    ),
  );
}

function formatPrimaryScore(
  metrics: Record<string, unknown>,
  taskType: string,
): string {
  const preferredMetrics =
    taskType === "classification"
      ? ["f1", "accuracy", "precision", "recall"]
      : ["rmse", "r2", "mae", "mse"];

  for (const metric of preferredMetrics) {
    const value = metrics[metric];

    if (typeof value === "number") {
      return formatNumberValue(value);
    }
  }

  const firstNumeric = Object.values(metrics).find(
    (value) => typeof value === "number",
  );

  return formatNumberValue(firstNumeric);
}

function formatNumberValue(value: unknown): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "—";
    }

    return value.toFixed(4);
  }

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "object") {
    return "Structured data";
  }

  return String(value);
}

function formatIntegerValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isFinite(value)
      ? String(Math.round(value))
      : "—";
  }

  return value === null || value === undefined
    ? "—"
    : String(value);
}

function formatOptionalNumber(
  value: number | undefined,
): string {
  return value === undefined
    ? "—"
    : formatNumberValue(value);
}

function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function formatLabel(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) =>
      character.toUpperCase(),
    );
}

function formatModelName(value: string): string {
  return formatLabel(value);
}

function formatInsight(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  if (
    typeof value === "object" &&
    value !== null
  ) {
    const record = value as Record<string, unknown>;

    const preferredKeys = [
      "message",
      "insight",
      "text",
      "description",
      "finding",
    ];

    for (const key of preferredKeys) {
      if (typeof record[key] === "string") {
        return record[key] as string;
      }
    }

    try {
      return JSON.stringify(value);
    } catch {
      return "Explainability insight";
    }
  }

  return String(value);
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