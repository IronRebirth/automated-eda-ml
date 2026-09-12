"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useState } from "react";
import {
  BarChart3,
  Lightbulb,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import type { Data, Layout } from "plotly.js";

import { apiRequest } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[360px] items-center justify-center">
      <div className="text-xs text-[var(--muted)]">
        Preparing visualization...
      </div>
    </div>
  ),
});

interface PlotlyFigure {
  data: Data[];
  layout?: Partial<Layout>;
}

interface EDAResult {
  numerical: unknown;
  categorical: unknown;
  correlations: unknown;
  target: unknown;
  insights: unknown[];
  visualizations: {
    numerical: PlotlyFigure[];
    categorical: PlotlyFigure[];
  };
}

interface EDAResponse {
  dataset_id: number;
  filename: string;
  target_column: string | null;
  eda: EDAResult;
}

interface EDASectionProps {
  datasetId: number;
}

export function EDASection({
  datasetId,
}: EDASectionProps) {
  const [result, setResult] =
    useState<EDAResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] =
    useState<string | null>(null);

  const loadEDA = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await apiRequest<EDAResponse>(
        `/datasets/${datasetId}/eda`,
        {
          cache: "no-store",
        },
      );

      setResult(data);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to load exploratory analysis.",
      );
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    let cancelled = false;

    async function fetchEDA() {
      try {
        const data = await apiRequest<EDAResponse>(
          `/datasets/${datasetId}/eda`,
          {
            cache: "no-store",
          },
        );

        if (cancelled) {
          return;
        }

        setResult(data);
        setError(null);
      } catch (requestError) {
        if (cancelled) {
          return;
        }

        setError(
          requestError instanceof Error
            ? requestError.message
            : "Unable to load exploratory analysis.",
        );
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void fetchEDA();

    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  return (
    <section className="mt-7">
      <div className="surface overflow-hidden rounded-xl">
        <div className="border-b border-[var(--border)] px-5 py-5 sm:px-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold">
                  Exploratory analysis
                </h2>

                {!loading && result && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--surface-muted)] px-2 py-1 text-[10px] font-medium text-[var(--muted)]">
                    <Sparkles size={11} />
                    Automated
                  </span>
                )}
              </div>

              <p className="mt-1 max-w-2xl text-xs leading-5 text-[var(--muted)]">
                Explore distributions, patterns, relationships,
                and automatically generated insights from your
                dataset.
              </p>
            </div>

            {!loading && (
              <button
                type="button"
                onClick={() => void loadEDA()}
                className="focus-ring inline-flex shrink-0 items-center justify-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs font-medium text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)]"
              >
                <RefreshCw size={13} />
                Refresh analysis
              </button>
            )}
          </div>
        </div>

        {loading && <EDALoadingState />}

        {!loading && error && (
          <EDAErrorState
            message={error}
            onRetry={() => void loadEDA()}
          />
        )}

        {!loading && !error && result && (
          <EDAContent result={result} />
        )}
      </div>
    </section>
  );
}

function EDAContent({
  result,
}: {
  result: EDAResponse;
}) {
  const insights = normalizeInsights(
    result.eda.insights,
  );

  const numericalFigures =
    result.eda.visualizations?.numerical ?? [];

  const categoricalFigures =
    result.eda.visualizations?.categorical ?? [];

  const totalVisualizations =
    numericalFigures.length +
    categoricalFigures.length;

  return (
    <div>
      <div className="grid border-b border-[var(--border)] sm:grid-cols-3">
        <EDASummaryItem
          label="Insights"
          value={insights.length}
          description="Automated findings"
        />

        <EDASummaryItem
          label="Numerical charts"
          value={numericalFigures.length}
          description="Distribution views"
        />

        <EDASummaryItem
          label="Categorical charts"
          value={categoricalFigures.length}
          description="Category views"
        />
      </div>

      {result.target_column && (
        <div className="border-b border-[var(--border)] bg-[var(--surface-muted)] px-5 py-3 sm:px-6">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-[var(--muted)]">
              Analysis target
            </span>

            <span className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-2 py-1 font-mono text-[10px] font-medium">
              {result.target_column}
            </span>
          </div>
        </div>
      )}

      <div className="px-5 py-6 sm:px-6">
        <div className="flex items-start gap-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--surface-muted)] text-[var(--muted)]">
            <Lightbulb size={16} />
          </div>

          <div>
            <h3 className="text-sm font-semibold">
              Key insights
            </h3>

            <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
              Automatically detected patterns and observations
              worth reviewing before modeling.
            </p>
          </div>
        </div>

        {insights.length > 0 ? (
          <div className="mt-5 grid gap-3 lg:grid-cols-2">
            {insights.map((insight, index) => (
              <InsightCard
                key={`${index}-${insight}`}
                insight={insight}
              />
            ))}
          </div>
        ) : (
          <div className="mt-5 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-4 py-6 text-center">
            <p className="text-xs text-[var(--muted)]">
              No automated insights were generated for this
              dataset.
            </p>
          </div>
        )}
      </div>

      {totalVisualizations > 0 ? (
        <div className="border-t border-[var(--border)] px-5 py-6 sm:px-6">
          <div className="flex items-start gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--surface-muted)] text-[var(--muted)]">
              <BarChart3 size={16} />
            </div>

            <div>
              <h3 className="text-sm font-semibold">
                Distributions
              </h3>

              <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
                Interactive visualizations generated automatically
                from the dataset.
              </p>
            </div>
          </div>

          {numericalFigures.length > 0 && (
            <div className="mt-5">
              <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Numerical distributions
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                {numericalFigures.map((figure, index) => (
                  <EDAPlot
                    key={`numerical-${index}`}
                    figure={figure}
                  />
                ))}
              </div>
            </div>
          )}

          {categoricalFigures.length > 0 && (
            <div className="mt-7">
              <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Categorical distributions
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                {categoricalFigures.map((figure, index) => (
                  <EDAPlot
                    key={`categorical-${index}`}
                    figure={figure}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="border-t border-[var(--border)] px-5 py-8 text-center sm:px-6">
          <p className="text-xs text-[var(--muted)]">
            No visualizations were generated for this dataset.
          </p>
        </div>
      )}
    </div>
  );
}

function EDAPlot({
  figure,
}: {
  figure: PlotlyFigure;
}) {
  return (
    <div className="overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
      <Plot
        data={figure.data}
        layout={{
          ...figure.layout,
          autosize: true,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: {
            ...(figure.layout?.font ?? {}),
            color: "var(--foreground)",
          },
          margin: {
            l: 50,
            r: 20,
            t: 55,
            b: 50,
            ...(figure.layout?.margin ?? {}),
          },
        }}
        config={{
          responsive: true,
          displayModeBar: false,
        }}
        useResizeHandler
        style={{
          width: "100%",
          height: "360px",
        }}
      />
    </div>
  );
}

function InsightCard({
  insight,
}: {
  insight: string;
}) {
  return (
    <article className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-4">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-[var(--surface)] text-[var(--muted)]">
          <Sparkles size={12} />
        </div>

        <p className="text-xs leading-5 text-[var(--foreground)]">
          {insight}
        </p>
      </div>
    </article>
  );
}

function EDASummaryItem({
  label,
  value,
  description,
}: {
  label: string;
  value: number;
  description: string;
}) {
  return (
    <div className="border-b border-[var(--border)] px-5 py-4 last:border-b-0 sm:border-b-0 sm:border-r sm:last:border-r-0 sm:px-6">
      <div className="text-[11px] font-medium text-[var(--muted)]">
        {label}
      </div>

      <div className="mt-1.5 text-xl font-semibold tracking-[-0.02em]">
        {value}
      </div>

      <div className="mt-1 text-[10px] text-[var(--muted-foreground)]">
        {description}
      </div>
    </div>
  );
}

function EDALoadingState() {
  return (
    <div className="px-5 py-12 sm:px-6">
      <div className="mx-auto max-w-md text-center">
        <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-[var(--surface-muted)]">
          <RefreshCw
            size={17}
            className="animate-spin text-[var(--muted)]"
          />
        </div>

        <h3 className="mt-4 text-sm font-semibold">
          Analyzing your dataset
        </h3>

        <p className="mt-1.5 text-xs leading-5 text-[var(--muted)]">
          Generating distributions, patterns, and automated
          insights. This may take a moment.
        </p>
      </div>
    </div>
  );
}

function EDAErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="px-5 py-10 sm:px-6">
      <div className="rounded-lg border border-[var(--danger)]/20 bg-[var(--danger-surface)] px-4 py-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold text-[var(--danger)]">
              Unable to load exploratory analysis
            </h3>

            <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
              {message}
            </p>
          </div>

          <button
            type="button"
            onClick={onRetry}
            className="focus-ring inline-flex shrink-0 items-center justify-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs font-medium text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)]"
          >
            <RefreshCw size={13} />
            Try again
          </button>
        </div>
      </div>
    </div>
  );
}

function normalizeInsights(
  insights: unknown[],
): string[] {
  return insights
    .map((insight) => {
      if (typeof insight === "string") {
        return insight;
      }

      if (
        typeof insight === "object" &&
        insight !== null &&
        "message" in insight &&
        typeof insight.message === "string"
      ) {
        return insight.message;
      }

      return String(insight);
    })
    .filter((insight) => insight.trim().length > 0);
}