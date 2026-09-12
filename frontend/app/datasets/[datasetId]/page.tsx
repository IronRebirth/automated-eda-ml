import { notFound } from "next/navigation";
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FileText,
  Rows3,
} from "lucide-react";

import { EDASection } from "@/components/datasets/eda-section";
import { apiRequest } from "@/lib/api";
import { MLRunLauncher } from "@/components/datasets/ml-run-launcher";

interface ColumnProfile {
  name: string;
  dtype: string;
  missing_values: number;
  missing_percentage: number;
  unique_values: number;
  sample_values: unknown[];
}

interface DatasetProfile {
  rows: number;
  columns: number;
  column_names: string[];
  numerical_columns: string[];
  categorical_columns: string[];
  datetime_columns: string[];
  missing_values: number;
  missing_percentage: number;
  duplicate_rows: number;
  constant_columns: string[];
  memory_usage_mb: number;
  column_profiles: ColumnProfile[];
}

interface Dataset {
  dataset_id: number;
  filename: string;
  rows: number;
  columns: number;
  created_at: string;
  profile: DatasetProfile;
}

interface DatasetPageProps {
  params: Promise<{
    datasetId: string;
  }>;
}

interface QualityFinding {
  title: string;
  severity: "high" | "medium" | "low";
  finding: string;
  whyItMatters: string;
  recommendation: string;
  details?: string;
}

export default async function DatasetPage({
  params,
}: DatasetPageProps) {
  const { datasetId } = await params;

  let dataset: Dataset;

  try {
    dataset = await apiRequest<Dataset>(
      `/datasets/${datasetId}`,
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

    return (
      <div className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
        <div className="surface rounded-2xl p-6 sm:p-8">
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--danger-surface)] text-[var(--danger)]">
              <AlertTriangle size={19} />
            </div>

            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Dataset
              </div>

              <h1 className="mt-1 text-2xl font-semibold tracking-[-0.025em]">
                Unable to load dataset
              </h1>

              <p className="mt-2 max-w-xl text-sm leading-6 text-[var(--muted)]">
                We could not retrieve this dataset right now.
                Make sure the backend is running and try again.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const profile = dataset.profile;
  const findings = buildQualityFindings(profile);

  const totalColumns =
    profile.numerical_columns.length +
    profile.categorical_columns.length +
    profile.datetime_columns.length;

  const hasQualityIssues = findings.length > 0;

  return (
    <div className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
      <section>
        <div className="flex flex-col gap-6 border-b border-[var(--border)] pb-7 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[var(--surface-muted)] text-[var(--muted)]">
              <Database size={19} strokeWidth={1.8} />
            </div>

            <div className="min-w-0">
              <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                Dataset overview
              </div>

              <h1 className="mt-1 truncate text-2xl font-semibold tracking-[-0.025em] sm:text-3xl">
                {dataset.filename}
              </h1>

              <p className="mt-2 text-sm text-[var(--muted)]">
                Dataset #{dataset.dataset_id} ·{" "}
                {formatDate(dataset.created_at)}
              </p>
            </div>
          </div>

          <div
            className={`inline-flex shrink-0 items-center gap-2 self-start rounded-full px-3 py-1.5 text-xs font-medium ${
              hasQualityIssues
                ? "bg-[var(--warning-surface)] text-[var(--warning)]"
                : "bg-[var(--success-surface)] text-[var(--success)]"
            }`}
          >
            {hasQualityIssues ? (
              <AlertTriangle size={13} />
            ) : (
              <CheckCircle2 size={13} />
            )}

            {hasQualityIssues
              ? `${findings.length} quality ${
                  findings.length === 1
                    ? "issue"
                    : "issues"
                } found`
              : "Dataset looks healthy"}
          </div>
        </div>
      </section>

      <section className="mt-7">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            icon={Rows3}
            label="Rows"
            value={formatNumber(dataset.rows)}
            description="Records in the dataset"
          />

          <MetricCard
            icon={FileText}
            label="Columns"
            value={formatNumber(dataset.columns)}
            description="Features and fields"
          />

          <MetricCard
            icon={AlertTriangle}
            label="Missing values"
            value={formatNumber(profile.missing_values)}
            description={`${profile.missing_percentage.toFixed(2)}% of all values`}
            warning={profile.missing_values > 0}
          />

          <MetricCard
            icon={Database}
            label="Duplicate rows"
            value={formatNumber(profile.duplicate_rows)}
            description={
              profile.duplicate_rows > 0
                ? "Rows that may need review"
                : "No duplicate records detected"
            }
            warning={profile.duplicate_rows > 0}
          />
        </div>
      </section>

      <section className="mt-7">
        <div className="surface overflow-hidden rounded-xl">
          <div className="border-b border-[var(--border)] px-5 py-5 sm:px-6">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-sm font-semibold">
                    Data quality
                  </h2>

                  {hasQualityIssues && (
                    <span className="rounded-full bg-[var(--warning-surface)] px-2 py-1 text-[10px] font-medium text-[var(--warning)]">
                      Review recommended
                    </span>
                  )}
                </div>

                <p className="mt-1 max-w-2xl text-xs leading-5 text-[var(--muted)]">
                  Findings are prioritized by impact so you can
                  focus on the issues most likely to affect analysis
                  or modeling.
                </p>
              </div>

              <div className="shrink-0 text-left sm:text-right">
                <div className="text-[11px] text-[var(--muted)]">
                  Findings
                </div>

                <div className="mt-0.5 text-sm font-semibold">
                  {formatNumber(findings.length)}
                </div>
              </div>
            </div>
          </div>

          {hasQualityIssues ? (
            <div className="divide-y divide-[var(--border)]">
              {findings.map((item) => (
                <QualityFindingCard
                  key={item.title}
                  finding={item}
                />
              ))}
            </div>
          ) : (
            <div className="px-5 py-10 text-center sm:px-6">
              <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-[var(--success-surface)] text-[var(--success)]">
                <CheckCircle2 size={18} />
              </div>

              <h3 className="mt-4 text-sm font-semibold">
                No major quality issues detected
              </h3>

              <p className="mx-auto mt-1.5 max-w-md text-xs leading-5 text-[var(--muted)]">
                The initial structural checks did not find missing
                values, duplicate rows, or constant columns.
              </p>
            </div>
          )}
        </div>
      </section>

      <section className="mt-7 grid gap-3 lg:grid-cols-3">
        <OverviewCard
          label="Numerical"
          value={profile.numerical_columns.length}
          description="Numeric features detected"
        />

        <OverviewCard
          label="Categorical"
          value={profile.categorical_columns.length}
          description="Categorical features detected"
        />

        <OverviewCard
          label="Datetime"
          value={profile.datetime_columns.length}
          description="Date or time features detected"
        />
      </section>

      <section className="mt-7">
        <div className="surface overflow-hidden rounded-xl">
          <div className="border-b border-[var(--border)] px-5 py-5 sm:px-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold">
                  Columns
                </h2>

                <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
                  Inspect the structure and quality of every column
                  in your dataset.
                </p>
              </div>

              <div className="shrink-0 text-right">
                <div className="text-[11px] text-[var(--muted)]">
                  Total
                </div>

                <div className="mt-0.5 text-sm font-semibold">
                  {formatNumber(profile.column_profiles.length)}
                </div>
              </div>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] text-left">
              <thead>
                <tr className="border-b border-[var(--border)] bg-[var(--surface-muted)]">
                  <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Column
                  </th>

                  <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Type
                  </th>

                  <th className="px-5 py-3 text-right text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Missing
                  </th>

                  <th className="px-5 py-3 text-right text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Missing %
                  </th>

                  <th className="px-5 py-3 text-right text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Unique
                  </th>

                  <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
                    Samples
                  </th>
                </tr>
              </thead>

              <tbody>
                {profile.column_profiles.map((column) => (
                  <ColumnRow
                    key={column.name}
                    column={column}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="mt-7">
        <div className="surface rounded-xl p-5 sm:p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold">
                Dataset health
              </h2>

              <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
                A quick summary of the structural checks performed
                on your dataset.
              </p>
            </div>

            <div className="hidden text-right sm:block">
              <div className="text-[11px] text-[var(--muted)]">
                Detected columns
              </div>

              <div className="mt-0.5 text-sm font-semibold">
                {formatNumber(totalColumns)}
              </div>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <HealthItem
              label="Missing data"
              value={
                profile.missing_values === 0
                  ? "None detected"
                  : `${formatNumber(profile.missing_values)} values`
              }
              healthy={profile.missing_values === 0}
            />

            <HealthItem
              label="Duplicate rows"
              value={
                profile.duplicate_rows === 0
                  ? "None detected"
                  : `${formatNumber(profile.duplicate_rows)} rows`
              }
              healthy={profile.duplicate_rows === 0}
            />

            <HealthItem
              label="Constant columns"
              value={
                profile.constant_columns.length === 0
                  ? "None detected"
                  : `${formatNumber(
                      profile.constant_columns.length,
                    )} columns`
              }
              healthy={profile.constant_columns.length === 0}
            />
          </div>
        </div>
      </section>

      <MLRunLauncher dataset={dataset} />
      
      <EDASection
        datasetId={dataset.dataset_id}
      />
    </div>
  );
}

function buildQualityFindings(
  profile: DatasetProfile,
): QualityFinding[] {
  const findings: QualityFinding[] = [];

  if (profile.missing_values > 0) {
    const affectedColumns = profile.column_profiles.filter(
      (column) => column.missing_values > 0,
    );

    const highestMissingColumn = affectedColumns.reduce(
      (highest, column) =>
        column.missing_percentage >
        highest.missing_percentage
          ? column
          : highest,
      affectedColumns[0],
    );

    findings.push({
      title: "Missing values detected",
      severity:
        profile.missing_percentage >= 20
          ? "high"
          : "medium",
      finding: `${formatNumber(
        profile.missing_values,
      )} values are missing across ${formatNumber(
        affectedColumns.length,
      )} column${
        affectedColumns.length === 1 ? "" : "s"
      }.`,
      whyItMatters:
        "Missing data can reduce the amount of usable information and may cause errors or biased results during modeling.",
      recommendation:
        "Review affected columns and choose an appropriate strategy such as imputation, removal, or retaining missingness as a meaningful signal.",
      details: highestMissingColumn
        ? `Highest affected column: ${highestMissingColumn.name} (${highestMissingColumn.missing_percentage.toFixed(2)}% missing).`
        : undefined,
    });
  }

  if (profile.duplicate_rows > 0) {
    findings.push({
      title: "Duplicate rows detected",
      severity:
        profile.duplicate_rows / Math.max(profile.rows, 1) >=
        0.05
          ? "high"
          : "medium",
      finding: `${formatNumber(
        profile.duplicate_rows,
      )} duplicate row${
        profile.duplicate_rows === 1 ? "" : "s"
      } were found.`,
      whyItMatters:
        "Duplicate records can overrepresent observations and distort descriptive statistics or model training.",
      recommendation:
        "Inspect the duplicated records and remove them when they represent accidental repeated observations.",
      details: `${(
        (profile.duplicate_rows /
          Math.max(profile.rows, 1)) *
        100
      ).toFixed(2)}% of all rows are duplicates.`,
    });
  }

  if (profile.constant_columns.length > 0) {
    findings.push({
      title: "Constant columns detected",
      severity: "low",
      finding: `${formatNumber(
        profile.constant_columns.length,
      )} column${
        profile.constant_columns.length === 1
          ? ""
          : "s"
      } contain only one unique value.`,
      whyItMatters:
        "A constant feature provides no variation and normally cannot help a predictive model distinguish between observations.",
      recommendation:
        "Consider removing constant columns before modeling unless they have a specific business or reporting purpose.",
      details: `Affected columns: ${profile.constant_columns.join(", ")}.`,
    });
  }

  return findings;
}

function QualityFindingCard({
  finding,
}: {
  finding: QualityFinding;
}) {
  return (
    <article className="px-5 py-6 sm:px-6">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold">
              {finding.title}
            </h3>

            <SeverityBadge severity={finding.severity} />
          </div>

          <p className="mt-2 text-xs leading-5 text-[var(--foreground)]">
            {finding.finding}
          </p>

          {finding.details && (
            <p className="mt-1.5 text-[11px] leading-5 text-[var(--muted)]">
              {finding.details}
            </p>
          )}
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:w-[520px] lg:shrink-0">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
              Why it matters
            </div>

            <p className="mt-1.5 text-[11px] leading-5 text-[var(--muted)]">
              {finding.whyItMatters}
            </p>
          </div>

          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
              Recommended action
            </div>

            <p className="mt-1.5 text-[11px] leading-5 text-[var(--muted)]">
              {finding.recommendation}
            </p>
          </div>
        </div>
      </div>
    </article>
  );
}

function SeverityBadge({
  severity,
}: {
  severity: QualityFinding["severity"];
}) {
  const label =
    severity === "high"
      ? "High impact"
      : severity === "medium"
        ? "Review"
        : "Low impact";

  return (
    <span
      className={`rounded-full px-2 py-1 text-[10px] font-medium ${
        severity === "high"
          ? "bg-[var(--danger-surface)] text-[var(--danger)]"
          : severity === "medium"
            ? "bg-[var(--warning-surface)] text-[var(--warning)]"
            : "bg-[var(--surface-muted)] text-[var(--muted)]"
      }`}
    >
      {label}
    </span>
  );
}

function ColumnRow({
  column,
}: {
  column: ColumnProfile;
}) {
  const hasMissing = column.missing_values > 0;

  return (
    <tr className="border-b border-[var(--border)] last:border-b-0">
      <td className="max-w-[220px] px-5 py-4">
        <div
          className="truncate text-xs font-medium"
          title={column.name}
        >
          {column.name}
        </div>
      </td>

      <td className="px-5 py-4">
        <span className="inline-flex rounded-md border border-[var(--border)] bg-[var(--surface-muted)] px-2 py-1 font-mono text-[10px] text-[var(--muted)]">
          {column.dtype}
        </span>
      </td>

      <td
        className={`px-5 py-4 text-right text-xs ${
          hasMissing
            ? "font-medium text-[var(--warning)]"
            : "text-[var(--muted)]"
        }`}
      >
        {formatNumber(column.missing_values)}
      </td>

      <td
        className={`px-5 py-4 text-right text-xs ${
          hasMissing
            ? "font-medium text-[var(--warning)]"
            : "text-[var(--muted)]"
        }`}
      >
        {column.missing_percentage.toFixed(2)}%
      </td>

      <td className="px-5 py-4 text-right text-xs text-[var(--muted)]">
        {formatNumber(column.unique_values)}
      </td>

      <td className="max-w-[280px] px-5 py-4">
        <div className="flex max-w-[280px] flex-wrap gap-1.5">
          {column.sample_values.length > 0 ? (
            column.sample_values.slice(0, 3).map((value, index) => (
              <span
                key={`${column.name}-${index}`}
                className="max-w-[180px] truncate rounded-md bg-[var(--surface-muted)] px-2 py-1 text-[10px] text-[var(--muted)]"
                title={formatSampleValue(value)}
              >
                {formatSampleValue(value)}
              </span>
            ))
          ) : (
            <span className="text-[10px] text-[var(--muted-foreground)]">
              No samples
            </span>
          )}
        </div>
      </td>
    </tr>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  description,
  warning = false,
}: {
  icon: typeof Database;
  label: string;
  value: string;
  description: string;
  warning?: boolean;
}) {
  return (
    <div className="surface rounded-xl p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs font-medium text-[var(--muted)]">
          {label}
        </div>

        <Icon
          size={16}
          strokeWidth={1.8}
          className={
            warning
              ? "text-[var(--warning)]"
              : "text-[var(--muted-foreground)]"
          }
        />
      </div>

      <div className="mt-4 text-2xl font-semibold tracking-[-0.025em]">
        {value}
      </div>

      <div className="mt-1 text-[11px] text-[var(--muted)]">
        {description}
      </div>
    </div>
  );
}

function OverviewCard({
  label,
  value,
  description,
}: {
  label: string;
  value: number;
  description: string;
}) {
  return (
    <div className="surface rounded-xl p-5">
      <div className="text-xs font-medium text-[var(--muted)]">
        {label}
      </div>

      <div className="mt-3 text-xl font-semibold tracking-[-0.02em]">
        {formatNumber(value)}
      </div>

      <div className="mt-1 text-[11px] text-[var(--muted)]">
        {description}
      </div>
    </div>
  );
}

function HealthItem({
  label,
  value,
  healthy,
}: {
  label: string;
  value: string;
  healthy: boolean;
}) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-4">
      {healthy ? (
        <CheckCircle2
          size={16}
          className="mt-0.5 shrink-0 text-[var(--success)]"
        />
      ) : (
        <AlertTriangle
          size={16}
          className="mt-0.5 shrink-0 text-[var(--warning)]"
        />
      )}

      <div className="min-w-0">
        <div className="text-xs font-medium">{label}</div>

        <div className="mt-1 text-[11px] text-[var(--muted)]">
          {value}
        </div>
      </div>
    </div>
  );
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
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

  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function formatSampleValue(value: unknown) {
  if (value === null || value === undefined) {
    return "null";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}