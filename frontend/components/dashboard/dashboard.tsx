"use client";

import {
  ArrowRight,
  CheckCircle2,
  Database,
  FileSearch,
  Plus,
  Sparkles,
  Upload,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import {
  DatasetUpload,
} from "@/components/dashboard/dataset-upload";

const recentDatasets = [
  {
    name: "customer_churn.csv",
    rows: "8,231",
    columns: "24",
    status: "Analyzed",
  },
  {
    name: "sales_forecast.csv",
    rows: "12,450",
    columns: "18",
    status: "Analyzed",
  },
  {
    name: "music_features.csv",
    rows: "4,102",
    columns: "31",
    status: "Ready",
  },
];

const workflowSteps = [
  {
    number: "01",
    title: "Profile",
    description: "Understand structure, types, and statistics.",
  },
  {
    number: "02",
    title: "Diagnose",
    description: "Find missing values, duplicates, outliers, and risks.",
  },
  {
    number: "03",
    title: "Model",
    description: "Train, compare, optimize, and evaluate models.",
  },
  {
    number: "04",
    title: "Explain",
    description: "Understand which features drive predictions.",
  },
];

export function Dashboard() {
  const router = useRouter();
  const [uploadOpen, setUploadOpen] = useState(false);

  function openUpload() {
    setUploadOpen(true);
  }

  function closeUpload() {
    setUploadOpen(false);
  }

  function handleUploadSuccess(dataset: {
    dataset_id: number;
  }) {
    setUploadOpen(false);

    router.push(
      `/datasets/${dataset.dataset_id}`,
    );
  }

  return (
    <>
      <div className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
        <section className="relative overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface)]">
          <div className="subtle-grid absolute inset-0 opacity-40" />

          <div className="relative px-6 py-10 sm:px-10 sm:py-12 lg:px-14 lg:py-14">
            <div className="max-w-2xl">
              <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--surface-elevated)] px-3 py-1.5 text-xs font-medium text-[var(--muted)]">
                <Sparkles size={13} />
                Automated data science workspace
              </div>

              <h1 className="max-w-xl text-3xl font-semibold tracking-[-0.035em] text-[var(--foreground)] sm:text-4xl lg:text-[44px] lg:leading-[1.08]">
                Turn your dataset into a clear, actionable
                analysis.
              </h1>

              <p className="mt-4 max-w-xl text-sm leading-6 text-[var(--muted)] sm:text-base">
                Upload a CSV and let the platform profile your
                data, diagnose quality issues, explore patterns,
                build models, and explain the results.
              </p>

              <div className="mt-7 flex flex-col gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={openUpload}
                  className="focus-ring inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-5 text-sm font-medium text-[var(--accent-foreground)] transition-opacity hover:opacity-90"
                >
                  <Upload size={16} />
                  Upload dataset
                </button>

                <button
                  type="button"
                  className="focus-ring inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] px-5 text-sm font-medium text-[var(--foreground)] transition-colors hover:bg-[var(--surface-muted)]"
                >
                  Explore a sample
                  <ArrowRight size={15} />
                </button>
              </div>
            </div>

            <div className="relative mt-10 grid max-w-3xl grid-cols-2 gap-3 border-t border-[var(--border)] pt-6 sm:grid-cols-4">
              <Stat label="Data profiling" />
              <Stat label="Quality analysis" />
              <Stat label="Automated ML" />
              <Stat label="SHAP insights" />
            </div>
          </div>
        </section>

        <section className="mt-10">
          <div className="mb-4 flex items-end justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold tracking-[-0.015em]">
                Recent datasets
              </h2>

              <p className="mt-1 text-sm text-[var(--muted)]">
                Continue working with a previous dataset.
              </p>
            </div>

            <button
              type="button"
              className="focus-ring hidden items-center gap-1.5 text-xs font-medium text-[var(--muted)] transition-colors hover:text-[var(--foreground)] sm:flex"
            >
              View all
              <ArrowRight size={14} />
            </button>
          </div>

          <div className="grid gap-3 lg:grid-cols-3">
            {recentDatasets.map((dataset) => (
              <DatasetCard
                key={dataset.name}
                name={dataset.name}
                rows={dataset.rows}
                columns={dataset.columns}
                status={dataset.status}
              />
            ))}

            <button
              type="button"
              onClick={openUpload}
              className="focus-ring group flex min-h-[142px] flex-col items-center justify-center rounded-xl border border-dashed border-[var(--border-strong)] bg-transparent p-5 text-center transition-colors hover:border-[var(--muted)] hover:bg-[var(--surface)]"
            >
              <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface)] text-[var(--muted)] transition-colors group-hover:text-[var(--foreground)]">
                <Plus size={17} />
              </div>

              <span className="text-sm font-medium">
                Add a dataset
              </span>

              <span className="mt-1 text-xs text-[var(--muted)]">
                Start a new analysis
              </span>
            </button>
          </div>
        </section>

        <section className="mt-10 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <div className="surface rounded-xl p-6">
            <div className="mb-6 flex items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[var(--surface-muted)]">
                <FileSearch size={17} />
              </div>

              <div>
                <h2 className="text-sm font-semibold">
                  From dataset to decision
                </h2>

                <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
                  Every analysis follows a clear path so you always
                  know what the platform is doing and why.
                </p>
              </div>
            </div>

            <div className="grid gap-0 sm:grid-cols-2">
              {workflowSteps.map((step, index) => (
                <div
                  key={step.number}
                  className={`border-[var(--border)] py-5 sm:px-4 ${
                    index >= 2 ? "border-t" : ""
                  } ${index % 2 === 1 ? "sm:border-l" : ""}`}
                >
                  <div className="mb-3 text-[11px] font-semibold tracking-[0.08em] text-[var(--muted-foreground)]">
                    {step.number}
                  </div>

                  <h3 className="text-sm font-semibold">
                    {step.title}
                  </h3>

                  <p className="mt-1.5 max-w-xs text-xs leading-5 text-[var(--muted)]">
                    {step.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="surface rounded-xl p-6">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--success-surface)] text-[var(--success)]">
                <CheckCircle2 size={17} />
              </div>

              <div>
                <h2 className="text-sm font-semibold">
                  Workspace ready
                </h2>

                <p className="mt-1 text-xs text-[var(--muted)]">
                  Everything you need to begin analyzing data.
                </p>
              </div>
            </div>

            <div className="mt-6 space-y-3">
              <Capability
                icon={Database}
                title="Upload CSV"
                description="Bring your own dataset."
              />

              <Capability
                icon={FileSearch}
                title="Automated analysis"
                description="Profile, diagnose, and explore."
              />

              <Capability
                icon={Sparkles}
                title="Intelligent insights"
                description="Understand what matters."
              />

              <Capability
                icon={CheckCircle2}
                title="Model & explain"
                description="Evaluate and interpret predictions."
              />
            </div>
          </div>
        </section>
      </div>

      {uploadOpen && (
        <DatasetUpload
          onClose={closeUpload}
          onSuccess={handleUploadSuccess}
        />
      )}
    </>
  );
}

function Stat({ label }: { label: string }) {
  return (
    <div>
      <div className="mb-1 h-1 w-5 rounded-full bg-[var(--accent)]" />
      <span className="text-xs text-[var(--muted)]">{label}</span>
    </div>
  );
}

function DatasetCard({
  name,
  rows,
  columns,
  status,
}: {
  name: string;
  rows: string;
  columns: string;
  status: string;
}) {
  return (
    <button
      type="button"
      className="focus-ring surface group min-h-[142px] rounded-xl p-5 text-left transition-all hover:-translate-y-0.5 hover:border-[var(--border-strong)] hover:shadow-sm"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--surface-muted)] text-[var(--muted)]">
          <Database size={17} />
        </div>

        <span className="rounded-full bg-[var(--surface-muted)] px-2 py-1 text-[10px] font-medium text-[var(--muted)]">
          {status}
        </span>
      </div>

      <div className="mt-4">
        <h3 className="truncate text-sm font-medium">{name}</h3>

        <div className="mt-2 flex gap-4 text-xs text-[var(--muted)]">
          <span>{rows} rows</span>
          <span>{columns} columns</span>
        </div>
      </div>
    </button>
  );
}

function Capability({
  icon: Icon,
  title,
  description,
}: {
  icon: typeof Database;
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-transparent p-2.5 transition-colors hover:border-[var(--border)] hover:bg-[var(--surface-muted)]">
      <Icon
        size={16}
        strokeWidth={1.8}
        className="shrink-0 text-[var(--muted)]"
      />

      <div className="min-w-0">
        <div className="text-xs font-medium">{title}</div>

        <div className="mt-0.5 truncate text-[11px] text-[var(--muted)]">
          {description}
        </div>
      </div>
    </div>
  );
}