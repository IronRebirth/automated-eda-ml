"use client";

import {
  CheckCircle2,
  FileText,
  Loader2,
  Upload,
  X,
} from "lucide-react";
import { useRef, useState } from "react";

import { apiRequest } from "@/lib/api";

interface DatasetUploadResponse {
  dataset_id: number;
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
}

interface DatasetUploadProps {
  onSuccess?: (dataset: DatasetUploadResponse) => void;
  onClose?: () => void;
}

const MAX_FILE_SIZE = 50 * 1024 * 1024;

export function DatasetUpload({
  onSuccess,
  onClose,
}: DatasetUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadedDataset, setUploadedDataset] =
    useState<DatasetUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function validateFile(selectedFile: File) {
    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      return "Only CSV files are supported.";
    }

    if (selectedFile.size === 0) {
      return "The selected file is empty.";
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      return "The maximum supported file size is 50 MB.";
    }

    return null;
  }

  function selectFile(selectedFile: File) {
    const validationError = validateFile(selectedFile);

    setError(validationError);

    if (validationError) {
      setFile(null);
      return;
    }

    setFile(selectedFile);
    setUploadedDataset(null);
  }

  function handleFileChange(
    event: React.ChangeEvent<HTMLInputElement>,
  ) {
    const selectedFile = event.target.files?.[0];

    if (selectedFile) {
      selectFile(selectedFile);
    }
  }

  function handleDragOver(
    event: React.DragEvent<HTMLDivElement>,
  ) {
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragLeave(
    event: React.DragEvent<HTMLDivElement>,
  ) {
    event.preventDefault();
    setDragActive(false);
  }

  function handleDrop(
    event: React.DragEvent<HTMLDivElement>,
  ) {
    event.preventDefault();
    setDragActive(false);

    const droppedFile = event.dataTransfer.files?.[0];

    if (droppedFile) {
      selectFile(droppedFile);
    }
  }

  function clearFile() {
    setFile(null);
    setError(null);
    setUploadedDataset(null);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  async function uploadFile() {
    if (!file) {
      setError("Select a CSV file before uploading.");
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const dataset = await apiRequest<DatasetUploadResponse>(
        "/datasets/upload",
        {
          method: "POST",
          body: formData,
        },
      );

      setUploadedDataset(dataset);
      onSuccess?.(dataset);
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Unable to upload the dataset.",
      );
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/30 p-4 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-upload-title"
        className="surface-elevated w-full max-w-lg rounded-2xl"
      >
        <div className="flex items-start justify-between border-b border-[var(--border)] px-6 py-5">
          <div>
            <h2
              id="dataset-upload-title"
              className="text-base font-semibold tracking-[-0.015em]"
            >
              Upload dataset
            </h2>

            <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
              Upload a CSV file to begin your analysis.
            </p>
          </div>

          {onClose && (
            <button
              type="button"
              aria-label="Close upload dialog"
              onClick={onClose}
              disabled={uploading}
              className="focus-ring flex h-8 w-8 items-center justify-center rounded-lg text-[var(--muted)] transition-colors hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              <X size={17} />
            </button>
          )}
        </div>

        <div className="p-6">
          {!uploadedDataset ? (
            <>
              <div
                role="button"
                tabIndex={0}
                aria-label="Choose a CSV file"
                onClick={() => inputRef.current?.click()}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" ||
                    event.key === " "
                  ) {
                    event.preventDefault();
                    inputRef.current?.click();
                  }
                }}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`focus-ring flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed px-6 text-center transition-colors ${
                  dragActive
                    ? "border-[var(--foreground)] bg-[var(--surface-muted)]"
                    : "border-[var(--border-strong)] hover:border-[var(--muted)] hover:bg-[var(--surface-muted)]"
                }`}
              >
                <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--surface-muted)] text-[var(--muted)]">
                  <Upload size={19} strokeWidth={1.8} />
                </div>

                <div className="text-sm font-medium">
                  Drop your CSV here
                </div>

                <div className="mt-1.5 text-xs text-[var(--muted)]">
                  or click to browse from your computer
                </div>

                <div className="mt-4 text-[11px] text-[var(--muted-foreground)]">
                  CSV files up to 50 MB
                </div>
              </div>

              <input
                ref={inputRef}
                type="file"
                accept=".csv,text/csv"
                onChange={handleFileChange}
                className="sr-only"
              />

              {file && (
                <div className="mt-4 flex items-center gap-3 rounded-xl border border-[var(--border)] bg-[var(--surface-muted)] p-3">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[var(--surface)] text-[var(--muted)]">
                    <FileText size={17} />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs font-medium">
                      {file.name}
                    </div>

                    <div className="mt-0.5 text-[11px] text-[var(--muted)]">
                      {formatFileSize(file.size)}
                    </div>
                  </div>

                  <button
                    type="button"
                    aria-label="Remove selected file"
                    onClick={clearFile}
                    disabled={uploading}
                    className="focus-ring flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[var(--muted)] transition-colors hover:bg-[var(--surface)] hover:text-[var(--foreground)] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <X size={15} />
                  </button>
                </div>
              )}

              {error && (
                <div
                  role="alert"
                  className="mt-4 rounded-lg border border-[var(--danger)]/20 bg-[var(--danger-surface)] px-3 py-2.5 text-xs leading-5 text-[var(--danger)]"
                >
                  {error}
                </div>
              )}

              <div className="mt-5 flex justify-end gap-2">
                {onClose && (
                  <button
                    type="button"
                    onClick={onClose}
                    disabled={uploading}
                    className="focus-ring h-10 rounded-lg border border-[var(--border)] px-4 text-xs font-medium text-[var(--foreground)] transition-colors hover:bg-[var(--surface-muted)] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Cancel
                  </button>
                )}

                <button
                  type="button"
                  onClick={uploadFile}
                  disabled={!file || uploading}
                  className="focus-ring inline-flex h-10 items-center gap-2 rounded-lg bg-[var(--accent)] px-4 text-xs font-medium text-[var(--accent-foreground)] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {uploading ? (
                    <>
                      <Loader2
                        size={14}
                        className="animate-spin"
                      />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload size={14} />
                      Upload dataset
                    </>
                  )}
                </button>
              </div>
            </>
          ) : (
            <div className="text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-[var(--success-surface)] text-[var(--success)]">
                <CheckCircle2 size={22} />
              </div>

              <h3 className="mt-4 text-sm font-semibold">
                Dataset uploaded successfully
              </h3>

              <p className="mx-auto mt-1.5 max-w-sm text-xs leading-5 text-[var(--muted)]">
                Your dataset has been saved and is ready for
                analysis.
              </p>

              <div className="mt-6 rounded-xl border border-[var(--border)] bg-[var(--surface-muted)] p-4 text-left">
                <div className="flex items-center gap-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--surface)] text-[var(--muted)]">
                    <FileText size={17} />
                  </div>

                  <div className="min-w-0">
                    <div className="truncate text-xs font-medium">
                      {uploadedDataset.filename}
                    </div>

                    <div className="mt-1 text-[11px] text-[var(--muted)]">
                      Dataset #{uploadedDataset.dataset_id}
                    </div>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-2 gap-3 border-t border-[var(--border)] pt-4">
                  <UploadStat
                    label="Rows"
                    value={formatNumber(uploadedDataset.rows)}
                  />

                  <UploadStat
                    label="Columns"
                    value={formatNumber(uploadedDataset.columns)}
                  />
                </div>
              </div>

              <div className="mt-5 flex justify-end">
                {onClose && (
                  <button
                    type="button"
                    onClick={onClose}
                    className="focus-ring h-10 rounded-lg bg-[var(--accent)] px-4 text-xs font-medium text-[var(--accent-foreground)] transition-opacity hover:opacity-90"
                  >
                    Done
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function UploadStat({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.08em] text-[var(--muted-foreground)]">
        {label}
      </div>

      <div className="mt-1 text-sm font-semibold">
        {value}
      </div>
    </div>
  );
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}