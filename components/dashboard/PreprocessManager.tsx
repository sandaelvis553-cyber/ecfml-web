"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CorrelationHeatmap } from "@/components/charts/CorrelationHeatmap";
import { SeasonalDecomposition } from "@/components/charts/SeasonalDecomposition";
import type { Dataset, WeatherDataset } from "@/types/dataset";
import type { EDACharts, JobStatus, PreprocessSummary } from "@/types/model";

interface PreprocessManagerProps {
  initialDatasets: Dataset[];
  initialWeatherDatasets: WeatherDataset[];
  initialError?: string | null;
}

type DatasetRecord = Dataset & {
  file_url?: string;
  uploadthing_url?: string;
  fileUrl?: string;
};

type WeatherDatasetRecord = WeatherDataset & {
  file_url?: string;
  uploadthing_url?: string;
  fileUrl?: string;
  dataset_id?: string;
};

interface PreprocessStatusResponse {
  job_id: string;
  status: JobStatus;
  progress?: number;
  error?: string;
  processed_file_path?: string;
  result_summary?: PreprocessSummary;
  eda_charts?: EDACharts;
}

type PreprocessSummaryRecord = PreprocessSummary & {
  row_count?: number;
  feature_count?: number;
  missing_values_before?: number;
  missing_values_after?: number;
  outlier_count?: number;
  outlier_percentage?: number;
};

type EDAChartsRecord = EDACharts & {
  correlation_matrix?: number[][];
  feature_names?: string[];
  distribution_data?: Record<string, number[]>;
  seasonal_decomposition?: {
    trend: number[];
    seasonal: number[];
    residual: number[];
    timestamps: string[];
  };
};

const STATUS_STYLES: Record<JobStatus, string> = {
  PENDING: "text-muted-foreground",
  RUNNING: "text-blue-500 dark:text-blue-400",
  COMPLETE: "text-emerald-500 dark:text-emerald-400",
  FAILED: "text-red-500 dark:text-red-400",
  WARNING: "text-amber-500 dark:text-amber-400",
};

/**
 * Resolve a file URL from either camelCase or snake_case API payloads.
 */
const resolveFileUrl = (record: DatasetRecord | WeatherDatasetRecord) => {
  return (
    record.uploadthingUrl ??
    record.uploadthing_url ??
    record.fileUrl ??
    record.file_url ??
    null
  );
};

/**
 * Resolve a dataset id from a weather dataset record.
 */
const resolveWeatherDatasetId = (record: WeatherDatasetRecord) => {
  return record.datasetId ?? record.dataset_id ?? null;
};

/**
 * Normalize preprocessing summary payloads from snake_case to camelCase.
 */
const normalizeSummary = (raw?: PreprocessSummaryRecord | null) => {
  if (!raw) return null;
  return {
    rowCount: raw.rowCount ?? raw.row_count ?? 0,
    featureCount: raw.featureCount ?? raw.feature_count ?? 0,
    missingValuesBefore:
      raw.missingValuesBefore ?? raw.missing_values_before ?? 0,
    missingValuesAfter: raw.missingValuesAfter ?? raw.missing_values_after ?? 0,
    outlierCount: raw.outlierCount ?? raw.outlier_count ?? 0,
    outlierPercentage: raw.outlierPercentage ?? raw.outlier_percentage ?? 0,
  };
};

/**
 * Normalize EDA chart payloads from snake_case to camelCase.
 */
const normalizeEdaCharts = (raw?: EDAChartsRecord | null) => {
  if (!raw) return null;
  const seasonal =
    raw.seasonalDecomposition ?? raw.seasonal_decomposition ?? null;
  return {
    correlationMatrix: raw.correlationMatrix ?? raw.correlation_matrix ?? [],
    featureNames: raw.featureNames ?? raw.feature_names ?? [],
    distributionData: raw.distributionData ?? raw.distribution_data ?? {},
    seasonalDecomposition: seasonal
      ? {
          trend: seasonal.trend ?? [],
          seasonal: seasonal.seasonal ?? [],
          residual: seasonal.residual ?? [],
          timestamps: seasonal.timestamps ?? [],
        }
      : null,
  };
};

/**
 * Create a stable job id for preprocessing requests.
 */
const createJobId = () => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
};

/**
 * Polls the preprocessing status endpoint until completion.
 */
function usePreprocessPoller(jobId: string | null, intervalMs = 3000) {
  const [state, setState] = useState<PreprocessStatusResponse | null>(null);

  const pollStatus = useCallback(async () => {
    if (!jobId) return;
    const res = await fetch(`/api/preprocess/${jobId}/status`);
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.error ?? "Failed to poll preprocessing status");
    }
    const data: PreprocessStatusResponse = await res.json();
    setState(data);
    return data;
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;

    let active = true;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const loop = async () => {
      if (!active) return;
      try {
        const data = await pollStatus();
        if (!data || data.status === "COMPLETE" || data.status === "FAILED") {
          return;
        }
      } catch {
        // Ignore polling errors here; the UI will surface the last known state.
      }
      timeoutId = setTimeout(loop, intervalMs);
    };

    loop();

    return () => {
      active = false;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [intervalMs, jobId, pollStatus]);

  return state;
}

/**
 * Preprocess pipeline UI with dataset selection, optional weather linking,
 * and live job status feedback.
 */
export default function PreprocessManager({
  initialDatasets,
  initialWeatherDatasets,
  initialError = null,
}: PreprocessManagerProps) {
  const [datasets] = useState<DatasetRecord[]>(initialDatasets);
  const [weatherDatasets] = useState<WeatherDatasetRecord[]>(
    initialWeatherDatasets,
  );
  const [error] = useState<string | null>(initialError);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(
    initialDatasets.length === 1 ? initialDatasets[0].id : null,
  );
  const [selectedWeatherId, setSelectedWeatherId] = useState<string | null>(
    null,
  );
  const [splits, setSplits] = useState({ train: 0.7, val: 0.15, test: 0.15 });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<PreprocessStatusResponse | null>(
    null,
  );

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId],
  );

  const filteredWeatherDatasets = useMemo(() => {
    if (!selectedDatasetId) return weatherDatasets;
    return weatherDatasets.filter(
      (record) => resolveWeatherDatasetId(record) === selectedDatasetId,
    );
  }, [selectedDatasetId, weatherDatasets]);

  const activeStatus = usePreprocessPoller(jobId) ?? jobStatus;

  const isRunning =
    activeStatus?.status === "RUNNING" || activeStatus?.status === "PENDING";

  useEffect(() => {
    if (!activeStatus) return;
    // Show a toast when the job completes or fails. We avoid writing state
    // here to prevent cascading renders; `jobStatus` is set when starting
    // the job and the poller updates UI via `usePreprocessPoller`.
    if (activeStatus.status === "COMPLETE") {
      toast.success("Preprocessing completed");
    } else if (activeStatus.status === "FAILED") {
      toast.error(activeStatus.error ?? "Preprocessing failed");
    }
  }, [activeStatus]);

  const updateSplit = (key: keyof typeof splits, value: string) => {
    const parsed = Number.parseFloat(value);
    if (Number.isNaN(parsed)) return;
    setSplits((prev) => ({
      ...prev,
      // Clamp split values to [0, 1] to avoid invalid payloads.
      [key]: Math.min(1, Math.max(0, parsed)),
    }));
  };

  const handleRun = async () => {
    if (!selectedDataset) {
      toast.error("Select a dataset to preprocess");
      return;
    }

    const datasetUrl = resolveFileUrl(selectedDataset);
    if (!datasetUrl) {
      toast.error("Dataset URL is missing for preprocessing");
      return;
    }

    const total = splits.train + splits.val + splits.test;
    if (Math.abs(total - 1) > 0.001) {
      toast.error("Split ratios must sum to 1.0");
      return;
    }

    const weatherDataset = weatherDatasets.find(
      (record) => record.id === selectedWeatherId,
    );
    const weatherUrl = weatherDataset ? resolveFileUrl(weatherDataset) : null;

    const payload = {
      job_id: createJobId(),
      dataset_id: selectedDataset.id,
      dataset_url: datasetUrl,
      weather_url: weatherUrl ?? undefined,
      splits: {
        train: splits.train,
        val: splits.val,
        test: splits.test,
      },
    };

    try {
      setIsSubmitting(true);
      const res = await fetch("/api/preprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? "Failed to start preprocessing");
      }
      const data: PreprocessStatusResponse = await res.json();
      setJobId(data.job_id ?? payload.job_id);
      setJobStatus(data);
      toast.success("Preprocessing started");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to start preprocessing";
      toast.error(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Support snake_case API responses while keeping the UI typed in camelCase.
  const summary = normalizeSummary(activeStatus?.result_summary ?? null);
  const edaCharts = normalizeEdaCharts(activeStatus?.eda_charts ?? null);

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <Card className="lg:col-span-1">
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
          <CardDescription>
            Select a dataset, optional weather data, and split ratios.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {error ? (
            <div className="rounded-lg border border-dashed px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          ) : null}

          <div className="space-y-2">
            <Label>Dataset</Label>
            <Select
              value={selectedDatasetId ?? undefined}
              onValueChange={(value) => {
                setSelectedDatasetId(value);
                setSelectedWeatherId(null);
              }}
              disabled={isRunning}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a dataset" />
              </SelectTrigger>
              <SelectContent>
                {datasets.map((dataset) => (
                  <SelectItem key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Weather Dataset (Optional)</Label>
            <Select
              value={selectedWeatherId ?? undefined}
              onValueChange={(value) => setSelectedWeatherId(value)}
              disabled={isRunning}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select weather data" />
              </SelectTrigger>
              <SelectContent>
                {filteredWeatherDatasets.length === 0 ? (
                  <SelectItem value="none" disabled>
                    No weather datasets available
                  </SelectItem>
                ) : (
                  filteredWeatherDatasets.map((record) => (
                    <SelectItem key={record.id} value={record.id}>
                      Weather dataset {record.id.slice(0, 8)}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Weather uploads must match the selected dataset.
            </p>
          </div>

          <div className="space-y-3">
            <Label>Split Ratios</Label>
            <div className="grid grid-cols-3 gap-3">
              <div className="space-y-1">
                <Label className="text-muted-foreground">Train</Label>
                <Input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={splits.train}
                  onChange={(event) => updateSplit("train", event.target.value)}
                  disabled={isRunning}
                />
              </div>
              <div className="space-y-1">
                <Label className="text-muted-foreground">Val</Label>
                <Input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={splits.val}
                  onChange={(event) => updateSplit("val", event.target.value)}
                  disabled={isRunning}
                />
              </div>
              <div className="space-y-1">
                <Label className="text-muted-foreground">Test</Label>
                <Input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={splits.test}
                  onChange={(event) => updateSplit("test", event.target.value)}
                  disabled={isRunning}
                />
              </div>
            </div>
          </div>

          <Button
            className="w-full"
            onClick={handleRun}
            disabled={!selectedDatasetId || isSubmitting || isRunning}
          >
            {isSubmitting ? "Starting..." : "Run Preprocessing"}
          </Button>

          {activeStatus ? (
            <div className="space-y-3 rounded-lg border p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs uppercase text-muted-foreground">
                  Status
                </p>
                <Badge
                  variant="secondary"
                  className={STATUS_STYLES[activeStatus.status]}
                >
                  {activeStatus.status}
                </Badge>
              </div>
              <Progress
                value={
                  activeStatus.progress ??
                  (activeStatus.status === "COMPLETE" ? 100 : 0)
                }
              />
              {activeStatus.processed_file_path ? (
                <div className="mt-2 flex items-center justify-end gap-2">
                  <Button
                    asChild
                    variant="secondary"
                    onClick={() =>
                      window.open(
                        `/dashboard/preprocess/${activeStatus.job_id ?? jobId}/view`,
                        "_blank",
                      )
                    }
                  >
                    <a>View results</a>
                  </Button>
                  <Button
                    variant="ghost"
                    onClick={() =>
                      window.open(activeStatus.processed_file_path, "_blank")
                    }
                  >
                    Download file
                  </Button>
                </div>
              ) : null}
              {activeStatus.error ? (
                <p className="text-xs text-destructive">{activeStatus.error}</p>
              ) : null}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Exploratory Data Analysis</CardTitle>
          <CardDescription>
            Summary metrics and EDA outputs generated after preprocessing.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {summary ? (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg border p-4">
                <p className="text-xs uppercase text-muted-foreground">Rows</p>
                <p className="text-lg font-semibold">{summary.rowCount}</p>
              </div>
              <div className="rounded-lg border p-4">
                <p className="text-xs uppercase text-muted-foreground">
                  Features
                </p>
                <p className="text-lg font-semibold">{summary.featureCount}</p>
              </div>
              <div className="rounded-lg border p-4">
                <p className="text-xs uppercase text-muted-foreground">
                  Missing Values
                </p>
                <p className="text-sm font-medium">
                  Before: {summary.missingValuesBefore}
                </p>
                <p className="text-sm font-medium">
                  After: {summary.missingValuesAfter}
                </p>
              </div>
              <div className="rounded-lg border p-4">
                <p className="text-xs uppercase text-muted-foreground">
                  Outliers
                </p>
                <p className="text-sm font-medium">
                  Count: {summary.outlierCount}
                </p>
                <p className="text-sm font-medium">
                  Rate: {summary.outlierPercentage}%
                </p>
              </div>
            </div>
          ) : (
            <div className="flex h-32 items-center justify-center rounded-lg border border-dashed">
              <p className="text-sm text-muted-foreground">
                Run preprocessing to generate summary metrics.
              </p>
            </div>
          )}

          {edaCharts ? (
            <div className="space-y-6">
              {/* Render chart widgets once the backend sends EDA payloads. */}
              <div className="grid gap-4 lg:grid-cols-2">
                <CorrelationHeatmap
                  matrix={edaCharts.correlationMatrix}
                  labels={edaCharts.featureNames}
                />
                <SeasonalDecomposition data={edaCharts.seasonalDecomposition} />
              </div>
              <div className="rounded-lg border p-4 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">
                    Distribution series
                  </span>
                  <span>
                    {Object.keys(edaCharts.distributionData ?? {}).length}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex h-32 items-center justify-center rounded-lg border border-dashed">
              <p className="text-sm text-muted-foreground">
                Run preprocessing to generate EDA outputs.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
