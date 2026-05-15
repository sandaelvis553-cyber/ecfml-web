import type {
  EvaluationResult,
  FeatureImportance,
  SourceType,
  TimeseriesPoint,
} from "@/types/model";

export interface ModelRunApiRecord {
  job_id?: string;
  jobId?: string;
  id?: string;
  status?: string;
  model_type?: string;
  modelType?: string;
  preprocess_job_id?: string;
  preprocessJobId?: string;
  model_file_path?: string | null;
  modelFilePath?: string | null;
  scaler_file_path?: string | null;
  scalerFilePath?: string | null;
  training_time_secs?: number | null;
  trainingTimeSecs?: number | null;
  error?: string | null;
  created_at?: string;
  createdAt?: string;
}

export interface PreprocessJobApiRecord {
  job_id?: string;
  jobId?: string;
  id?: string;
  status?: string;
  processed_file_path?: string | null;
  processedFilePath?: string | null;
  created_at?: string;
  createdAt?: string;
  result_summary?: Record<string, unknown> | null;
  resultSummary?: Record<string, unknown> | null;
}

export interface EvaluationApiRecord {
  id?: string;
  model_run_id?: string | null;
  modelRunId?: string | null;
  agent_run_id?: string | null;
  agentRunId?: string | null;
  source_type?: SourceType | string;
  sourceType?: SourceType | string;
  rmse?: number;
  mae?: number;
  mape?: number;
  r2?: number;
  test_set_size?: number;
  testSetSize?: number;
  actual?: unknown;
  actualJson?: unknown;
  predicted?: unknown;
  predictedJson?: unknown;
  feature_importance?: unknown;
  featureImportance?: unknown;
  featureImportanceJson?: unknown;
  evaluated_at?: string;
  evaluatedAt?: string;
  reasoning?: string | null;
  agent_reasoning?: string | null;
  agentReasoning?: string | null;
  confidence?: string | null;
}

export function getEvaluationCacheKey(modelRunId: string) {
  return `ecfml:evaluation:${modelRunId}`;
}

export function mapModelTypeToSourceType(
  modelType?: string | null,
): SourceType {
  if (modelType === "SVR") return "SVR";
  if (modelType === "AGENT") return "AGENT";
  return "RF";
}

function coerceNumber(value: unknown, fallback = 0) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function normalizeTimeseriesPoint(value: unknown): TimeseriesPoint | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const timestamp =
    typeof record.timestamp === "string"
      ? record.timestamp
      : typeof record.time === "string"
        ? record.time
        : typeof record.date === "string"
          ? record.date
          : typeof record.created_at === "string"
            ? record.created_at
            : typeof record.createdAt === "string"
              ? record.createdAt
              : new Date().toISOString();
  const rawValue =
    record.value ??
    record.predicted ??
    record.actual ??
    record.consumption ??
    record.y;
  return {
    timestamp,
    value: coerceNumber(rawValue, 0),
  };
}

export function normalizeTimeseriesPoints(value: unknown): TimeseriesPoint[] {
  if (!Array.isArray(value)) return [];
  return value
    .map(normalizeTimeseriesPoint)
    .filter(Boolean) as TimeseriesPoint[];
}

function normalizeFeatureImportancePoint(
  value: unknown,
): FeatureImportance | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const feature =
    typeof record.feature === "string"
      ? record.feature
      : typeof record.name === "string"
        ? record.name
        : typeof record.label === "string"
          ? record.label
          : null;
  if (!feature) return null;
  return {
    feature,
    importance: coerceNumber(
      record.importance ?? record.score ?? record.weight,
      0,
    ),
  };
}

export function normalizeFeatureImportance(
  value: unknown,
): FeatureImportance[] | null {
  if (!Array.isArray(value)) return null;
  const items = value
    .map(normalizeFeatureImportancePoint)
    .filter(Boolean) as FeatureImportance[];
  return items.length > 0 ? items : null;
}

export function normalizeModelRunRecord(record: ModelRunApiRecord) {
  return {
    job_id: record.job_id ?? record.jobId ?? record.id ?? "",
    status: record.status ?? "PENDING",
    model_type: (record.model_type ??
      record.modelType ??
      "RANDOM_FOREST") as string,
    preprocess_job_id: record.preprocess_job_id ?? record.preprocessJobId ?? "",
    model_file_path: record.model_file_path ?? record.modelFilePath ?? null,
    scaler_file_path: record.scaler_file_path ?? record.scalerFilePath ?? null,
    training_time_secs:
      record.training_time_secs ?? record.trainingTimeSecs ?? null,
    error: record.error ?? null,
    created_at:
      record.created_at ?? record.createdAt ?? new Date().toISOString(),
  };
}

export function normalizePreprocessJobRecord(record: PreprocessJobApiRecord) {
  return {
    job_id: record.job_id ?? record.jobId ?? record.id ?? "",
    status: record.status ?? "PENDING",
    processed_file_path:
      record.processed_file_path ?? record.processedFilePath ?? null,
    created_at:
      record.created_at ?? record.createdAt ?? new Date().toISOString(),
    result_summary: record.result_summary ?? record.resultSummary ?? null,
  };
}

export function normalizeEvaluationResult(
  record: EvaluationApiRecord,
  sourceTypeFallback?: SourceType,
): EvaluationResult {
  const modelRunId = record.model_run_id ?? record.modelRunId ?? null;
  const agentRunId = record.agent_run_id ?? record.agentRunId ?? null;
  const sourceType =
    (record.sourceType as SourceType | undefined) ??
    (record.source_type as SourceType | undefined) ??
    sourceTypeFallback ??
    (agentRunId ? "AGENT" : modelRunId ? "RF" : "RF");

  return {
    id:
      record.id ??
      modelRunId ??
      agentRunId ??
      `${sourceType.toLowerCase()}-${Date.now()}`,
    modelRunId,
    agentRunId,
    sourceType,
    rmse: coerceNumber(record.rmse, 0),
    mae: coerceNumber(record.mae, 0),
    mape: coerceNumber(record.mape, 0),
    r2: coerceNumber(record.r2, 0),
    testSetSize: coerceNumber(record.testSetSize ?? record.test_set_size, 0),
    actualJson: normalizeTimeseriesPoints(record.actual ?? record.actualJson),
    predictedJson: normalizeTimeseriesPoints(
      record.predicted ?? record.predictedJson,
    ),
    featureImportanceJson: normalizeFeatureImportance(
      record.feature_importance ??
        record.featureImportance ??
        record.featureImportanceJson,
    ),
    evaluatedAt:
      record.evaluated_at ?? record.evaluatedAt ?? new Date().toISOString(),
    reasoning:
      record.reasoning ??
      record.agent_reasoning ??
      record.agentReasoning ??
      null,
    confidence: record.confidence ?? null,
  };
}

export function readEvaluationCache(modelRunId: string) {
  if (typeof window === "undefined") return null;

  try {
    const raw = window.sessionStorage.getItem(
      getEvaluationCacheKey(modelRunId),
    );
    if (!raw) return null;
    return JSON.parse(raw) as EvaluationResult;
  } catch {
    return null;
  }
}

export function writeEvaluationCache(
  modelRunId: string,
  result: EvaluationResult,
) {
  if (typeof window === "undefined") return;

  try {
    window.sessionStorage.setItem(
      getEvaluationCacheKey(modelRunId),
      JSON.stringify(result),
    );
  } catch {
    // Ignore storage failures.
  }
}
