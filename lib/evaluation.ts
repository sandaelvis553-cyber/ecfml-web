import type {
  EvaluationResult,
  FeatureImportance,
  ModelType,
  SourceType,
  TimeseriesPoint,
} from "@/types/model";

export interface EvaluationApiResponse {
  rmse: number;
  mae: number;
  mape: number;
  r2: number;
  test_set_size: number;
  actual: TimeseriesPoint[];
  predicted: TimeseriesPoint[];
  feature_importance?: FeatureImportance[] | null;
  source_type?: SourceType;
  agent_reasoning?: string | null;
  confidence?: "high" | "medium" | "low" | null;
}

export interface EvaluationRecord extends EvaluationResult {
  agentReasoning?: string | null;
  confidence?: "high" | "medium" | "low" | null;
}

export function sourceTypeFromModelType(modelType: ModelType): SourceType {
  return modelType === "SVR" ? "SVR" : "RF";
}

export function evaluationCacheKey(modelRunId: string): string {
  return `ecfml:evaluation:${modelRunId}`;
}

export function normalizeEvaluationResult(
  payload: EvaluationApiResponse,
  fallback: {
    modelRunId: string;
    sourceType: SourceType;
  },
): EvaluationRecord {
  return {
    id: fallback.modelRunId,
    modelRunId: fallback.sourceType === "AGENT" ? null : fallback.modelRunId,
    agentRunId: fallback.sourceType === "AGENT" ? fallback.modelRunId : null,
    sourceType: payload.source_type ?? fallback.sourceType,
    rmse: payload.rmse,
    mae: payload.mae,
    mape: payload.mape,
    r2: payload.r2,
    testSetSize: payload.test_set_size,
    actualJson: payload.actual,
    predictedJson: payload.predicted,
    featureImportanceJson: payload.feature_importance ?? null,
    evaluatedAt: new Date().toISOString(),
    agentReasoning: payload.agent_reasoning ?? null,
    confidence: payload.confidence ?? null,
  };
}
