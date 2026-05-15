/** ML Model-related TypeScript types */

export type ModelType = "RANDOM_FOREST" | "SVR";
export type JobStatus =
  | "PENDING"
  | "RUNNING"
  | "COMPLETE"
  | "FAILED"
  | "WARNING";
export type SourceType = "RF" | "SVR" | "AGENT";

export interface ModelRun {
  id: string;
  userId: string;
  preprocessJobId: string;
  modelType: ModelType;
  hyperparamsJson: RFHyperparams | SVRHyperparams;
  status: JobStatus;
  modelFilePath: string | null;
  scalerFilePath: string | null;
  trainingTimeSecs: number | null;
  createdAt: string;
}

export interface RFHyperparams {
  n_estimators: number;
  max_depth: number | null;
  min_samples_split: number;
  enable_grid_search: boolean;
}

export interface SVRHyperparams {
  kernel: "rbf" | "linear" | "poly";
  C: number;
  epsilon: number;
  gamma: "scale" | "auto" | number;
}

export interface PreprocessingJob {
  id: string;
  userId: string;
  datasetId: string;
  status: JobStatus;
  splitRatioTrain: number;
  splitRatioVal: number;
  splitRatioTest: number;
  resultSummaryJson: PreprocessSummary | null;
  edaChartsJson: EDACharts | null;
  processedFilePath: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface PreprocessSummary {
  rowCount: number;
  featureCount: number;
  missingValuesBefore: number;
  missingValuesAfter: number;
  outlierCount: number;
  outlierPercentage: number;
}

export interface EDACharts {
  correlationMatrix: number[][];
  featureNames: string[];
  distributionData: Record<string, number[]>;
  seasonalDecomposition: {
    trend: number[];
    seasonal: number[];
    residual: number[];
    timestamps: string[];
  } | null;
}

export interface EvaluationResult {
  id: string;
  modelRunId: string | null;
  agentRunId: string | null;
  sourceType: SourceType;
  rmse: number;
  mae: number;
  mape: number;
  r2: number;
  testSetSize: number;
  actualJson: TimeseriesPoint[];
  predictedJson: TimeseriesPoint[];
  featureImportanceJson: FeatureImportance[] | null;
  evaluatedAt: string;
  reasoning?: string | null;
  confidence?: "high" | "medium" | "low" | string | null;
}

export interface TimeseriesPoint {
  timestamp: string;
  value: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}
