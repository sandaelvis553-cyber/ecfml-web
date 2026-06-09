import type { Metadata } from "next";
import EvaluateManager from "@/components/dashboard/EvaluateManager";
import { mlFetch } from "@/lib/api-client";
import {
  mapModelTypeToSourceType,
  normalizeEvaluationResult,
  normalizeModelRunRecord,
  normalizePreprocessJobRecord,
  type EvaluationApiRecord,
  type ModelRunApiRecord,
  type PreprocessJobApiRecord,
} from "@/lib/evaluation-utils";
import type { EvaluationResult, SourceType } from "@/types/model";

type NormalizedRun = ReturnType<typeof normalizeModelRunRecord>;

export const metadata: Metadata = {
  title: "Model Evaluation",
  description:
    "Compare RF, SVR, and LangGraph Agent forecasting performance — RMSE, MAE, MAPE, R².",
};

export default async function EvaluatePage() {
  let initialResults: EvaluationResult[] = [];
  let initialError: string | null = null;

  try {
    const [modelsRes, preprocessRes] = await Promise.all([
      mlFetch<ModelRunApiRecord[] | { data: ModelRunApiRecord[] }>("/api/v1/models"),
      mlFetch<PreprocessJobApiRecord[]>("/api/v1/preprocessing/jobs"),
    ]);

    const modelRuns = (
      Array.isArray((modelsRes as { data?: ModelRunApiRecord[] })?.data)
        ? (modelsRes as { data: ModelRunApiRecord[] }).data
        : Array.isArray(modelsRes)
        ? modelsRes
        : []
    )
      .map(normalizeModelRunRecord)
      .filter((run) => run.status === "COMPLETE")
      .filter(
        (run) =>
          run.model_type === "RANDOM_FOREST" || run.model_type === "SVR",
      );

    const preprocessJobs = preprocessRes.map(normalizePreprocessJobRecord);
    const preprocessById = new Map(
      preprocessJobs.map((job) => [job.job_id, job] as const),
    );

    const latestBySourceType = new Map<SourceType, NormalizedRun>();

    for (const run of modelRuns) {
      const sourceType = mapModelTypeToSourceType(run.model_type);
      const existing = latestBySourceType.get(sourceType);

      if (!existing) {
        latestBySourceType.set(sourceType, run);
        continue;
      }

      const currentCreatedAt = new Date(run.created_at).getTime();
      const existingCreatedAt = new Date(existing.created_at).getTime();
      if (currentCreatedAt >= existingCreatedAt) {
        latestBySourceType.set(sourceType, run);
      }
    }

    const evaluationResults: EvaluationResult[] = [];

    for (const [sourceType, run] of latestBySourceType.entries()) {
      const modelRunId = run.job_id;
      const modelFilePath = run.model_file_path;
      const processedFilePath =
        preprocessById.get(run.preprocess_job_id)?.processed_file_path ??
        null;

      if (!modelRunId || !modelFilePath || !processedFilePath) {
        continue;
      }

      try {
        const evaluationPayload = await mlFetch<EvaluationApiRecord>(
          `/api/v1/models/${encodeURIComponent(modelRunId)}/evaluate`,
          {
            method: "POST",
            body: JSON.stringify({
              model_run_id: modelRunId,
              model_file_path: modelFilePath,
              processed_file_path: processedFilePath,
            }),
          },
        );

        const normalized = normalizeEvaluationResult(
          evaluationPayload,
          sourceType,
        );
        evaluationResults.push(normalized);
      } catch (err) {
        console.error(`[evaluate:ssr] Failed to evaluate ${sourceType} run:`, err);
      }
    }

    evaluationResults.sort((left, right) => {
      const order: Record<SourceType, number> = { RF: 0, SVR: 1, AGENT: 2 };
      return order[left.sourceType] - order[right.sourceType];
    });

    initialResults = evaluationResults;
  } catch (error) {
    initialError =
      error instanceof Error ? error.message : "Failed to load evaluation data";
  }

  return (
    <EvaluateManager
      initialResults={initialResults}
      initialError={initialError}
    />
  );
}

