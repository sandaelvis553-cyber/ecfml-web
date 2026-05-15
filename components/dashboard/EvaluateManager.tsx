"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { AgentReasoningPanel } from "@/components/agent/AgentReasoningPanel";
import { ActualVsPredicted } from "@/components/charts/ActualVsPredicted";
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";
import { ModelComparisonTable } from "@/components/dashboard/ModelComparisonTable";
import { extractApiErrorMessage, unwrapApiData } from "@/lib/api-contract";
import {
  mapModelTypeToSourceType,
  normalizeEvaluationResult,
  normalizeModelRunRecord,
  normalizePreprocessJobRecord,
  readEvaluationCache,
  type EvaluationApiRecord,
  type ModelRunApiRecord,
  type PreprocessJobApiRecord,
} from "@/lib/evaluation-utils";
import type { EvaluationResult, SourceType } from "@/types/model";

type NormalizedRun = ReturnType<typeof normalizeModelRunRecord>;

function getRunId(run: NormalizedRun) {
  return run.job_id;
}

export default function EvaluateManager() {
  const [results, setResults] = useState<EvaluationResult[]>([]);
  const [selectedSourceType, setSelectedSourceType] =
    useState<SourceType>("RF");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selectedSourceTypeRef = useRef<SourceType>("RF");

  useEffect(() => {
    selectedSourceTypeRef.current = selectedSourceType;
  }, [selectedSourceType]);

  const selectedResult = useMemo(
    () =>
      results.find((result) => result.sourceType === selectedSourceType) ??
      results[0] ??
      null,
    [results, selectedSourceType],
  );

  const loadResults = useCallback(
    async (mode: "initial" | "refresh" = "refresh") => {
      if (mode === "initial") {
        setLoading(true);
      } else {
        setRefreshing(true);
      }

      try {
        setError(null);
        const [modelsResponse, preprocessResponse] = await Promise.all([
          fetch("/api/models/jobs"),
          fetch("/api/preprocess/jobs"),
        ]);

        const modelsPayload = await modelsResponse.json().catch(() => null);
        const preprocessPayload = await preprocessResponse
          .json()
          .catch(() => null);

        if (!modelsResponse.ok) {
          throw new Error(
            extractApiErrorMessage(modelsPayload, "Failed to fetch model runs"),
          );
        }

        if (!preprocessResponse.ok) {
          throw new Error(
            extractApiErrorMessage(
              preprocessPayload,
              "Failed to fetch preprocessing jobs",
            ),
          );
        }

        const modelRuns = (
          unwrapApiData<ModelRunApiRecord[]>(modelsPayload) ?? []
        )
          .map(normalizeModelRunRecord)
          .filter((run) => run.status === "COMPLETE")
          .filter(
            (run) =>
              run.model_type === "RANDOM_FOREST" || run.model_type === "SVR",
          );

        const preprocessJobs = (
          unwrapApiData<PreprocessJobApiRecord[]>(preprocessPayload) ?? []
        ).map(normalizePreprocessJobRecord);

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
        const warnings: string[] = [];

        for (const [sourceType, run] of latestBySourceType.entries()) {
          const modelRunId = getRunId(run);
          const cachedEvaluation = readEvaluationCache(modelRunId);
          if (cachedEvaluation) {
            evaluationResults.push(cachedEvaluation);
            continue;
          }

          const modelFilePath = run.model_file_path ?? null;
          const processedFilePath =
            preprocessById.get(run.preprocess_job_id)?.processed_file_path ??
            null;

          if (!modelRunId || !modelFilePath || !processedFilePath) {
            warnings.push(
              `Missing evaluation inputs for ${sourceType}. Train again to refresh artifacts.`,
            );
            continue;
          }

          const evaluationResponse = await fetch(
            `/api/models/${encodeURIComponent(modelRunId)}/evaluate`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model_run_id: modelRunId,
                model_file_path: modelFilePath,
                processed_file_path: processedFilePath,
              }),
            },
          );

          const evaluationPayload = await evaluationResponse
            .json()
            .catch(() => null);

          if (!evaluationResponse.ok) {
            warnings.push(
              extractApiErrorMessage(
                evaluationPayload,
                `Failed to evaluate ${sourceType} run`,
              ),
            );
            continue;
          }

          const normalized = normalizeEvaluationResult(
            unwrapApiData<EvaluationApiRecord>(evaluationPayload),
            sourceType,
          );
          evaluationResults.push(normalized);
          try {
            window.sessionStorage.setItem(
              `ecfml:evaluation:${modelRunId}`,
              JSON.stringify(normalized),
            );
          } catch {
            // Ignore cache write failures.
          }
        }

        evaluationResults.sort((left, right) => {
          const order: Record<SourceType, number> = { RF: 0, SVR: 1, AGENT: 2 };
          return order[left.sourceType] - order[right.sourceType];
        });

        setResults(evaluationResults);
        if (evaluationResults.length > 0) {
          const nextSelected =
            evaluationResults.find(
              (result) => result.sourceType === selectedSourceTypeRef.current,
            ) ?? evaluationResults[0];
          setSelectedSourceType(nextSelected.sourceType);
        }

        if (warnings.length > 0) {
          toast.error(warnings[0]);
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to load evaluations";
        setError(message);
        toast.error(message);
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [],
  );

  useEffect(() => {
    queueMicrotask(() => {
      void loadResults("initial");
    });

    const handleEvaluationComplete = () => {
      void loadResults("refresh");
    };

    window.addEventListener(
      "ecfml:evaluation-complete",
      handleEvaluationComplete,
    );

    return () => {
      window.removeEventListener(
        "ecfml:evaluation-complete",
        handleEvaluationComplete,
      );
    };
  }, [loadResults]);

  if (loading) {
    return (
      <div className="space-y-8">
        <div>
          <Skeleton className="h-10 w-72" />
          <Skeleton className="mt-3 h-5 w-115" />
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          <Skeleton className="h-105 rounded-xl" />
          <Skeleton className="h-105 rounded-xl" />
        </div>
      </div>
    );
  }

  if (error && results.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model Evaluation</CardTitle>
          <CardDescription>
            Unable to load evaluation results from the backend contract.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">{error}</p>
          <Button variant="outline" onClick={() => void loadResults("refresh")}>
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (results.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model Evaluation</CardTitle>
          <CardDescription>
            Train RF or SVR models first to populate the dashboard.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Spinner />
            No completed model runs found.
          </div>
          <Button variant="outline" onClick={() => void loadResults("refresh")}>
            Refresh
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Model Evaluation
          </h1>
          <p className="mt-1 text-muted-foreground">
            Compare the latest completed Random Forest and SVR runs. Charts and
            warnings are driven by the FastAPI evaluation contract.
          </p>
        </div>
        <Button variant="outline" onClick={() => void loadResults("refresh")}>
          {refreshing ? "Refreshing..." : "Refresh"}
        </Button>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-4">
            <div>
              <CardTitle>Performance Comparison</CardTitle>
              <CardDescription>
                RMSE, MAE, MAPE, and R² for the latest completed engine runs.
              </CardDescription>
            </div>
            <Badge variant="outline">{results.length} evaluated run(s)</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <ModelComparisonTable results={results} />
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Chart Target</CardTitle>
            <CardDescription>
              Select the engine whose evaluation output you want to inspect.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Engine</label>
              <Select
                value={selectedSourceType}
                onValueChange={(value) =>
                  setSelectedSourceType(value as SourceType)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select an engine" />
                </SelectTrigger>
                <SelectContent>
                  {results.map((result) => (
                    <SelectItem key={result.id} value={result.sourceType}>
                      {result.sourceType} — MAPE {result.mape.toFixed(1)}%
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedResult?.mape > 20 && (
              <Badge variant="destructive" className="w-fit">
                MAPE above 20% warning
              </Badge>
            )}

            <div className="rounded-lg border bg-muted/40 p-4 text-sm text-muted-foreground">
              The selected engine drives the chart panels on the right. RF rows
              may include feature importance, while AGENT rows can surface
              reasoning if the backend provides it.
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6">
          {selectedResult ? (
            <ActualVsPredicted
              actual={selectedResult.actualJson}
              predicted={selectedResult.predictedJson}
              title={`${selectedResult.sourceType} Actual vs Predicted`}
              description={`Test size: ${selectedResult.testSetSize}`}
            />
          ) : null}

          {selectedResult?.sourceType === "RF" &&
          selectedResult.featureImportanceJson ? (
            <FeatureImportanceChart
              data={selectedResult.featureImportanceJson}
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Feature Importance</CardTitle>
                <CardDescription>
                  Random Forest exposes feature contributions from the same
                  evaluation.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex h-64 items-center justify-center rounded-lg border border-dashed">
                  <p className="text-sm text-muted-foreground">
                    Select a Random Forest result to view feature importance.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {selectedResult?.reasoning ? (
            <AgentReasoningPanel
              reasoning={selectedResult.reasoning}
              confidence={selectedResult.confidence ?? undefined}
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>LLM Agent Reasoning</CardTitle>
                <CardDescription>
                  Reserved for agent-backed evaluation payloads returned by the
                  backend.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex h-32 items-center justify-center rounded-lg border border-dashed">
                  <p className="text-sm text-muted-foreground">
                    Agent reasoning will appear when an AGENT evaluation result
                    is available.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
