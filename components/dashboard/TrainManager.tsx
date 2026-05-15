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
import { ScrollArea } from "@/components/ui/scroll-area";
import { useJobPoller } from "@/hooks/use-job-poller";
import { extractApiErrorMessage, unwrapApiData } from "@/lib/api-contract";
import {
  mapModelTypeToSourceType,
  normalizeEvaluationResult,
  normalizeModelRunRecord,
  normalizePreprocessJobRecord,
  readEvaluationCache,
  writeEvaluationCache,
  type EvaluationApiRecord,
  type ModelRunApiRecord,
  type PreprocessJobApiRecord,
} from "@/lib/evaluation-utils";
import type {
  JobStatus,
  ModelType,
  RFHyperparams,
  SVRHyperparams,
  EvaluationResult,
} from "@/types/model";

interface PreprocessJobRecord {
  job_id: string;
  status: JobStatus;
  created_at: string;
  result_summary?: {
    row_count?: number;
    feature_count?: number;
    missing_values_before?: number;
    missing_values_after?: number;
    outlier_count?: number;
    outlier_percentage?: number;
  } | null;
  processed_file_path?: string;
}

interface TrainingHistoryRecord {
  job_id: string;
  status: JobStatus;
  model_type: ModelType;
  preprocess_job_id?: string;
  model_file_path?: string | null;
  created_at: string;
  training_time_secs?: number;
  error?: string;
}

interface ModelRunListRecord {
  job_id?: string;
  jobId?: string;
  id?: string;
  status?: JobStatus;
  model_type?: ModelType | string;
  modelType?: ModelType | string;
  preprocess_job_id?: string;
  preprocessJobId?: string;
  model_file_path?: string | null;
  modelFilePath?: string | null;
  created_at?: string;
  createdAt?: string;
  training_time_secs?: number | null;
  trainingTimeSecs?: number | null;
  error?: string | null;
}

const normalizeHistoryRecord = (
  record: ModelRunListRecord,
): TrainingHistoryRecord => {
  const normalized = normalizeModelRunRecord(record);

  return {
    job_id: normalized.job_id,
    status: normalized.status as JobStatus,
    model_type: (normalized.model_type ?? "RANDOM_FOREST") as ModelType,
    preprocess_job_id: normalized.preprocess_job_id,
    model_file_path: normalized.model_file_path,
    created_at: normalized.created_at,
    training_time_secs: normalized.training_time_secs ?? undefined,
    error: normalized.error ?? undefined,
  };
};

const STATUS_STYLES: Record<JobStatus, string> = {
  PENDING: "text-muted-foreground",
  RUNNING: "text-blue-500 dark:text-blue-400",
  COMPLETE: "text-emerald-500 dark:text-emerald-400",
  FAILED: "text-red-500 dark:text-red-400",
  WARNING: "text-amber-500 dark:text-amber-400",
};

const DEFAULT_RF_PARAMS: RFHyperparams = {
  n_estimators: 200,
  max_depth: null,
  min_samples_split: 2,
  enable_grid_search: false,
};

const DEFAULT_SVR_PARAMS: SVRHyperparams = {
  kernel: "rbf",
  C: 1.0,
  epsilon: 0.1,
  gamma: "scale",
};

/**
 * Component for configuring and training ML models.
 */
export default function TrainManager() {
  const [jobs, setJobs] = useState<PreprocessJobRecord[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<
    TrainingHistoryRecord[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] =
    useState<ModelType>("RANDOM_FOREST");
  const [isTraining, setIsTraining] = useState(false);
  const [currentTrainingJobId, setCurrentTrainingJobId] = useState<
    string | null
  >(null);
  const [lastEvaluationResult, setLastEvaluationResult] =
    useState<EvaluationResult | null>(null);
  const [lastEvaluationJobId, setLastEvaluationJobId] = useState<string | null>(
    null,
  );
  const evaluatedJobIdsRef = useRef<Set<string>>(new Set());

  // Hyperparameters
  const [rfParams, setRfParams] = useState<RFHyperparams>(DEFAULT_RF_PARAMS);
  const [svrParams, setSvrParams] =
    useState<SVRHyperparams>(DEFAULT_SVR_PARAMS);

  // Fetch preprocessing jobs on mount
  useEffect(() => {
    const fetchJobs = async () => {
      try {
        setLoading(true);
        const res = await fetch("/api/preprocess/jobs");
        const payload = await res.json().catch(() => null);
        if (!res.ok) {
          throw new Error(
            extractApiErrorMessage(
              payload,
              "Failed to fetch preprocessing jobs",
            ),
          );
        }
        const data = unwrapApiData<PreprocessJobRecord[]>(payload);
        // Filter to only completed jobs
        const completed = (Array.isArray(data) ? data : []).filter(
          (job: PreprocessJobRecord) => job.status === "COMPLETE",
        );
        setJobs(completed);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to load jobs";
        toast.error(message);
        setJobs([]);
      } finally {
        setLoading(false);
      }
    };

    fetchJobs();
  }, []);

  // Fetch training history
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch("/api/models/jobs");
        const payload = await res.json().catch(() => null);
        if (!res.ok) {
          throw new Error(
            extractApiErrorMessage(payload, "Failed to fetch training history"),
          );
        }
        const data = unwrapApiData<ModelRunListRecord[]>(payload);
        setTrainingHistory(
          Array.isArray(data) ? data.map(normalizeHistoryRecord) : [],
        );
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to load history";
        toast.error(message);
      }
    };

    fetchHistory();
  }, []);

  const selectedJob = useMemo(
    () => jobs.find((job) => job.job_id === selectedJobId) ?? null,
    [jobs, selectedJobId],
  );

  const handleStartTraining = useCallback(async () => {
    if (!selectedJob) {
      toast.error("Select a preprocessing job to train");
      return;
    }

    try {
      setIsTraining(true);
      setLastEvaluationResult(null);
      setLastEvaluationJobId(null);
      const hyperparams =
        selectedModel === "RANDOM_FOREST" ? rfParams : svrParams;

      const payload = {
        job_id: `train_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        preprocess_job_id: selectedJob.job_id,
        model_type: selectedModel,
        hyperparams,
        processed_file_path: selectedJob.processed_file_path,
      };

      const res = await fetch("/api/models/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const body = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(
          extractApiErrorMessage(body, "Failed to start training"),
        );
      }

      const returned = (unwrapApiData<ModelRunListRecord | null>(body) ??
        {}) as ModelRunListRecord;
      const returnedJobId = returned.job_id ?? returned.jobId ?? payload.job_id;
      setCurrentTrainingJobId(returnedJobId ?? null);
      toast.success("Training started");
      setTrainingHistory((prev) => [
        normalizeHistoryRecord({
          job_id: returnedJobId ?? payload.job_id,
          status: (returned.status ?? "PENDING") as JobStatus,
          model_type: selectedModel,
          preprocess_job_id: selectedJob.job_id,
          model_file_path:
            (returned as ModelRunListRecord).model_file_path ??
            (returned as ModelRunListRecord).modelFilePath ??
            null,
          created_at: returned.created_at ?? new Date().toISOString(),
          training_time_secs: returned.training_time_secs ?? null,
          error: returned.error ?? null,
        }),
        ...prev,
      ]);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to start training";
      toast.error(message);
    } finally {
      setIsTraining(false);
    }
  }, [selectedJob, selectedModel, rfParams, svrParams]);

  // Poll training job status
  const trainingState = useJobPoller(currentTrainingJobId);

  useEffect(() => {
    if (!currentTrainingJobId || !trainingState) return;
    // Update history when status changes
    if (trainingState.status) {
      queueMicrotask(() => {
        setTrainingHistory((prev) => {
          const exists = prev.find((p) => p.job_id === currentTrainingJobId);
          const entry = {
            job_id: currentTrainingJobId,
            status: trainingState.status as JobStatus,
            model_type: selectedModel,
            created_at: new Date().toISOString(),
            training_time_secs: undefined,
            error: trainingState.error,
          } as TrainingHistoryRecord;
          if (exists) {
            return prev.map((p) =>
              p.job_id === currentTrainingJobId ? { ...p, ...entry } : p,
            );
          }
          return [entry, ...prev];
        });
      });
    }

    if (trainingState.status === "COMPLETE") {
      toast.success("Training completed");
      const completedJobId = currentTrainingJobId;
      const completedRecord = trainingHistory.find(
        (record) => record.job_id === completedJobId,
      );
      const completedSourceType = mapModelTypeToSourceType(
        completedRecord?.model_type ?? selectedModel,
      );
      if (!completedJobId || evaluatedJobIdsRef.current.has(completedJobId)) {
        setCurrentTrainingJobId(null);
        return;
      }

      evaluatedJobIdsRef.current.add(completedJobId);

      (async () => {
        try {
          const embeddedEvaluation =
            trainingState.evaluation_result ?? trainingState.evaluationResult;
          const embeddedModelRunId =
            trainingState.model_run_id ??
            trainingState.modelRunId ??
            completedJobId;
          const embeddedModelFilePath =
            trainingState.model_file_path ??
            trainingState.modelFilePath ??
            null;
          const embeddedProcessedFilePath =
            trainingState.processed_file_path ??
            trainingState.processedFilePath ??
            null;
          const historyPreprocessJobId =
            completedRecord?.preprocess_job_id ?? selectedJob?.job_id ?? null;
          const historyModelFilePath =
            completedRecord?.model_file_path ?? embeddedModelFilePath ?? null;
          if (embeddedModelRunId) {
            const cachedEvaluation = readEvaluationCache(embeddedModelRunId);
            if (cachedEvaluation) {
              setLastEvaluationResult(cachedEvaluation);
              setLastEvaluationJobId(embeddedModelRunId);
              window.dispatchEvent(
                new CustomEvent("ecfml:evaluation-complete", {
                  detail: { modelRunId: embeddedModelRunId },
                }),
              );
              return;
            }
          }

          if (embeddedEvaluation && embeddedModelRunId) {
            const normalized = normalizeEvaluationResult(
              embeddedEvaluation as EvaluationApiRecord,
              completedSourceType,
            );
            writeEvaluationCache(embeddedModelRunId, normalized);
            setLastEvaluationResult(normalized);
            setLastEvaluationJobId(embeddedModelRunId);
            window.dispatchEvent(
              new CustomEvent("ecfml:evaluation-complete", {
                detail: { modelRunId: embeddedModelRunId },
              }),
            );
            return;
          }

          let modelRunId = embeddedModelRunId;
          let modelFilePath = historyModelFilePath;
          let processedFilePath = embeddedProcessedFilePath;
          if (!modelRunId || !modelFilePath || !processedFilePath) {
            const runsResponse = await fetch("/api/models/jobs");
            const runsPayload = await runsResponse.json().catch(() => null);
            if (!runsResponse.ok) {
              throw new Error(
                extractApiErrorMessage(
                  runsPayload,
                  "Failed to fetch model runs",
                ),
              );
            }

            const runs = (
              unwrapApiData<ModelRunApiRecord[]>(runsPayload) ?? []
            ).map(normalizeModelRunRecord);
            const run = runs.find((record) => record.job_id === completedJobId);
            if (!run) {
              throw new Error("Completed model run not found");
            }

            modelRunId = run.job_id;
            modelFilePath = run.model_file_path ?? modelFilePath;
            // Use || to catch empty strings from backend (not just null/undefined)
            const preprocessJobId =
              run.preprocess_job_id ||
              historyPreprocessJobId ||
              selectedJob?.job_id ||
              null;

            const preprocessResponse = await fetch("/api/preprocess/jobs");
            const preprocessPayload = await preprocessResponse
              .json()
              .catch(() => null);
            if (!preprocessResponse.ok) {
              throw new Error(
                extractApiErrorMessage(
                  preprocessPayload,
                  "Failed to fetch preprocessing jobs",
                ),
              );
            }

            const preprocessJobs = (
              unwrapApiData<PreprocessJobApiRecord[]>(preprocessPayload) ?? []
            ).map(normalizePreprocessJobRecord);
            processedFilePath =
              preprocessJobs.find((job) => job.job_id === preprocessJobId)
                ?.processed_file_path ?? processedFilePath;
          }

          if (!modelRunId || !modelFilePath || !processedFilePath) {
            throw new Error(
              "Missing evaluation inputs for completed model run",
            );
          }

          const evalRes = await fetch(
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

          const evalBody = await evalRes.json().catch(() => null);

          if (!evalRes.ok) {
            const errPayload =
              typeof evalBody === "string"
                ? { error: { message: evalBody } }
                : evalBody;
            throw new Error(
              extractApiErrorMessage(errPayload, "Evaluation failed"),
            );
          }

          const evaluation = normalizeEvaluationResult(
            unwrapApiData<EvaluationApiRecord>(evalBody),
            completedSourceType,
          );
          writeEvaluationCache(modelRunId, evaluation);
          setLastEvaluationResult(evaluation);
          setLastEvaluationJobId(modelRunId);
          window.dispatchEvent(
            new CustomEvent("ecfml:evaluation-complete", {
              detail: { modelRunId },
            }),
          );
        } catch (err) {
          const message =
            err instanceof Error ? err.message : "Failed to run evaluation";
          toast.error(message);
        } finally {
          setCurrentTrainingJobId(null);
        }
      })();
    } else if (trainingState.status === "FAILED") {
      toast.error(trainingState.error ?? "Training failed");
      queueMicrotask(() => {
        setCurrentTrainingJobId(null);
      });
    }
  }, [
    currentTrainingJobId,
    selectedJob?.job_id,
    selectedModel,
    trainingHistory,
    trainingState,
  ]);

  const updateRFParam = (key: keyof RFHyperparams, value: unknown) => {
    setRfParams((prev) => ({ ...prev, [key]: value }));
  };

  const updateSVRParam = (key: keyof SVRHyperparams, value: unknown) => {
    setSvrParams((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Model configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Model Configuration</CardTitle>
          <CardDescription>
            Select a preprocessing job and configure hyperparameters.
          </CardDescription>
        </CardHeader>
        {(currentTrainingJobId || isTraining) && (
          <div className="px-4 pb-4">
            <div className="flex items-center justify-between">
              <div className="text-sm">
                Training job: {currentTrainingJobId ?? "pending"}
              </div>
              <div
                className={`text-sm ${STATUS_STYLES[(trainingState?.status ?? "PENDING") as JobStatus]}`}
              >
                {trainingState?.status ?? (isTraining ? "PENDING" : "IDLE")}
              </div>
            </div>
            <div className="mt-3">
              <Progress value={trainingState?.progress ?? 0} />
            </div>
            {trainingState?.error && (
              <p className="mt-2 text-sm text-red-500">{trainingState.error}</p>
            )}
          </div>
        )}
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="job-select">Pre-processing Job</Label>
            <Select
              value={selectedJobId ?? undefined}
              onValueChange={setSelectedJobId}
            >
              <SelectTrigger id="job-select" className="w-full">
                <SelectValue placeholder="Select a completed preprocessing job" />
              </SelectTrigger>
              <SelectContent>
                {loading ? (
                  <div className="px-2 py-1.5 text-sm text-muted-foreground">
                    Loading jobs...
                  </div>
                ) : jobs.length === 0 ? (
                  <div className="px-2 py-1.5 text-sm text-muted-foreground">
                    No completed preprocessing jobs
                  </div>
                ) : (
                  jobs.map((job) => (
                    <SelectItem key={job.job_id} value={job.job_id}>
                      {job.job_id} ({job.status})
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
            {selectedJob?.result_summary && (
              <div className="rounded-md bg-muted p-2 text-xs">
                <p>
                  <strong>Rows:</strong>{" "}
                  {selectedJob.result_summary.row_count ?? 0}
                </p>
                <p>
                  <strong>Features:</strong>{" "}
                  {selectedJob.result_summary.feature_count ?? 0}
                </p>
              </div>
            )}
          </div>

          <div className="space-y-2">
            <Label>Select Model</Label>
            <div className="flex gap-3">
              <Badge
                variant={
                  selectedModel === "RANDOM_FOREST" ? "default" : "secondary"
                }
                className="cursor-pointer px-4 py-1.5"
                onClick={() => setSelectedModel("RANDOM_FOREST")}
              >
                Random Forest
              </Badge>
              <Badge
                variant={selectedModel === "SVR" ? "default" : "secondary"}
                className="cursor-pointer px-4 py-1.5"
                onClick={() => setSelectedModel("SVR")}
              >
                SVR
              </Badge>
            </div>
          </div>

          {/* RF Hyperparameters */}
          {selectedModel === "RANDOM_FOREST" && (
            <div className="space-y-3 rounded-lg border p-4">
              <p className="text-sm font-medium">
                Random Forest Hyperparameters
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">
                    N Estimators
                  </label>
                  <Input
                    type="number"
                    value={rfParams.n_estimators}
                    onChange={(e) =>
                      updateRFParam(
                        "n_estimators",
                        parseInt(e.target.value) || 200,
                      )
                    }
                    className="h-9"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">
                    Max Depth
                  </label>
                  <Input
                    type="number"
                    placeholder="None"
                    value={rfParams.max_depth ?? ""}
                    onChange={(e) =>
                      updateRFParam(
                        "max_depth",
                        e.target.value ? parseInt(e.target.value) : null,
                      )
                    }
                    className="h-9"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">
                    Min Samples Split
                  </label>
                  <Input
                    type="number"
                    value={rfParams.min_samples_split}
                    onChange={(e) =>
                      updateRFParam(
                        "min_samples_split",
                        parseInt(e.target.value) || 2,
                      )
                    }
                    className="h-9"
                  />
                </div>
              </div>
            </div>
          )}

          {selectedModel === "SVR" && (
            <div className="space-y-3 rounded-lg border p-4">
              <p className="text-sm font-medium">SVR Hyperparameters</p>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">
                    Kernel
                  </label>
                  <Input
                    value={svrParams.kernel}
                    onChange={(e) => updateSVRParam("kernel", e.target.value)}
                    className="h-9"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">C</label>
                  <Input
                    type="number"
                    value={svrParams.C}
                    onChange={(e) =>
                      updateSVRParam("C", parseFloat(e.target.value) || 1.0)
                    }
                    className="h-9"
                  />
                </div>
              </div>
            </div>
          )}

          <div className="mt-2">
            <Button
              className="w-full"
              onClick={handleStartTraining}
              disabled={isTraining || !selectedJobId}
            >
              Start Training
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Training history */}
      <Card>
        <CardHeader>
          <CardTitle>Training History</CardTitle>
          <CardDescription>View previous training runs</CardDescription>
        </CardHeader>
        <CardContent>
          {trainingHistory.length === 0 ? (
            <div className="text-sm text-muted-foreground">
              No training runs yet
            </div>
          ) : (
            <ScrollArea className="h-112 pr-3">
              <div className="space-y-2">
                {trainingHistory.map((h) => (
                  <div
                    key={h.job_id}
                    className={`rounded-lg border p-2 ${
                      lastEvaluationJobId === h.job_id
                        ? "border-emerald-500/40 bg-emerald-500/5"
                        : ""
                    }`}
                  >
                    <div className="text-sm">{h.job_id}</div>
                    <div className="text-xs text-muted-foreground">
                      {h.created_at ? new Date(h.created_at).toLocaleString() : ""}
                    </div>
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <span>{h.model_type}</span>
                      <span>— {h.status}</span>
                      {lastEvaluationJobId === h.job_id && (
                        <Badge
                          variant="outline"
                          className="border-emerald-500/30 text-emerald-600"
                        >
                          Evaluation complete
                        </Badge>
                      )}
                    </div>
                    {typeof h.training_time_secs === "number" && (
                      <div className="text-xs text-muted-foreground">
                        {h.training_time_secs.toFixed(2)}s
                      </div>
                    )}
                    {lastEvaluationJobId === h.job_id &&
                      lastEvaluationResult && (
                        <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-4">
                          <div>
                            <span className="text-muted-foreground">RMSE</span>
                            <div className="font-medium">
                              {lastEvaluationResult.rmse.toFixed(3)}
                            </div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">MAE</span>
                            <div className="font-medium">
                              {lastEvaluationResult.mae.toFixed(3)}
                            </div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">MAPE</span>
                            <div className="font-medium">
                              {lastEvaluationResult.mape.toFixed(3)}%
                            </div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">R²</span>
                            <div className="font-medium">
                              {lastEvaluationResult.r2.toFixed(3)}
                            </div>
                          </div>
                        </div>
                      )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
