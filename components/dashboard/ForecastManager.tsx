"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { AgentProgressPanel } from "@/components/agent/AgentProgressPanel";
import { AgentReasoningPanel } from "@/components/agent/AgentReasoningPanel";
import { useAgentStream } from "@/hooks/use-agent-stream";
import { extractApiErrorMessage, unwrapApiData } from "@/lib/api-contract";
import { normalizePreprocessJobRecord, normalizeModelRunRecord } from "@/lib/evaluation-utils";
import type { ModelRunApiRecord, PreprocessJobApiRecord } from "@/lib/evaluation-utils";
import type { JobStatus, ModelType } from "@/types/model";
import type { Resolution, ForecastEngine } from "@/types/forecast";

interface PreprocessJobRecord {
  job_id: string;
  status: JobStatus;
  created_at: string;
  processed_file_path?: string | null;
}

interface ModelRunRecord {
  job_id: string;
  status: JobStatus;
  model_type: ModelType;
  preprocess_job_id: string;
  model_file_path?: string | null;
}

interface ForecastResult {
  predictions: { timestamp: string; value: number }[];
  reasoning: string | null;
  confidence: string | null;
}

interface ForecastManagerProps {
  initialJobs?: PreprocessJobApiRecord[];
  initialModels?: ModelRunApiRecord[];
  initialError?: string | null;
}

export default function ForecastManager({
  initialJobs = [],
  initialModels = [],
  initialError = null,
}: ForecastManagerProps) {
  // Normalize initial props
  const jobs = useMemo(() => {
    return initialJobs
      .map((job) => {
        const normalized = normalizePreprocessJobRecord(job);
        return {
          job_id: normalized.job_id,
          status: normalized.status as JobStatus,
          created_at: normalized.created_at,
          processed_file_path: normalized.processed_file_path,
        };
      })
      .filter((j) => j.status === "COMPLETE");
  }, [initialJobs]);

  const models = useMemo(() => {
    return initialModels
      .map((model) => {
        const normalized = normalizeModelRunRecord(model);
        return {
          job_id: normalized.job_id,
          status: normalized.status as JobStatus,
          model_type: normalized.model_type as ModelType,
          preprocess_job_id: normalized.preprocess_job_id,
          model_file_path: normalized.model_file_path,
        };
      })
      .filter((m) => m.status === "COMPLETE");
  }, [initialModels]);

  // Form State
  const [engine, setEngine] = useState<ForecastEngine>("AGENT");
  const [selectedJobId, setSelectedJobId] = useState<string | null>(() => {
    return jobs.length > 0 ? jobs[0].job_id : null;
  });
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [modelOverride, setModelOverride] = useState<string>("openai/gpt-5.4");
  const [startDate, setStartDate] = useState<string>(() => {
    const today = new Date();
    return today.toISOString().split("T")[0];
  });
  const [horizonDays, setHorizonDays] = useState<number>(7);
  const [resolution, setResolution] = useState<Resolution>("DAILY");

  // Flow State
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeAgentRunId, setActiveAgentRunId] = useState<string | null>(null);
  const [forecastResult, setForecastResult] = useState<ForecastResult | null>(null);

  // Filter models based on selected engine and preprocess job
  const filteredModels = useMemo(() => {
    if (!selectedJobId) return [];
    const targetModelType = engine === "RF" ? "RANDOM_FOREST" : "SVR";
    return models.filter(
      (m) => m.preprocess_job_id === selectedJobId && m.model_type === targetModelType
    );
  }, [models, selectedJobId, engine]);

  // Reset model selection when preprocess job or engine changes
  useEffect(() => {
    if (filteredModels.length > 0) {
      setSelectedModelId(filteredModels[0].job_id);
    } else {
      setSelectedModelId(null);
    }
  }, [filteredModels]);

  // Trigger error toast on mount if server fetch failed
  useEffect(() => {
    if (initialError) {
      toast.error(initialError);
    }
  }, [initialError]);

  // Subscribe to SSE updates for active agent runs
  const { events, done } = useAgentStream(activeAgentRunId);

  // Handle agent run completion
  useEffect(() => {
    if (activeAgentRunId && done) {
      const hasError = events.some((e) => e.type === "error");
      if (hasError) {
        const errorMsg = events.find((e) => e.type === "error")?.error;
        toast.error(errorMsg ?? "Agent run failed");
        setActiveAgentRunId(null);
        return;
      }

      const fetchAgentStatus = async () => {
        try {
          const res = await fetch(`/api/agents/${activeAgentRunId}/status`);
          const payload = await res.json().catch(() => null);
          if (!res.ok) {
            throw new Error(extractApiErrorMessage(payload, "Failed to fetch agent status"));
          }
          const data = unwrapApiData<{ status: string; result: any }>(payload);
          if (data && data.status === "COMPLETE" && data.result) {
            setForecastResult({
              predictions: data.result.predictions ?? [],
              reasoning: data.result.agent_reasoning ?? data.result.reasoning ?? null,
              confidence: data.result.confidence ?? null,
            });
            toast.success("Agent forecast completed successfully");
          } else {
            toast.error("Agent failed to complete the forecast run");
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : "Failed to load agent results";
          toast.error(message);
        } finally {
          setActiveAgentRunId(null);
        }
      };

      fetchAgentStatus();
    }
  }, [activeAgentRunId, done, events]);

  const handleGenerateForecast = useCallback(async () => {
    if (!selectedJobId) {
      toast.error("Please select a preprocessing job");
      return;
    }

    const job = jobs.find((j) => j.job_id === selectedJobId);
    if (!job || !job.processed_file_path) {
      toast.error("Selected preprocessing job lacks a valid data path");
      return;
    }

    if (engine !== "AGENT" && !selectedModelId) {
      toast.error(`Please select a trained ${engine} model`);
      return;
    }

    try {
      setIsSubmitting(true);
      setForecastResult(null);
      setActiveAgentRunId(null);

      const payload = {
        engine,
        preprocessJobId: selectedJobId,
        modelRunId: engine !== "AGENT" ? selectedModelId : undefined,
        startDate,
        horizonDays,
        resolution,
        modelOverride: engine === "AGENT" ? modelOverride : undefined,
        processedFilePath: job.processed_file_path,
      };

      const res = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const body = await res.json().catch(() => null);

      if (!res.ok) {
        throw new Error(extractApiErrorMessage(body, "Failed to start forecast"));
      }

      const data = unwrapApiData<any>(body);

      if (engine === "AGENT") {
        const agentRunId = data?.agentRunId ?? data?.agent_run_id;
        if (!agentRunId) {
          throw new Error("Backend did not return an agent run identifier");
        }
        setActiveAgentRunId(agentRunId);
        toast.success("Agent forecast process started");
      } else {
        // RF / SVR results returned directly
        setForecastResult({
          predictions: data?.predictions ?? [],
          reasoning: data?.agent_reasoning ?? data?.reasoning ?? null,
          confidence: data?.confidence ?? null,
        });
        toast.success(`${engine} forecast completed`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to generate forecast";
      toast.error(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [engine, selectedJobId, selectedModelId, startDate, horizonDays, resolution, modelOverride, jobs]);

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      {/* Configuration column */}
      <Card className="lg:col-span-1">
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
          <CardDescription>Configure engine parameters and target scope.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Engine Selector */}
          <div className="space-y-2">
            <Label>Engine</Label>
            <div className="flex gap-2">
              {(["RF", "SVR", "AGENT"] as ForecastEngine[]).map((e) => (
                <Badge
                  key={e}
                  variant={engine === e ? "default" : "secondary"}
                  className="cursor-pointer px-4 py-1.5 transition-all"
                  onClick={() => setEngine(e)}
                >
                  {e === "AGENT" ? "Agent (LangGraph)" : e}
                </Badge>
              ))}
            </div>
          </div>

          {/* Preprocess Job */}
          <div className="space-y-2">
            <Label htmlFor="job-select">Preprocessing Scope</Label>
            <Select value={selectedJobId ?? undefined} onValueChange={setSelectedJobId}>
              <SelectTrigger id="job-select">
                <SelectValue placeholder="Select completed preprocessed data" />
              </SelectTrigger>
              <SelectContent>
                {jobs.length === 0 ? (
                  <div className="px-2 py-1.5 text-sm text-muted-foreground">
                    No preprocessed datasets available
                  </div>
                ) : (
                  jobs.map((j) => (
                    <SelectItem key={j.job_id} value={j.job_id}>
                      {j.job_id.slice(0, 15)}... ({new Date(j.created_at).toLocaleDateString()})
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>

          {/* Model Selection (RF/SVR only) */}
          {engine !== "AGENT" && (
            <div className="space-y-2">
              <Label htmlFor="model-select">Trained Model</Label>
              <Select value={selectedModelId ?? undefined} onValueChange={setSelectedModelId}>
                <SelectTrigger id="model-select" disabled={filteredModels.length === 0}>
                  <SelectValue placeholder={filteredModels.length === 0 ? "No trained models found" : "Select trained model run"} />
                </SelectTrigger>
                <SelectContent>
                  {filteredModels.map((m) => (
                    <SelectItem key={m.job_id} value={m.job_id}>
                      {m.job_id.slice(0, 15)}... ({m.model_type})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Model Override (Agent only) */}
          {engine === "AGENT" && (
            <div className="space-y-2">
              <Label htmlFor="model-override">Agent LLM Model</Label>
              <Select value={modelOverride} onValueChange={setModelOverride}>
                <SelectTrigger id="model-override">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai/gpt-5.4">GPT-5.4 (Default)</SelectItem>
                  <SelectItem value="google/gemini-3.1-pro-preview">Gemini 3.1 Pro</SelectItem>
                  <SelectItem value="anthropic/claude-sonnet-4-6">Claude 4.6 Sonnet</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Start Date */}
          <div className="space-y-2">
            <Label htmlFor="start-date">Start Date</Label>
            <Input
              id="start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>

          {/* Horizon & Resolution */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label htmlFor="horizon">Horizon (Days)</Label>
              <Input
                id="horizon"
                type="number"
                min={1}
                max={30}
                value={horizonDays}
                onChange={(e) => setHorizonDays(Math.max(1, parseInt(e.target.value) || 1))}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="resolution">Resolution</Label>
              <Select value={resolution} onValueChange={(val) => setResolution(val as Resolution)}>
                <SelectTrigger id="resolution">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="HOURLY">Hourly</SelectItem>
                  <SelectItem value="DAILY">Daily</SelectItem>
                  <SelectItem value="WEEKLY">Weekly</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Generate Button */}
          <Button
            className="w-full mt-2"
            onClick={handleGenerateForecast}
            disabled={isSubmitting || !!activeAgentRunId || !selectedJobId}
          >
            {isSubmitting ? "Generating..." : "Generate Forecast"}
          </Button>
        </CardContent>
      </Card>

      {/* Output column */}
      <div className="lg:col-span-2 space-y-6">
        {/* Streaming output */}
        {activeAgentRunId && (
          <AgentProgressPanel events={events} done={done} />
        )}

        {/* Forecast chart & Reasoning */}
        {forecastResult ? (
          <>
            <ForecastChart
              data={forecastResult.predictions}
              title={`${engine} Forecast Results`}
              description={`Forecast starting from ${new Date(startDate).toLocaleDateString()} for ${horizonDays} days`}
            />

            {forecastResult.reasoning && (
              <AgentReasoningPanel
                reasoning={forecastResult.reasoning}
                confidence={forecastResult.confidence ?? undefined}
              />
            )}
          </>
        ) : (
          !activeAgentRunId && (
            <div className="flex h-96 flex-col items-center justify-center rounded-xl border border-dashed bg-muted/20 p-8 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                🔮
              </div>
              <h3 className="mt-4 text-sm font-semibold">No forecast generated</h3>
              <p className="mt-1 text-sm text-muted-foreground max-w-sm">
                Configure your forecasting options on the left and click Generate Forecast to execute models.
              </p>
            </div>
          )
        )}
      </div>
    </div>
  );
}
