"use client";
import { useEffect, useState } from "react";
import type { JobStatus } from "@/types/model";
import { extractApiErrorMessage, unwrapApiData } from "@/lib/api-contract";

interface JobState {
  status: JobStatus;
  progress?: number;
  error?: string;
  model_run_id?: string;
  modelRunId?: string;
  model_file_path?: string;
  modelFilePath?: string;
  processed_file_path?: string;
  processedFilePath?: string;
  evaluation_result?: unknown;
  evaluationResult?: unknown;
}

/**
 * Polls the RF/SVR training job status endpoint at a regular interval.
 * Stops polling when the job reaches a terminal state (COMPLETE or FAILED).
 */
export function useJobPoller(jobId: string | null, intervalMs = 3000) {
  const [state, setState] = useState<JobState>({ status: "PENDING" });

  useEffect(() => {
    if (!jobId) return;

    let active = true;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const res = await fetch(`/api/models/jobs/${jobId}/status`);
        const payload = await res.json().catch(() => null);

        if (!res.ok) {
          throw new Error(
            extractApiErrorMessage(payload, "Failed to poll job status"),
          );
        }

        const data = unwrapApiData<JobState | null>(payload);
        if (!data) {
          throw new Error("Failed to poll job status");
        }
        if (!active) return;

        setState(data);

        if (data.status !== "COMPLETE" && data.status !== "FAILED") {
          timeoutId = setTimeout(() => {
            void poll();
          }, intervalMs);
        }
      } catch {
        if (!active) return;
        setState({ status: "FAILED", error: "Failed to poll job status" });
      }
    };

    void poll();

    return () => {
      active = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [jobId, intervalMs]);

  return state;
}
