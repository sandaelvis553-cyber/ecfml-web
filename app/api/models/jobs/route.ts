import { mlFetch } from "@/lib/api-client";
import {
  wrapSuccess,
  parseStatusFromErrorMessage,
  getOrGenerateCorrelationId,
  buildErrorResponse,
} from "@/lib/api-response";
import { NextResponse } from "next/server";

interface BackendModelRunRecord {
  id?: string;
  job_id?: string;
  modelType?: string;
  model_type?: string;
  preprocessJobId?: string;
  preprocess_job_id?: string;
  status?: string;
  modelFilePath?: string | null;
  model_file_path?: string | null;
  scalerFilePath?: string | null;
  scaler_file_path?: string | null;
  trainingTimeSecs?: number | null;
  training_time_secs?: number | null;
  error?: string | null;
  createdAt?: string;
  created_at?: string;
}

function normalizeModelRun(record: BackendModelRunRecord) {
  return {
    job_id: record.job_id ?? record.id ?? "",
    model_type: record.model_type ?? record.modelType ?? "RANDOM_FOREST",
    preprocess_job_id: record.preprocess_job_id ?? record.preprocessJobId ?? "",
    status: record.status ?? "PENDING",
    model_file_path: record.model_file_path ?? record.modelFilePath ?? null,
    scaler_file_path: record.scaler_file_path ?? record.scalerFilePath ?? null,
    training_time_secs:
      record.training_time_secs ?? record.trainingTimeSecs ?? null,
    error: record.error ?? null,
    created_at:
      record.created_at ?? record.createdAt ?? new Date().toISOString(),
  };
}

/**
 * GET /api/models/jobs — List model training runs.
 */
export async function GET(request: Request) {
  const correlationId = getOrGenerateCorrelationId(request);
  try {
    const data = await mlFetch<
      BackendModelRunRecord[] | { data?: BackendModelRunRecord[] }
    >("/api/v1/models", {
      method: "GET",
      headers: { "X-Correlation-ID": correlationId },
    });

    const records = Array.isArray(
      (data as { data?: BackendModelRunRecord[] })?.data,
    )
      ? ((data as { data?: BackendModelRunRecord[] }).data ?? [])
      : Array.isArray(data)
        ? data
        : [];

    const body = wrapSuccess(records.map(normalizeModelRun));
    return NextResponse.json(body, {
      status: 200,
      headers: { "X-Correlation-ID": correlationId },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    const status = parseStatusFromErrorMessage(message);
    console.error(
      `[models:jobs][GET] correlation_id=${correlationId} error=`,
      message,
    );
    const errBody = buildErrorResponse(
      "BACKEND_ERROR",
      message,
      status,
      correlationId,
    );
    return NextResponse.json(errBody, {
      status,
      headers: { "X-Correlation-ID": correlationId },
    });
  }
}
