import { mlFetchRaw } from "@/lib/api-client";
import {
  wrapSuccess,
  parseStatusFromErrorMessage,
  getOrGenerateCorrelationId,
  buildErrorResponse,
} from "@/lib/api-response";
import { NextResponse } from "next/server";

/**
 * POST /api/preprocess — Start preprocessing pipeline
 */
export async function POST(request: Request) {
  const correlationId = getOrGenerateCorrelationId(request);
  try {
    const body = await request.json();

    const jobId =
      (body?.job_id as string | undefined) ??
      (body?.jobId as string | undefined);
    const datasetId =
      (body?.dataset_id as string | undefined) ??
      (body?.datasetId as string | undefined);
    const datasetUrl =
      (body?.dataset_url as string | undefined) ??
      (body?.datasetUrl as string | undefined);
    const weatherUrl =
      (body?.weather_url as string | undefined) ??
      (body?.weatherUrl as string | undefined);
    const splits = (body?.splits as unknown) ?? body?.splits;

    const payload: Record<string, unknown> = {
      ...(jobId ? { job_id: jobId } : {}),
      ...(datasetId ? { dataset_id: datasetId } : {}),
      ...(datasetUrl ? { dataset_url: datasetUrl } : {}),
      ...(weatherUrl ? { weather_url: weatherUrl } : {}),
      ...(splits ? { splits } : {}),
    };

    const res = await mlFetchRaw("/api/v1/preprocessing/run", {
      method: "POST",
      headers: { "X-Correlation-ID": correlationId },
      body: JSON.stringify(payload),
    });
    const responsePayload = await res.json().catch(() => null);
    const wrapped = wrapSuccess(responsePayload);
    const headers: Record<string, string> = {
      "X-Correlation-ID": correlationId,
    };
    const location = res.headers.get("Location");
    if (location) headers["Location"] = location;
    return NextResponse.json(wrapped, { status: res.status || 202, headers });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    const status = parseStatusFromErrorMessage(message);
    console.error(
      `[preprocess][POST] correlation_id=${correlationId} error=`,
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
