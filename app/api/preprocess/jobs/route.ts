import { mlFetch } from "@/lib/api-client";
import {
  wrapSuccess,
  parseStatusFromErrorMessage,
  getOrGenerateCorrelationId,
  buildErrorResponse,
} from "@/lib/api-response";
import { NextResponse } from "next/server";

/**
 * GET /api/preprocess/jobs — List preprocessing jobs (proxy to FastAPI)
 */
export async function GET(request: Request) {
  const correlationId = getOrGenerateCorrelationId(request);
  try {
    const data = await mlFetch("/api/v1/preprocessing/jobs", {
      method: "GET",
      headers: { "X-Correlation-ID": correlationId },
    });
    const body = wrapSuccess(data);
    return NextResponse.json(body, {
      status: 200,
      headers: { "X-Correlation-ID": correlationId },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    const status = parseStatusFromErrorMessage(message);
    console.error(
      `[preprocess:jobs][GET] correlation_id=${correlationId} error=`,
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
