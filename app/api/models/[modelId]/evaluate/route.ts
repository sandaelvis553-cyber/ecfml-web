import { mlFetchRaw } from "@/lib/api-client";
import {
  wrapSuccess,
  parseStatusFromErrorMessage,
  getOrGenerateCorrelationId,
  buildErrorResponse,
} from "@/lib/api-response";
import { NextResponse } from "next/server";

/**
 * POST /api/models/[modelId]/evaluate — Proxy model evaluation requests to FastAPI.
 */
export async function POST(
  request: Request,
  ctx: RouteContext<"/api/models/[modelId]/evaluate">,
) {
  const correlationId = getOrGenerateCorrelationId(request);

  try {
    const { modelId } = await ctx.params;
    const body = await request.json().catch(() => ({}));

    const modelRunId =
      (body?.model_run_id as string | undefined) ??
      (body?.modelRunId as string | undefined) ??
      modelId;
    const modelFilePath =
      (body?.model_file_path as string | undefined) ??
      (body?.modelFilePath as string | undefined);
    const processedFilePath =
      (body?.processed_file_path as string | undefined) ??
      (body?.processedFilePath as string | undefined);

    const payload: Record<string, unknown> = {
      model_run_id: modelRunId,
      ...(modelFilePath ? { model_file_path: modelFilePath } : {}),
      ...(processedFilePath ? { processed_file_path: processedFilePath } : {}),
    };

    const res = await mlFetchRaw(`/api/v1/models/${modelId}/evaluate`, {
      method: "POST",
      headers: { "X-Correlation-ID": correlationId },
      body: JSON.stringify(payload),
    });

    const responsePayload = await res.json().catch(() => null);
    const wrapped = wrapSuccess(responsePayload);

    return NextResponse.json(wrapped, {
      status: res.status || 200,
      headers: { "X-Correlation-ID": correlationId },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    const status = parseStatusFromErrorMessage(message);
    console.error(`[evaluate:route] correlation_id=${correlationId} error=`, {
      message,
      status,
      errorFull: error,
    });
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
