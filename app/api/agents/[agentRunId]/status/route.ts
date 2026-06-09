import { mlFetch } from "@/lib/api-client";
import {
  wrapSuccess,
  parseStatusFromErrorMessage,
  getOrGenerateCorrelationId,
  buildErrorResponse,
} from "@/lib/api-response";
import { NextResponse } from "next/server";

interface AgentStatusResponse {
  status: "PENDING" | "COMPLETE";
  result?: Record<string, unknown> | null;
}

/**
 * GET /api/agents/[agentRunId]/status — Proxy agent status queries to FastAPI.
 */
export async function GET(
  request: Request,
  ctx: { params: Promise<{ agentRunId: string }> }
) {
  const correlationId = getOrGenerateCorrelationId(request);

  try {
    const { agentRunId } = await ctx.params;

    const data = await mlFetch<AgentStatusResponse>(
      `/api/v1/agents/${agentRunId}/status`,
      {
        method: "GET",
        headers: { "X-Correlation-ID": correlationId },
      }
    );

    const body = wrapSuccess(data);
    return NextResponse.json(body, {
      status: 200,
      headers: { "X-Correlation-ID": correlationId },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    const status = parseStatusFromErrorMessage(message);
    console.error(`[agent:status][GET] correlation_id=${correlationId} error=`, message);
    
    const errBody = buildErrorResponse(
      "BACKEND_ERROR",
      message,
      status,
      correlationId
    );
    return NextResponse.json(errBody, {
      status,
      headers: { "X-Correlation-ID": correlationId },
    });
  }
}
