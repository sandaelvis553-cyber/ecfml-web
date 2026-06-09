import type { Metadata } from "next";
import { Alert, AlertDescription } from "@/components/ui/alert";
import TrainManager from "@/components/dashboard/TrainManager";
import { mlFetch } from "@/lib/api-client";
import type { ModelRunApiRecord, PreprocessJobApiRecord } from "@/lib/evaluation-utils";

export const metadata: Metadata = {
  title: "Model Training",
  description: "Train Random Forest and SVR models on preprocessed data.",
};

export default async function TrainPage() {
  let initialJobs: PreprocessJobApiRecord[] = [];
  let initialHistory: ModelRunApiRecord[] = [];
  let initialError: string | null = null;

  try {
    const [jobsRes, modelsRes] = await Promise.all([
      mlFetch<PreprocessJobApiRecord[]>("/api/v1/preprocessing/jobs"),
      mlFetch<ModelRunApiRecord[] | { data: ModelRunApiRecord[] }>("/api/v1/models"),
    ]);

    initialJobs = jobsRes;

    const modelsData = Array.isArray((modelsRes as { data?: ModelRunApiRecord[] })?.data)
      ? (modelsRes as { data: ModelRunApiRecord[] }).data
      : Array.isArray(modelsRes)
      ? modelsRes
      : [];
    initialHistory = modelsData;
  } catch (error) {
    initialError = error instanceof Error ? error.message : "Failed to load page data";
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Model Training</h1>
        <p className="mt-1 text-muted-foreground">
          Train Random Forest and SVR models. ANN and LSTM have been removed in
          v3.0 — use the LangGraph Agent on the Forecast page for LLM-based
          forecasting.
        </p>
      </div>

      <Alert>
        <AlertDescription>
          <strong>v3.0 Note:</strong> Only Random Forest and SVR are available
          for training. The LangGraph Agent engine does not require training —
          access it from the{" "}
          <a
            href="/dashboard/forecast"
            className="font-medium underline underline-offset-4"
          >
            Forecast
          </a>{" "}
          page.
        </AlertDescription>
      </Alert>

      {initialError && (
        <Alert variant="destructive">
          <AlertDescription>
            Failed to load initial training data: {initialError}
          </AlertDescription>
        </Alert>
      )}

      <TrainManager
        initialJobs={initialJobs}
        initialHistory={initialHistory}
        initialError={initialError}
      />
    </div>
  );
}

