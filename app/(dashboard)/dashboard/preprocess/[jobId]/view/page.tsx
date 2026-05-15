import type { Metadata } from "next";
import { mlFetch } from "@/lib/api-client";
import { CorrelationHeatmap } from "@/components/charts/CorrelationHeatmap";
import { SeasonalDecomposition } from "@/components/charts/SeasonalDecomposition";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";

export async function generateMetadata({
  params,
}: {
  params: { jobId: string };
}): Promise<Metadata> {
  return { title: `Preprocess ${params.jobId}` };
}

export default async function PreprocessViewPage({
  params,
}: {
  params: { jobId: string };
}) {
  const { jobId } = params;
  let data: any = null;
  try {
    data = await mlFetch(`/api/v1/preprocessing/${jobId}/status`);
  } catch (err) {
    data = null;
  }

  const payload = data ?? null;
  const summary = payload?.result_summary ?? null;
  const eda = payload?.eda_charts ?? null;
  const processedPath = payload?.processed_file_path ?? null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Preprocessing Results — {jobId}</h1>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Result Summary</CardTitle>
            <CardDescription>
              Basic stats about the processed dataset
            </CardDescription>
          </CardHeader>
          <CardContent>
            {summary ? (
              <div className="space-y-2">
                <div>
                  Feature columns: {(summary.feature_columns ?? []).length}
                </div>
                <div>Train rows: {summary.train_rows ?? "-"}</div>
                <div>Val rows: {summary.val_rows ?? "-"}</div>
                <div>Test rows: {summary.test_rows ?? "-"}</div>
                {processedPath ? (
                  <div>
                    <a
                      href={processedPath}
                      className="text-blue-600 underline"
                      target="_blank"
                      rel="noreferrer"
                    >
                      Download processed file
                    </a>
                  </div>
                ) : null}
              </div>
            ) : (
              <div>No summary available</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>EDA Charts</CardTitle>
            <CardDescription>
              Correlation matrix and seasonal decomposition
            </CardDescription>
          </CardHeader>
          <CardContent>
            {eda ? (
              <div className="space-y-6">
                <CorrelationHeatmap
                  matrix={eda.correlation_matrix ?? eda.correlationMatrix ?? []}
                  labels={eda.feature_names ?? eda.featureNames ?? []}
                />
                {eda.seasonal_decomposition || eda.seasonalDecomposition ? (
                  <SeasonalDecomposition
                    data={
                      eda.seasonal_decomposition ?? eda.seasonalDecomposition
                    }
                  />
                ) : null}
              </div>
            ) : (
              <div>No EDA available</div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
