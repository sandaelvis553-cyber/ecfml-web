import type { Metadata } from "next";
import EvaluateManager from "@/components/dashboard/EvaluateManager";

export const metadata: Metadata = {
  title: "Model Evaluation",
  description:
    "Compare RF, SVR, and LangGraph Agent forecasting performance — RMSE, MAE, MAPE, R².",
};

export default function EvaluatePage() {
  return <EvaluateManager />;
}
