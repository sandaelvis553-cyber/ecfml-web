import type { Metadata } from 'next'
import ForecastManager from '@/components/dashboard/ForecastManager'
import { mlFetch } from '@/lib/api-client'
import type { ModelRunApiRecord, PreprocessJobApiRecord } from '@/lib/evaluation-utils'

export const metadata: Metadata = {
  title: 'Forecast',
  description: 'Generate forecasts using RF, SVR, or LangGraph Agent.',
}

export default async function ForecastPage() {
  let initialJobs: PreprocessJobApiRecord[] = []
  let initialModels: ModelRunApiRecord[] = []
  let initialError: string | null = null

  try {
    const [jobsRes, modelsRes] = await Promise.all([
      mlFetch<PreprocessJobApiRecord[]>('/api/v1/preprocessing/jobs'),
      mlFetch<ModelRunApiRecord[] | { data: ModelRunApiRecord[] }>('/api/v1/models'),
    ])

    initialJobs = jobsRes

    const modelsData = Array.isArray((modelsRes as { data?: ModelRunApiRecord[] })?.data)
      ? (modelsRes as { data: ModelRunApiRecord[] }).data
      : Array.isArray(modelsRes)
      ? modelsRes
      : []
    initialModels = modelsData
  } catch (error) {
    initialError = error instanceof Error ? error.message : 'Failed to load page data'
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Forecast</h1>
        <p className="mt-1 text-muted-foreground">
          Generate short- and medium-term electricity consumption forecasts.
        </p>
      </div>

      <ForecastManager
        initialJobs={initialJobs}
        initialModels={initialModels}
        initialError={initialError}
      />
    </div>
  )
}

