# ECFML — System Design Document v2.0
**Electricity Consumption Forecasting using Machine Learning**
*A Case Study of the North West Region of Cameroon*

| Field | Value |
|---|---|
| Author | Sanda Elvis Toge (UBa22S1297) |
| Supervisor | Prof. Fautso Gaetan |
| Version | 2.0 — Hybrid RF/SVR + LangGraph Architecture |
| Supersedes | System Design Document v1.0 (ANN/LSTM/RF/SVR stack) |
| Date | May 2026 |
| Stack | Next.js 16 · FastAPI · LangGraph v1.1 · Prisma 7 · Clerk v7 · Recharts · Vercel AI SDK v6 · Neon Postgres · Vercel · Render |

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [System Architecture](#3-system-architecture)
4. [Component Breakdown](#4-component-breakdown)
5. [Database Schema (Prisma 7)](#5-database-schema-prisma-7)
6. [API Design (FastAPI)](#6-api-design-fastapi)
7. [ML Pipeline Design](#7-ml-pipeline-design)
8. [LangGraph Agent Design](#8-langgraph-agent-design)
9. [UI/UX Wireframes & User Flows](#9-uiux-wireframes--user-flows)
10. [Data Flow Diagrams](#10-data-flow-diagrams)
11. [Deployment Architecture](#11-deployment-architecture)
12. [Environment Variables](#12-environment-variables)
13. [Error Handling Strategy](#13-error-handling-strategy)

---

## 1. System Overview

### 1.1 What Changed from v1.0

| Component | v1.0 | v2.0 |
|---|---|---|
| ML Models | ANN + LSTM + RF + SVR | **RF + SVR only** (owned) |
| ANN | TensorFlow/Keras | **Removed** |
| LSTM | TensorFlow/Keras, GPU | **Removed** |
| Third forecasting engine | Vercel AI SDK (low priority) | **LangGraph v1.1 agent** (core feature) |
| Agent framework | None | **LangGraph v1.1** — StateGraph, MemorySaver, SSE streaming |
| LLM models | AI SDK only | **GPT-5.4 (default), Gemini 3.1 Pro Preview, Claude Sonnet 4.6** |
| GPU required | Yes (LSTM) | **No** |
| Difficulty | 7/10 | **4/10** |

### 1.2 Architecture Summary

```
Browser
  │
  ▼
Next.js 16 (Vercel)           ←── Clerk v7 (Auth)
  │  │                             Uploadthing (Files)
  │  └── /api/* routes ─────────► FastAPI (Render)
  │       (JWT proxy)                │         │
  │                          ┌───────┘         └────────┐
  ▼                          ▼                          ▼
Prisma 7 ──────────► Neon Postgres       Traditional ML   LangGraph Agent
                                         (RF + SVR)        (GPT/Gemini/Claude)
```

### 1.3 The Three Forecasting Engines

| Engine | Tech | Training | Notes |
|---|---|---|---|
| Random Forest | scikit-learn | Minutes, CPU | Explainable; feature importance |
| SVR | scikit-learn | Minutes, CPU | Strong baseline |
| LangGraph Agent | LangGraph v1.1 + LLM | None | Context-driven; model-swappable |

---

## 2. Repository Structure

Two independent repositories (or two root folders in one repo — no monorepo tooling).

### 2.1 Frontend — `ecfml-web/`

```
ecfml-web/
├── src/
│   ├── app/
│   │   ├── layout.tsx                    # Root layout — ClerkProvider, fonts
│   │   ├── page.tsx                      # Landing → redirect to /dashboard
│   │   ├── (auth)/
│   │   │   ├── sign-in/[[...sign-in]]/page.tsx
│   │   │   └── sign-up/[[...sign-up]]/page.tsx
│   │   ├── (dashboard)/
│   │   │   ├── layout.tsx                # Sidebar + Header shell
│   │   │   ├── dashboard/page.tsx        # Overview cards
│   │   │   ├── dashboard/data/page.tsx   # Upload + preview
│   │   │   ├── dashboard/preprocess/page.tsx
│   │   │   ├── dashboard/train/page.tsx  # RF + SVR only
│   │   │   ├── dashboard/evaluate/page.tsx  # 3-way comparison
│   │   │   ├── dashboard/forecast/page.tsx  # Engine selector + charts
│   │   │   └── dashboard/ai/page.tsx     # AI chat (Low Priority)
│   │   └── api/
│   │       ├── datasets/
│   │       │   ├── route.ts
│   │       │   └── [id]/
│   │       │       ├── validate/route.ts
│   │       │       └── preview/route.ts
│   │       ├── preprocess/
│   │       │   ├── route.ts
│   │       │   └── [jobId]/status/route.ts
│   │       ├── models/
│   │       │   ├── train/route.ts        # RF + SVR only
│   │       │   ├── tune/route.ts
│   │       │   └── [modelId]/evaluate/route.ts
│   │       ├── forecast/
│   │       │   ├── route.ts              # POST — all 3 engines
│   │       │   └── stream/[agentRunId]/route.ts  # SSE proxy
│   │       └── ai/chat/route.ts          # Vercel AI SDK v6 (Low Pri)
│   ├── components/
│   │   ├── ui/                           # shadcn/ui base components
│   │   ├── charts/
│   │   │   ├── ForecastChart.tsx         # AreaChart
│   │   │   ├── ActualVsPredicted.tsx     # LineChart overlay
│   │   │   ├── FeatureImportanceChart.tsx # BarChart (RF)
│   │   │   ├── CorrelationHeatmap.tsx
│   │   │   └── SeasonalDecomposition.tsx
│   │   ├── dashboard/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── MetricCard.tsx
│   │   │   └── ModelComparisonTable.tsx  # RF vs SVR vs Agent
│   │   ├── agent/
│   │   │   ├── AgentProgressPanel.tsx    # SSE node status display
│   │   │   └── AgentReasoningPanel.tsx   # Collapsible reasoning text
│   │   ├── data/
│   │   │   ├── DatasetUploader.tsx
│   │   │   └── DatasetPreviewTable.tsx
│   │   ├── training/
│   │   │   ├── ModelConfigForm.tsx       # RF + SVR config only
│   │   │   └── TrainingProgressBar.tsx
│   │   └── ai/
│   │       └── ChatPanel.tsx             # Vercel AI SDK v6 (Low Pri)
│   ├── lib/
│   │   ├── prisma.ts                     # Prisma 7 singleton
│   │   ├── api-client.ts                 # mlFetch() — JWT-injecting fetch wrapper
│   │   ├── uploadthing.ts
│   │   └── utils.ts
│   ├── hooks/
│   │   ├── useJobPoller.ts               # Poll /status until COMPLETE/FAILED
│   │   └── useAgentStream.ts             # Subscribe to SSE agent stream
│   └── types/
│       ├── dataset.ts
│       ├── model.ts
│       ├── agent.ts                      # AgentRun, NodeEvent, ForecastResult
│       └── forecast.ts
├── prisma/
│   ├── schema.prisma
│   └── migrations/
├── prisma.config.ts                      # Prisma 7 DB connection
├── proxy.ts                              # Clerk clerkMiddleware() — NOT middleware.ts
├── next.config.ts
├── tailwind.config.ts
├── .env.local
├── .env.example
└── package.json
```

### 2.2 ML + Agent Backend — `ecfml-api/`

```
ecfml-api/
├── app/
│   ├── __init__.py
│   ├── main.py                           # FastAPI app + lifespan (graph compile)
│   ├── config.py                         # Pydantic BaseSettings
│   ├── dependencies.py                   # JWT validation, shared deps
│   ├── routers/
│   │   ├── datasets.py                   # /api/v1/datasets/*
│   │   ├── preprocessing.py              # /api/v1/preprocessing/*
│   │   ├── models.py                     # /api/v1/models/* (RF + SVR)
│   │   └── forecast.py                   # /api/v1/forecast + /stream/{id}
│   ├── schemas/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── model.py                      # RF + SVR only
│   │   ├── agent.py                      # AgentState, ForecastOutput, NodeEvent
│   │   └── forecast.py
│   ├── services/
│   │   ├── dataset_service.py
│   │   ├── preprocessing_service.py
│   │   ├── training_service.py           # RF + SVR dispatch
│   │   ├── evaluation_service.py
│   │   └── forecast_service.py           # Dispatches to RF/SVR or agent
│   ├── ml/
│   │   ├── preprocessing/
│   │   │   ├── imputer.py
│   │   │   ├── outlier.py
│   │   │   ├── features.py
│   │   │   ├── scaler.py
│   │   │   └── splitter.py
│   │   ├── models/
│   │   │   ├── random_forest.py
│   │   │   └── svr.py
│   │   ├── evaluation/
│   │   │   └── metrics.py
│   │   └── persistence/
│   │       └── model_store.py
│   ├── agents/                           # NEW — LangGraph agent system
│   │   ├── __init__.py
│   │   ├── graph.py                      # StateGraph definition + compile()
│   │   ├── state.py                      # AgentState TypedDict
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── data_preparation.py       # Node 1
│   │   │   ├── forecasting.py            # Node 2 — LLM call
│   │   │   ├── validation.py             # Node 3
│   │   │   └── revision.py              # Node 4 (conditional)
│   │   ├── tools/
│   │   │   ├── load_dataset.py           # @tool — reads processed parquet
│   │   │   ├── compute_stats.py          # @tool — pandas stats summary
│   │   │   └── build_prompt.py           # @tool — assembles LLM prompt
│   │   └── llm.py                        # LLM factory — model selection
│   └── utils/
│       ├── logger.py
│       └── job_store.py                  # In-memory job status dict
├── data/
│   ├── raw/
│   └── processed/
├── models/                               # .joblib files (RF, SVR, scalers)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_agent.py
│   └── test_api.py
├── requirements.txt
├── .env
├── .env.example
├── Dockerfile
└── render.yaml
```

---

## 3. System Architecture

### 3.1 High-Level Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                            BROWSER                                  │
│  Next.js 16 (React 19.2 · Tailwind · shadcn/ui · Recharts)         │
│                                                                      │
│  useAgentStream() ─── subscribes to SSE ────────────────────────┐  │
└──────────────────────┬───────────────────────────────────────────┼──┘
                       │ HTTPS                                      │
                       ▼                                            │ SSE
┌──────────────────────────────────────────────────────────────┐    │
│                   VERCEL (Next.js 16 Server)                   │    │
│                                                                │    │
│  ┌──────────────────┐   ┌────────────────────────────────┐   │    │
│  │  App Router RSCs  │   │  /api/* Route Handlers          │   │    │
│  │  (page.tsx)       │   │  - Inject Clerk JWT             │   │    │
│  │  use cache        │   │  - Proxy to FastAPI             │   │    │
│  └────────┬─────────┘   │  - /forecast/stream → SSE proxy │───┼────┘
│           │              └────────────┬───────────────────┘   │
│  ┌────────▼─────────┐                 │ HTTPS + Bearer token   │
│  │   Prisma 7        │                 │                        │
│  │  @prisma/adapter  │                 │                        │
│  │  -neon            │                 │                        │
│  └────────┬─────────┘                 │                        │
└───────────┼─────────────────────────────────────────────────────┘
            │                           │
            ▼                           ▼
┌───────────────────┐     ┌──────────────────────────────────────────┐
│   NEON POSTGRES   │     │           RENDER (FastAPI)                 │
│                   │     │                                            │
│  Users            │     │  ┌──────────────────────────────────────┐ │
│  Datasets         │     │  │  FastAPI + Uvicorn                    │ │
│  PreprocessJobs   │     │  │                                        │ │
│  ModelRuns        │     │  │  ┌─────────────┐  ┌────────────────┐ │ │
│  AgentRuns        │     │  │  │  /models/*   │  │  /forecast/*   │ │ │
│  EvalResults      │     │  │  │  RF + SVR    │  │  + /stream/*   │ │ │
│  ForecastRuns     │     │  │  │  training    │  │  SSE endpoint  │ │ │
└───────────────────┘     │  │  └──────┬──────┘  └───────┬────────┘ │ │
                          │  │         │                  │           │ │
                          │  │  ┌──────▼──────┐  ┌───────▼────────┐ │ │
                          │  │  │ scikit-learn │  │  LangGraph     │ │ │
                          │  │  │ RF / SVR     │  │  StateGraph    │ │ │
                          │  │  └─────────────┘  │  ┌───────────┐ │ │ │
                          │  │                   │  │data_prep  │ │ │ │
                          │  │                   │  │forecasting│ │ │ │
                          │  │                   │  │validation │ │ │ │
                          │  │                   │  │revision   │ │ │ │
                          │  │                   │  └─────┬─────┘ │ │ │
                          │  │                   └─────────┼───────┘ │ │
                          │  │                             │           │ │
                          │  │                   ┌─────────▼─────────┐│ │
                          │  │                   │  LLM API calls     ││ │
                          │  │                   │  GPT-5.4 (default) ││ │
                          │  │                   │  Gemini 3.1 Pro    ││ │
                          │  │                   │  Claude Sonnet 4.6 ││ │
                          │  │                   └───────────────────┘│ │
                          │  └──────────────────────────────────────── ┘ │
                          └──────────────────────────────────────────────┘
```

### 3.2 LangGraph StateGraph Flow

```
START
  │
  ▼
┌─────────────────────┐
│  data_preparation   │  Downloads processed parquet → computes stats
│  node               │  → builds context_json for LLM prompt
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   forecasting       │  Builds structured prompt → calls LLM
│   node              │  with .with_structured_output() → parses JSON
│   (timeout: 55s)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   validation        │  Checks: all values positive,
│   node              │  within ±3 std of historical mean,
│                     │  anomaly_pct computed
└──────────┬──────────┘
           │
     ┌─────┴──────┐
     │            │
     │ anomaly    │ anomaly_pct ≤ 10%
     │ _pct > 10% │ OR revision_count ≥ 2
     │            │
     ▼            ▼
┌──────────┐   ┌─────────────────────┐
│ revision │   │        END          │
│ node     │   │  returns best-effort│
│ (max 2x) │   │  result + reasoning │
└────┬─────┘   └─────────────────────┘
     │
     └──► back to forecasting_node
          with revision prompt
```

### 3.3 SSE Streaming Architecture

```
FastAPI /api/v1/forecast/stream/{agent_run_id}
  │
  └── async generator: stream_agent_events(agent_run_id)
        │
        └── graph.astream_events(state, version="v3")
              │
              ├── node_transition event → { type: "node_start", node: "data_preparation" }
              ├── node_transition event → { type: "node_complete", node: "data_preparation", duration_ms: 412 }
              ├── node_transition event → { type: "node_start", node: "forecasting" }
              ├── llm_token events     → { type: "token", content: "..." }   (streamed)
              ├── node_transition event → { type: "node_complete", node: "forecasting" }
              ├── node_transition event → { type: "node_start", node: "validation" }
              └── completion event     → { type: "complete", forecast_id: "..." }

Next.js /api/forecast/stream/[agentRunId]/route.ts
  └── ReadableStream proxy → Response with text/event-stream

Frontend: useAgentStream(agentRunId)
  └── EventSource subscription
        └── updates AgentProgressPanel component in real time
```

---

## 4. Component Breakdown

### 4.1 `proxy.ts` — Clerk Authentication Gate
```typescript
// proxy.ts  (NOT middleware.ts — Next.js 16 requirement)
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'

const isPublic = createRouteMatcher(['/', '/sign-in(.*)', '/sign-up(.*)'])

export const proxy = clerkMiddleware(async (auth, req) => {
  if (!isPublic(req)) await auth.protect()
})

export const config = {
  matcher: [
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    '/(api|trpc)(.*)',
  ],
}
```

### 4.2 `lib/prisma.ts` — Prisma 7 Singleton (Neon adapter)
```typescript
import { PrismaClient } from '@prisma/client'
import { PrismaNeon } from '@prisma/adapter-neon'
import { Pool } from '@neondatabase/serverless'

const pool = new Pool({ connectionString: process.env.DATABASE_URL })
const adapter = new PrismaNeon(pool)

const globalForPrisma = globalThis as unknown as { prisma: PrismaClient }

export const prisma =
  globalForPrisma.prisma ?? new PrismaClient({ adapter })

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma
```

### 4.3 `lib/api-client.ts` — JWT-Injecting Fetch Wrapper
```typescript
import { auth } from '@clerk/nextjs/server'

const BASE = process.env.FASTAPI_URL ?? 'http://localhost:8000'

export async function mlFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const { getToken } = await auth()
  const token = await getToken()

  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...options.headers,
    },
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? `FastAPI ${res.status}`)
  }
  return res.json() as Promise<T>
}
```

### 4.4 `hooks/useAgentStream.ts` — SSE Subscriber
```typescript
'use client'
import { useEffect, useState } from 'react'

export type NodeEvent = {
  type: 'node_start' | 'node_complete' | 'token' | 'complete' | 'error'
  node?: string
  content?: string
  duration_ms?: number
  forecast_id?: string
  error?: string
}

export function useAgentStream(agentRunId: string | null) {
  const [events, setEvents] = useState<NodeEvent[]>([])
  const [done, setDone] = useState(false)

  useEffect(() => {
    if (!agentRunId) return
    const es = new EventSource(`/api/forecast/stream/${agentRunId}`)

    es.onmessage = (e) => {
      const event: NodeEvent = JSON.parse(e.data)
      setEvents(prev => [...prev, event])
      if (event.type === 'complete' || event.type === 'error') {
        setDone(true)
        es.close()
      }
    }
    es.onerror = () => { setDone(true); es.close() }
    return () => es.close()
  }, [agentRunId])

  return { events, done }
}
```

### 4.5 `components/agent/AgentProgressPanel.tsx`
```typescript
'use client'
import { NodeEvent } from '@/hooks/useAgentStream'

const NODE_LABELS: Record<string, string> = {
  data_preparation: 'Preparing data context...',
  forecasting:      'Generating forecast with LLM...',
  validation:       'Validating predictions...',
  revision:         'Revising forecast...',
}

interface Props { events: NodeEvent[]; done: boolean }

export function AgentProgressPanel({ events, done }: Props) {
  const activeNode = [...events]
    .reverse()
    .find(e => e.type === 'node_start')?.node

  return (
    <div className="rounded-lg border p-4 space-y-2">
      <p className="text-sm font-medium text-muted-foreground">Agent Progress</p>
      {Object.keys(NODE_LABELS).map(node => {
        const started  = events.some(e => e.type === 'node_start'   && e.node === node)
        const complete = events.some(e => e.type === 'node_complete' && e.node === node)
        return (
          <div key={node} className="flex items-center gap-2 text-sm">
            <span>{complete ? '✅' : started ? '⏳' : '⬜'}</span>
            <span className={complete ? 'text-green-600' : started ? 'text-blue-600' : 'text-muted-foreground'}>
              {NODE_LABELS[node]}
            </span>
          </div>
        )
      })}
      {done && <p className="text-xs text-green-600 font-medium">Agent complete.</p>}
    </div>
  )
}
```

### 4.6 `hooks/useJobPoller.ts` — RF/SVR Training Status Poller
```typescript
'use client'
import { useEffect, useState } from 'react'

type JobStatus = 'PENDING' | 'RUNNING' | 'COMPLETE' | 'FAILED' | 'WARNING'
interface JobState { status: JobStatus; progress?: number; error?: string }

export function useJobPoller(jobId: string | null, intervalMs = 3000) {
  const [state, setState] = useState<JobState>({ status: 'PENDING' })

  useEffect(() => {
    if (!jobId) return
    const id = setInterval(async () => {
      const res  = await fetch(`/api/models/jobs/${jobId}/status`)
      const data: JobState = await res.json()
      setState(data)
      if (data.status === 'COMPLETE' || data.status === 'FAILED') clearInterval(id)
    }, intervalMs)
    return () => clearInterval(id)
  }, [jobId, intervalMs])

  return state
}
```

---

## 5. Database Schema (Prisma 7)

### 5.1 `prisma.config.ts`
```typescript
import { defineConfig } from 'prisma/config'
import { Pool } from '@neondatabase/serverless'
import 'dotenv/config'

export default defineConfig({
  earlyAccess: true,
  schema: 'prisma/schema.prisma',
  migrate: {
    async adapter() {
      const { PrismaNeon } = await import('@prisma/adapter-neon')
      const pool = new Pool({ connectionString: process.env.DATABASE_URL! })
      return new PrismaNeon(pool)
    },
  },
})
```

### 5.2 `prisma/schema.prisma`
```prisma
generator client {
  provider = "prisma-client"
  output   = "../node_modules/.prisma/client"
}

// ─── User ────────────────────────────────────────────────────────────────────
model User {
  id        String   @id           // Clerk userId
  email     String   @unique
  createdAt DateTime @default(now())

  datasets        Dataset[]
  weatherDatasets WeatherDataset[]
  preprocessJobs  PreprocessingJob[]
  modelRuns       ModelRun[]
  agentRuns       AgentRun[]
  forecastRuns    ForecastRun[]
}

// ─── Dataset ─────────────────────────────────────────────────────────────────
model Dataset {
  id               String           @id @default(cuid())
  userId           String
  user             User             @relation(fields: [userId], references: [id])
  name             String
  uploadthingUrl   String
  uploadthingKey   String
  rowCount         Int?
  validationStatus ValidationStatus @default(PENDING)
  validationReport Json?
  createdAt        DateTime         @default(now())
  deletedAt        DateTime?

  weatherDatasets WeatherDataset[]
  preprocessJobs  PreprocessingJob[]

  @@index([userId, createdAt])
}

enum ValidationStatus { PENDING VALID INVALID WARNING }

// ─── WeatherDataset ──────────────────────────────────────────────────────────
model WeatherDataset {
  id             String   @id @default(cuid())
  userId         String
  user           User     @relation(fields: [userId], references: [id])
  datasetId      String
  dataset        Dataset  @relation(fields: [datasetId], references: [id])
  uploadthingUrl String
  uploadthingKey String
  createdAt      DateTime @default(now())
}

// ─── PreprocessingJob ────────────────────────────────────────────────────────
model PreprocessingJob {
  id                String    @id @default(cuid())
  userId            String
  user              User      @relation(fields: [userId], references: [id])
  datasetId         String
  dataset           Dataset   @relation(fields: [datasetId], references: [id])
  status            JobStatus @default(PENDING)
  splitRatioTrain   Float     @default(0.70)
  splitRatioVal     Float     @default(0.15)
  splitRatioTest    Float     @default(0.15)
  resultSummaryJson Json?
  edaChartsJson     Json?
  processedFilePath String?
  createdAt         DateTime  @default(now())
  updatedAt         DateTime  @updatedAt

  modelRuns ModelRun[]
  agentRuns AgentRun[]

  @@index([userId, status])
}

enum JobStatus { PENDING RUNNING COMPLETE FAILED WARNING }

// ─── ModelRun — RF and SVR only ──────────────────────────────────────────────
model ModelRun {
  id               String           @id @default(cuid())
  userId           String
  user             User             @relation(fields: [userId], references: [id])
  preprocessJobId  String
  preprocessJob    PreprocessingJob @relation(fields: [preprocessJobId], references: [id])
  modelType        ModelType
  hyperparamsJson  Json
  status           JobStatus        @default(PENDING)
  modelFilePath    String?
  scalerFilePath   String?
  trainingTimeSecs Float?
  createdAt        DateTime         @default(now())

  evaluationResult EvaluationResult?
  forecastRuns     ForecastRun[]

  @@index([userId, status])
}

enum ModelType { RANDOM_FOREST SVR }   // ANN and LSTM removed

// ─── AgentRun — LangGraph pipeline runs ─────────────────────────────────────
model AgentRun {
  id              String           @id @default(cuid())
  userId          String
  user            User             @relation(fields: [userId], references: [id])
  preprocessJobId String
  preprocessJob   PreprocessingJob @relation(fields: [preprocessJobId], references: [id])
  status          JobStatus        @default(PENDING)
  modelUsed       String           // "openai/gpt-5.4" | "google/gemini-3.1-pro-preview" | "anthropic/claude-sonnet-4-6"
  nodeTraceJson   Json?            // [{node, startedAt, completedAt, durationMs}]
  revisionCount   Int              @default(0)
  tokenCount      Int?
  createdAt       DateTime         @default(now())
  updatedAt       DateTime         @updatedAt

  evaluationResult EvaluationResult?
  forecastRuns     ForecastRun[]

  @@index([userId, status])
}

// ─── EvaluationResult ────────────────────────────────────────────────────────
model EvaluationResult {
  id                    String    @id @default(cuid())
  // One of modelRunId OR agentRunId is set — not both
  modelRunId            String?   @unique
  modelRun              ModelRun? @relation(fields: [modelRunId], references: [id])
  agentRunId            String?   @unique
  agentRun              AgentRun? @relation(fields: [agentRunId], references: [id])
  sourceType            SourceType
  rmse                  Float
  mae                   Float
  mape                  Float
  r2                    Float
  testSetSize           Int
  actualJson            Json      // [{timestamp, value}]
  predictedJson         Json      // [{timestamp, value}]
  featureImportanceJson Json?     // RF only: [{feature, importance}]
  evaluatedAt           DateTime  @default(now())
}

enum SourceType { RF SVR AGENT }

// ─── ForecastRun ─────────────────────────────────────────────────────────────
model ForecastRun {
  id                  String     @id @default(cuid())
  userId              String
  user                User       @relation(fields: [userId], references: [id])
  // One of modelRunId OR agentRunId is set
  modelRunId          String?
  modelRun            ModelRun?  @relation(fields: [modelRunId], references: [id])
  agentRunId          String?
  agentRun            AgentRun?  @relation(fields: [agentRunId], references: [id])
  sourceType          SourceType
  startDate           DateTime
  horizonDays         Int
  resolution          Resolution
  forecastJson        Json       // [{timestamp, value}]
  agentReasoningText  String?    // LangGraph agent only
  createdAt           DateTime   @default(now())

  @@index([userId, createdAt])
}

enum Resolution { HOURLY DAILY WEEKLY }
```

---

## 6. API Design (FastAPI)

### 6.1 `app/main.py` — Startup with Graph Compilation
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import datasets, preprocessing, models, forecast
from app.agents.graph import compile_graph
from app.config import settings

# Compile the LangGraph StateGraph ONCE at startup
compiled_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph
    compiled_graph = compile_graph()   # expensive — do not call per request
    app.state.agent_graph = compiled_graph
    yield

app = FastAPI(title="ECFML API", version="2.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(datasets.router,      prefix="/api/v1")
app.include_router(preprocessing.router, prefix="/api/v1")
app.include_router(models.router,        prefix="/api/v1")
app.include_router(forecast.router,      prefix="/api/v1")

@app.get("/health")
async def health(): return {"status": "ok", "version": "2.0.0"}
```

### 6.2 `app/config.py`
```python
from pydantic_settings import BaseSettings
from typing import Literal

ALLOWED_MODELS = {
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4-6",
}

class Settings(BaseSettings):
    CLERK_JWKS_URL: str
    ALLOWED_ORIGINS: list[str]
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"
    ACTIVE_LLM_MODEL: str = "openai/gpt-5.4"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    LANGCHAIN_API_KEY: str = ""           # LangSmith tracing
    LANGCHAIN_TRACING_V2: str = "true"

    def model_post_init(self, __context):
        if self.ACTIVE_LLM_MODEL not in ALLOWED_MODELS:
            raise ValueError(
                f"ACTIVE_LLM_MODEL must be one of {ALLOWED_MODELS}. "
                f"Got: {self.ACTIVE_LLM_MODEL}"
            )

    class Config:
        env_file = ".env"

settings = Settings()
```

### 6.3 `app/agents/llm.py` — LLM Factory
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings

def get_llm():
    """Return the configured LLM with structured output support."""
    model = settings.ACTIVE_LLM_MODEL

    if model == "openai/gpt-5.4":
        return ChatOpenAI(
            model="gpt-5.4",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1,
        )
    elif model == "anthropic/claude-sonnet-4-6":
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1,
        )
    elif model == "google/gemini-3.1-pro-preview":
        return ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
        )
    raise ValueError(f"Unknown model: {model}")
```

### 6.4 `app/agents/state.py` — LangGraph Agent State
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Input
    dataset_path: str
    forecast_params: dict          # {start_date, horizon_days, resolution}
    agent_run_id: str

    # Computed by nodes
    context_json: dict             # data_preparation output
    llm_response: dict | None      # raw LLM structured output
    predictions: list[dict] | None # [{timestamp, value}]
    validation_report: dict | None # {passed, anomaly_pct, failed_indices}
    revision_count: int
    reasoning: str | None
    status: str                    # for SSE streaming labels
    error: str | None
```

### 6.5 `app/agents/graph.py` — StateGraph Definition
```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy, TimeoutPolicy
from app.agents.state import AgentState
from app.agents.nodes.data_preparation import data_preparation_node
from app.agents.nodes.forecasting import forecasting_node
from app.agents.nodes.validation import validation_node
from app.agents.nodes.revision import revision_node

MAX_REVISIONS = 2

def should_revise(state: AgentState) -> str:
    """Conditional edge: revise or end."""
    report = state.get("validation_report", {})
    if (report.get("anomaly_pct", 0) > 10 and
            state.get("revision_count", 0) < MAX_REVISIONS):
        return "revision"
    return END

def compile_graph():
    builder = StateGraph(AgentState)

    builder.add_node("data_preparation", data_preparation_node)
    builder.add_node(
        "forecasting",
        forecasting_node,
        retry=RetryPolicy(max_attempts=3),
        timeout=TimeoutPolicy(run_timeout=55),   # enforce < 60s NFR
    )
    builder.add_node("validation",  validation_node)
    builder.add_node("revision",    revision_node)

    builder.add_edge(START, "data_preparation")
    builder.add_edge("data_preparation", "forecasting")
    builder.add_edge("forecasting", "validation")
    builder.add_conditional_edges("validation", should_revise, ["revision", END])
    builder.add_edge("revision", "forecasting")   # loop back

    return builder.compile(checkpointer=MemorySaver())
```

### 6.6 `app/agents/nodes/data_preparation.py`
```python
import pandas as pd
from langgraph.config import get_stream_writer
from app.agents.state import AgentState

async def data_preparation_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer({"type": "node_start", "node": "data_preparation",
            "message": "Loading and summarising consumption data..."})

    df = pd.read_parquet(state["dataset_path"])
    consumption = df["consumption_kwh"]

    context = {
        "row_count": len(df),
        "date_range": [str(df.index.min()), str(df.index.max())],
        "mean_kwh": round(float(consumption.mean()), 2),
        "std_kwh": round(float(consumption.std()), 2),
        "min_kwh": round(float(consumption.min()), 2),
        "max_kwh": round(float(consumption.max()), 2),
        "recent_24h": consumption.tail(24).tolist(),
        "recent_7d_daily_means": (
            consumption.tail(168).resample("D").mean().tolist()
        ),
        "rolling_mean_24h": round(float(consumption.tail(24).mean()), 2),
        "rolling_mean_7d":  round(float(consumption.tail(168).mean()), 2),
    }

    writer({"type": "node_complete", "node": "data_preparation"})
    return {"context_json": context, "status": "data_prepared"}
```

### 6.7 `app/agents/nodes/forecasting.py`
```python
import json
from pydantic import BaseModel
from langgraph.config import get_stream_writer
from app.agents.state import AgentState
from app.agents.llm import get_llm

class ForecastOutput(BaseModel):
    predictions: list[dict]   # [{timestamp: str, value: float}]
    reasoning:   str
    confidence:  str          # "high" | "medium" | "low"

SYSTEM_PROMPT = """You are an expert electricity demand forecasting assistant.
You will be given historical consumption statistics and asked to generate a
structured short-term or medium-term forecast for the North West Region of Cameroon.
Return ONLY valid structured data matching the required schema.
Base your forecast on the provided statistics, seasonality patterns, and context.
Do not invent data not present in the context."""

async def forecasting_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer({"type": "node_start", "node": "forecasting",
            "message": "Generating forecast with LLM..."})

    ctx    = state["context_json"]
    params = state["forecast_params"]
    rev    = state.get("revision_count", 0)

    revision_note = ""
    if rev > 0:
        report = state.get("validation_report", {})
        revision_note = (
            f"\nIMPORTANT: Your previous forecast had {report.get('anomaly_pct', 0):.1f}% "
            f"anomalous predictions (outside ±3 std of historical mean = {ctx['mean_kwh']} ± "
            f"{ctx['std_kwh'] * 3:.2f} kWh). Please correct this in your revised forecast."
        )

    user_prompt = f"""
Historical consumption summary:
{json.dumps(ctx, indent=2)}

Forecast request:
- Start date: {params['start_date']}
- Horizon: {params['horizon_days']} days
- Resolution: {params['resolution']}

Generate hourly/daily/weekly predictions from the start date covering the full horizon.
Each prediction must have a timestamp (ISO 8601) and value (float, kWh).
{revision_note}
"""

    llm = get_llm().with_structured_output(ForecastOutput)
    result: ForecastOutput = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ])

    writer({"type": "node_complete", "node": "forecasting"})
    return {
        "predictions": result.predictions,
        "reasoning":   result.reasoning,
        "llm_response": result.model_dump(),
        "status": "forecast_generated",
    }
```

### 6.8 `app/agents/nodes/validation.py`
```python
import numpy as np
from langgraph.config import get_stream_writer
from app.agents.state import AgentState

async def validation_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer({"type": "node_start", "node": "validation",
            "message": "Validating predicted values..."})

    predictions = state["predictions"] or []
    ctx = state["context_json"]
    mean, std = ctx["mean_kwh"], ctx["std_kwh"]
    lower, upper = mean - 3 * std, mean + 3 * std

    failed = [
        i for i, p in enumerate(predictions)
        if not (lower <= p.get("value", -1) <= upper) or p.get("value", -1) < 0
    ]
    anomaly_pct = len(failed) / max(len(predictions), 1) * 100

    report = {
        "passed": len(predictions) - len(failed),
        "failed_indices": failed,
        "anomaly_pct": round(anomaly_pct, 2),
        "bounds": {"lower": round(lower, 2), "upper": round(upper, 2)},
    }

    writer({"type": "node_complete", "node": "validation",
            "anomaly_pct": anomaly_pct})
    return {
        "validation_report": report,
        "status": "validated",
    }
```

### 6.9 `app/agents/nodes/revision.py`
```python
from langgraph.config import get_stream_writer
from app.agents.state import AgentState

async def revision_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer({"type": "node_start", "node": "revision",
            "message": f"Revision cycle {state.get('revision_count', 0) + 1}/2..."})
    writer({"type": "node_complete", "node": "revision"})
    return {"revision_count": state.get("revision_count", 0) + 1}
```

### 6.10 Forecast Router — SSE Endpoint
```python
# app/routers/forecast.py (key endpoints)
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
import json, asyncio
from app.dependencies import get_current_user
from app.agents.state import AgentState

router = APIRouter(tags=["forecast"])

@router.post("/forecast")
async def create_forecast(
    body: ForecastRequest,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Dispatch to RF, SVR, or LangGraph agent."""
    if body.engine == "AGENT":
        # Start agent run asynchronously, return agent_run_id immediately
        agent_run_id = generate_id()
        asyncio.create_task(
            run_agent(request.app.state.agent_graph, agent_run_id, body, user)
        )
        return {"agent_run_id": agent_run_id, "status": "PENDING"}
    else:
        # RF or SVR — synchronous (fast)
        result = await forecast_service.run_ml_forecast(body, user)
        return result

@router.get("/forecast/stream/{agent_run_id}")
async def stream_agent_forecast(
    agent_run_id: str,
    user: dict = Depends(get_current_user)
):
    """SSE endpoint — streams LangGraph node events."""
    async def event_generator():
        async for event in agent_event_store.subscribe(agent_run_id):
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("complete", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",    # essential for nginx proxy on Render
        }
    )
```

### 6.11 Full API Endpoint Reference

```
# Authentication
POST   /api/v1/auth/validate-token        Internal JWT check

# Datasets
GET    /api/v1/datasets/{id}/preview      ?rows=100
POST   /api/v1/datasets/{id}/validate     { file_url }

# Pre-processing
POST   /api/v1/preprocessing/run          { job_id, dataset_url, weather_url?, splits }
GET    /api/v1/preprocessing/{job_id}/status
GET    /api/v1/preprocessing/{job_id}/eda

# Models — RF and SVR only
POST   /api/v1/models/train               { job_id, preprocess_job_id, model_type: RF|SVR, hyperparams }
POST   /api/v1/models/tune                { job_id, preprocess_job_id, model_type, param_grid }
GET    /api/v1/models/jobs/{job_id}/status
POST   /api/v1/models/{model_id}/evaluate

# Forecast
POST   /api/v1/forecast                   { engine: RF|SVR|AGENT, model_run_id?, preprocess_job_id, start_date, horizon_days, resolution, model_override? }
GET    /api/v1/forecast/stream/{agent_run_id}   SSE
GET    /api/v1/agents/{agent_run_id}/status

# Health
GET    /health
```

---

## 7. ML Pipeline Design (RF + SVR)

### 7.1 Pre-processing Pipeline
*(Unchanged from Design Doc v1.0 — see Section 7.1 of that document for the full imputer/outlier/features/scaler/splitter pipeline.)*

### 7.2 Feature Engineering Reference
*(Unchanged from Design Doc v1.0 — all 21 features retained.)*

### 7.3 Random Forest (`app/ml/models/random_forest.py`)
```python
from sklearn.ensemble import RandomForestRegressor
import joblib, os
from app.schemas.model import RFHyperparams

def train_rf(X_train, y_train, params: RFHyperparams,
             job_id: str, job_store: dict) -> RandomForestRegressor:
    job_store[job_id] = {"status": "RUNNING", "progress": 0}
    model = RandomForestRegressor(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)
    job_store[job_id] = {"status": "COMPLETE", "progress": 100}
    return model

def get_feature_importance(model: RandomForestRegressor,
                            feature_names: list[str]) -> list[dict]:
    return sorted(
        [{"feature": f, "importance": round(float(i), 6)}
         for f, i in zip(feature_names, model.feature_importances_)],
        key=lambda x: x["importance"], reverse=True
    )[:15]  # top 15 features for Recharts BarChart
```

### 7.4 SVR (`app/ml/models/svr.py`)
```python
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from app.schemas.model import SVRHyperparams

def train_svr(X_train, y_train, params: SVRHyperparams,
              job_id: str, job_store: dict):
    # SVR is sensitive to feature scale — wrap in Pipeline
    job_store[job_id] = {"status": "RUNNING", "progress": 0}
    pipeline = Pipeline([
        ("scaler", SKStandardScaler()),
        ("svr", SVR(
            kernel=params.kernel,
            C=params.C,
            epsilon=params.epsilon,
            gamma=params.gamma,
        ))
    ])
    pipeline.fit(X_train, y_train)
    job_store[job_id] = {"status": "COMPLETE", "progress": 100}
    return pipeline
```

### 7.5 Evaluation Metrics (`app/ml/evaluation/metrics.py`)
```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    scaler=None) -> dict:
    # Inverse-transform if scaler provided (consumption column index = 0)
    if scaler:
        y_true = scaler.inverse_transform(
            np.zeros((len(y_true), scaler.n_features_in_))
        )[:, 0]
        y_pred = scaler.inverse_transform(
            np.zeros((len(y_pred), scaler.n_features_in_))
        )[:, 0]

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) /
                                     np.maximum(np.abs(y_true), 1e-8))) * 100),
        "r2":   float(r2_score(y_true, y_pred)),
    }
```

### 7.6 Model Persistence
```python
# app/ml/persistence/model_store.py
import os, joblib

MODELS_DIR = os.getenv("MODELS_DIR", "./models")

def save_sklearn_model(model, model_run_id: str) -> str:
    path = os.path.join(MODELS_DIR, f"{model_run_id}.joblib")
    joblib.dump(model, path)
    return path

def load_sklearn_model(path: str):
    return joblib.load(path)

def save_scaler(scaler, model_run_id: str) -> str:
    path = os.path.join(MODELS_DIR, f"{model_run_id}_scaler.joblib")
    joblib.dump(scaler, path)
    return path
```

---

## 8. LangGraph Agent Design

*(Full StateGraph, node implementations, and routing logic are in Section 6 above. This section provides supplementary design notes.)*

### 8.1 Why StateGraph over create_agent

`create_react_agent` is deprecated in LangGraph v1.0 in favour of LangChain's `create_agent`. However, for this pipeline, a manual `StateGraph` is preferred because:
- The forecasting pipeline is not a generic tool-calling ReAct loop — it is a fixed, ordered sequence with one conditional branch.
- `StateGraph` gives explicit control over node ordering, timeout policies per node, and revision routing logic.
- The `get_stream_writer()` SSE integration works naturally at the node level.

Use `create_agent` only if you add open-ended tool-calling behaviour in a future sprint.

### 8.2 LangSmith Observability

Set these environment variables in the FastAPI service to enable LangSmith tracing on all LangGraph runs:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=ecfml-agent
```
Every node execution, LLM call, token count, and latency will be visible in the LangSmith dashboard. Free tier: 5,000 traces/month — sufficient for development and evaluation.

### 8.3 Token Budget

To stay within the 8,000-token NFR per forecast request:
- `context_json` is capped: `recent_24h` list is limited to 24 floats, `recent_7d_daily_means` to 7 floats.
- Horizon > 30 days is rejected at the API level for the AGENT engine (HTTP 400) to prevent oversized response schemas.
- `.with_structured_output()` enforces JSON output, preventing verbose prose from consuming tokens.

### 8.4 MemorySaver vs PostgresSaver

| | MemorySaver | PostgresSaver |
|---|---|---|
| Use case | Prototype / development | Production |
| Persistence | In-memory (lost on restart) | Neon Postgres |
| Setup | Zero config | `pip install langgraph-checkpoint-postgres` + connection string |
| Recommended for | This academic prototype | Post-submission production use |

To upgrade: replace `MemorySaver()` in `compile_graph()` with:
```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
```

---

## 9. UI/UX Wireframes & User Flows

### 9.1 Updated User Flow

```
Landing (/) ──► Sign In ──► /dashboard (Overview)
                                  │
          ┌───────────────────────┼──────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
     /data                  /preprocess              /train
  Upload dataset           Run pipeline            Configure RF or SVR
  Preview table            EDA Recharts            Poll progress bar
          │                       │                       │
          └───────────────────────┴───────────────────────┘
                                  │
                    ┌─────────────┴───────────────┐
                    │                             │
                    ▼                             ▼
              /evaluate                      /forecast
          3-way comparison            Select engine (RF / SVR / Agent)
          RF vs SVR vs Agent          RF/SVR → direct chart
          Feature importance          Agent → SSE progress panel
          Agent reasoning panel            → AreaChart + reasoning
```

### 9.2 `/dashboard/train` — RF + SVR Only

```
┌──────────────────────────────────────────────────────────────┐
│  Model Training                                               │
│                                                               │
│  Note: ANN and LSTM are not available in this version.        │
│  Use the LangGraph Agent on the Forecast page for            │
│  LLM-based forecasting.                                       │
│                                                               │
│  Pre-processing Job: [job_abc123 ▼]                          │
│  Select Model:  ● Random Forest  ○ SVR                        │
│                                                               │
│  Random Forest Hyperparameters                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  N Estimators:   [200]    Max Depth:     [None]        │  │
│  │  Min Samples:    [2]      ☐ Enable Grid Search CV      │  │
│  └────────────────────────────────────────────────────────┘  │
│                              [Start Training]                 │
│                                                               │
│  ──── Status: COMPLETE ─────────────────────────────────     │
│  Training time: 1m 43s     [Evaluate Model →]                │
│                                                               │
│  Previous Runs                                                │
│  ┌──────────────┬──────────┬──────────┬──────────────────┐  │
│  │  Model       │  Status  │  Time    │  Action          │  │
│  ├──────────────┼──────────┼──────────┼──────────────────┤  │
│  │  RF          │ COMPLETE │ 1m 43s   │ [Evaluate]       │  │
│  │  SVR         │ COMPLETE │ 4m 12s   │ [Evaluate]       │  │
│  └──────────────┴──────────┴──────────┴──────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 9.3 `/dashboard/evaluate` — 3-Way Comparison

```
┌──────────────────────────────────────────────────────────────┐
│  Model Evaluation                                             │
│                                                               │
│  Comparison: RF vs SVR vs LangGraph Agent                     │
│  ┌─────────────────┬──────────┬──────────┬──────────┬──────┐ │
│  │  Engine         │  RMSE    │  MAE     │  MAPE    │  R²  │ │
│  ├─────────────────┼──────────┼──────────┼──────────┼──────┤ │
│  │★ Random Forest  │  89.2    │  67.4    │  7.2%    │ 0.961│ │  ← best
│  │  SVR            │  112.8   │  84.1    │  9.1%    │ 0.943│ │
│  │  LLM Agent      │  134.5   │  98.2    │  11.4%   │ 0.921│ │
│  │  (GPT-5.4)      │          │          │          │      │ │
│  └─────────────────┴──────────┴──────────┴──────────┴──────┘ │
│                                                               │
│  View chart for: [Random Forest ▼]                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Actual vs Predicted — Test Set (Recharts LineChart)   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  RF Feature Importance (top 15)                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  lag_168h ████████████████████████  0.312              │  │
│  │  lag_24h  ████████████████          0.198              │  │
│  │  hour     ████████████              0.147              │  │
│  │  ...      (Recharts HorizontalBarChart)                │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ▼ LLM Agent Reasoning (GPT-5.4)                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  "Based on the provided 7-day rolling mean of 1,243 kWh│  │
│  │  and the clear weekday/weekend pattern in the recent   │  │
│  │  24-hour data, I forecast a moderate peak on Tuesday..." │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 9.4 `/dashboard/forecast` — Engine Selector + Agent Progress

```
┌──────────────────────────────────────────────────────────────┐
│  Forecast                                                     │
│                                                               │
│  Engine:  ○ Random Forest  ○ SVR  ● LangGraph Agent          │
│           Active model: openai/gpt-5.4                        │
│                                                               │
│  Start Date: [2025-06-01]   Horizon: [7] days                │
│  Resolution: ● Hourly  ○ Daily  ○ Weekly                     │
│                                    [Generate Forecast]        │
│                                                               │
│  ─── Agent Progress ────────────────────────────────────     │
│  ✅ Preparing data context...            (412 ms)             │
│  ⏳ Generating forecast with LLM...                           │
│  ⬜ Validating predictions...                                  │
│                                                               │
│  [streaming tokens appear here during forecasting node]       │
│                                                               │
│  ─── Forecast: June 1–7, 2025 (Hourly) ────────────────     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  [Recharts AreaChart — predicted kWh]                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ▼ Agent Reasoning                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  "Confidence: high. Based on rolling mean of 1,243 kWh │  │
│  │  and typical weekday peaks at 18:00–20:00..."          │  │
│  └────────────────────────────────────────────────────────┘  │
│                              [Download CSV ↓]                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. Data Flow Diagrams

### 10.1 LangGraph Agent Forecast Flow

```
User clicks "Generate Forecast" (engine = AGENT)
        │
        ▼
POST /api/forecast (Next.js API route)
  → mlFetch('/api/v1/forecast', { engine: "AGENT", ... })
        │
        ▼
FastAPI POST /api/v1/forecast
  1. Validate JWT
  2. Create AgentRun record (status: PENDING) via Prisma Server Action callback
  3. asyncio.create_task(run_agent(graph, agent_run_id, body))
  4. Return { agent_run_id, status: "PENDING" } immediately
        │
        ├──── Response returned to frontend ────────────────────────────┐
        │                                                               │
        ▼ (background)                                                  ▼
run_agent() coroutine:                                    Frontend receives agent_run_id
  1. Build initial AgentState                            → useAgentStream(agent_run_id) activates
  2. async for event in graph.astream_events(            → EventSource('/api/forecast/stream/{id}')
       state, version="v3"):                             → AgentProgressPanel updates in real time
       - get_stream_writer() events → agent_event_store
       - agent_event_store notifies SSE subscribers
  3. On completion:
     - Update AgentRun (status: COMPLETE, nodeTraceJson)
     - Create ForecastRun record
     - Push { type: "complete", forecast_id } event
        │
        ▼
Frontend: SSE complete event received
  → useAgentStream sets done=true
  → Fetches forecast data via GET /api/forecast/{forecast_id}
  → Renders ForecastChart + AgentReasoningPanel
```

### 10.2 RF/SVR Forecast Flow

```
User clicks "Generate Forecast" (engine = RF or SVR)
        │
        ▼
POST /api/forecast (Next.js API route)
  → mlFetch('/api/v1/forecast', { engine: "RF", model_run_id, ... })
        │
        ▼
FastAPI POST /api/v1/forecast (synchronous path)
  1. Validate JWT
  2. Load .joblib model + scaler from ModelRun.modelFilePath
  3. Build feature vectors for forecast horizon
  4. Run model.predict(X_forecast)
  5. Inverse-transform predictions → real kWh values
  6. Create ForecastRun record via Server Action callback
  7. Return { forecast_id, predictions: [{timestamp, value}] }
        │
        ▼
Frontend renders ForecastChart (Recharts AreaChart)
No SSE stream needed — response is synchronous (< 3s)
```

---

## 11. Deployment Architecture

### 11.1 Overview

```
Git push to main
        │
   ┌────┴──────────┐
   ▼               ▼
Vercel           Render
(Next.js 16)     (FastAPI Docker)
Build:           CMD: uvicorn app.main:app
  prisma generate       --host 0.0.0.0 --port 8000
  && next build
        │               │
        └──────┬─────────┘
               ▼
        Neon Postgres
        (shared DB)

External:
  Clerk → auth
  Uploadthing → file storage
  OpenAI / Anthropic / Google → LLM APIs
  LangSmith → agent observability
```

### 11.2 `Dockerfile` (FastAPI)
```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.3 `render.yaml`
```yaml
services:
  - type: web
    name: ecfml-api
    env: docker
    dockerfilePath: ./Dockerfile
    plan: starter
    envVars:
      - key: CLERK_JWKS_URL
        sync: false
      - key: ALLOWED_ORIGINS
        value: https://ecfml.vercel.app
      - key: ACTIVE_LLM_MODEL
        value: openai/gpt-5.4
      - key: OPENAI_API_KEY
        sync: false
      - key: ANTHROPIC_API_KEY
        sync: false
      - key: GOOGLE_API_KEY
        sync: false
      - key: LANGCHAIN_API_KEY
        sync: false
      - key: LANGCHAIN_TRACING_V2
        value: "true"
      - key: LANGCHAIN_PROJECT
        value: ecfml-agent
    disk:
      name: model-storage
      mountPath: /app/models
      sizeGB: 5
```

### 11.4 `requirements.txt`
```
fastapi>=0.136.1
uvicorn[standard]>=0.46.0
pydantic>=2.13.3
pydantic-settings>=2.14.0
greenlet==3.5.0
sqlalchemy==2.0.49
sqlmodel==0.0.38
pyjwt[crypto]>=2.12.0
httpx>=0.28.1,<0.29.0
pandas==2.2.3
numpy==2.1.2
scikit-learn==1.5.2
joblib==1.4.2
openpyxl==3.1.5
pyarrow>=24.0.0
langgraph==1.2.0
langgraph-checkpoint>=2.1.0,<5.0.0
langchain-core>=1.4.0,<2 
langchain-openai==1.2.1
langchain-anthropic==1.3.5
langchain-google-genai==4.1.1
```

### 11.5 `package.json` (key deps)
```json
{
  "dependencies": {
    "next": "^16.2.4",
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "@clerk/nextjs": "^7.0.0",
    "@prisma/client": "^7.0.0",
    "@prisma/adapter-neon": "^7.0.0",
    "@neondatabase/serverless": "^0.10.4",
    "recharts": "^2.15.3",
    "uploadthing": "^7.6.0",
    "@uploadthing/react": "^7.6.0",
    "ai": "^6.0.0",
    "tailwindcss": "^4.1.0",
    "class-variance-authority": "^0.7.1",
    "clsx": "^2.1.1",
    "tailwind-merge": "^3.3.0"
  },
  "devDependencies": {
    "prisma": "^7.0.0",
    "typescript": "^5.8.0",
    "@types/react": "^19.1.0",
    "@types/node": "^22.0.0"
  },
  "scripts": {
    "dev": "next dev --turbopack",
    "build": "prisma generate && next build",
    "start": "next start"
  }
}
```

---

## 12. Environment Variables

### 12.1 `ecfml-web/.env.example`
```bash
# Neon Postgres
DATABASE_URL="postgresql://user:password@host.neon.tech/ecfml?sslmode=require"

# Clerk v7
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY="pk_test_..."
CLERK_SECRET_KEY="sk_test_..."
NEXT_PUBLIC_CLERK_SIGN_IN_URL="/sign-in"
NEXT_PUBLIC_CLERK_SIGN_UP_URL="/sign-up"
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL="/dashboard"
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL="/dashboard"

# Uploadthing
UPLOADTHING_SECRET="sk_live_..."
NEXT_PUBLIC_UPLOADTHING_APP_ID="..."

# FastAPI ML + Agent Service
FASTAPI_URL="https://ecfml-api.onrender.com"

# Vercel AI SDK v6 (Low Priority — dashboard chat panel only)
AI_GATEWAY_URL="https://ai-gateway.vercel.sh"
AI_GATEWAY_KEY="..."
```

### 12.2 `ecfml-api/.env.example`
```bash
# Clerk JWKS
CLERK_JWKS_URL="https://your-domain.clerk.accounts.dev/.well-known/jwks.json"

# CORS
ALLOWED_ORIGINS='["https://ecfml.vercel.app","http://localhost:3000"]'

# Storage
DATA_DIR="./data"
MODELS_DIR="./models"

# LLM — set exactly ONE of these, or all three (only ACTIVE_LLM_MODEL is used at runtime)
ACTIVE_LLM_MODEL="openai/gpt-5.4"
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GOOGLE_API_KEY="AIza..."

# LangSmith observability
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__..."
LANGCHAIN_PROJECT="ecfml-agent"
```

---

## 13. Error Handling Strategy

### 13.1 FastAPI — RFC 7807 Error Responses
```json
{
  "type": "https://ecfml.api/errors/agent-timeout",
  "title": "Agent Forecast Timeout",
  "status": 504,
  "detail": "LangGraph forecasting node exceeded 55-second timeout. Try reducing forecast horizon."
}
```

### 13.2 LangGraph — Resilience Patterns

| Scenario | Behaviour |
|---|---|
| LLM API rate limit | RetryPolicy(max_attempts=3) with exponential backoff on forecasting_node |
| LLM timeout (> 55s) | NodeTimeoutPolicy fires → NodeTimeoutError → AgentRun status = FAILED |
| Invalid LLM JSON | .with_structured_output() raises OutputParserException → treated as failed validation → revision cycle |
| 2 revision cycles exhausted | Agent returns best-effort predictions with WARNING status and note in reasoning field |
| LLM API key missing | Startup validation in Settings raises ValueError — service refuses to start |
| Unknown ACTIVE_LLM_MODEL | Startup validation raises ValueError with allowed values listed |

### 13.3 Frontend — SSE Error Handling
```typescript
// useAgentStream.ts — error handling
es.onerror = () => {
  setEvents(prev => [...prev, {
    type: 'error',
    error: 'Connection to agent stream lost. Check your network and try again.'
  }])
  setDone(true)
  es.close()
}
```

### 13.4 RF/SVR Pipeline — Graceful Failures

| Scenario | Behaviour |
|---|---|
| Model file missing at inference | HTTP 404: "Model file not found — retrain required" |
| Outlier rate > 5% in preprocessing | Job status = WARNING; frontend prompts user to confirm |
| SVR training > 10 min (large dataset) | BackgroundTask timeout → FAILED with suggestion to reduce dataset or use subsampling |
| MAPE > 20% | Warning flag on EvaluationResult; yellow badge on comparison table row |

---

*End of ECFML System Design Document — v2.0  |  May 2026*
