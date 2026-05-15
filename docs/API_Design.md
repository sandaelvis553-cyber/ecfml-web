# ECFML API Design Document (v2.0.0)

This document outlines the API endpoints for the Electricity Consumption Forecasting (ECFML) backend. This is intended for use by the frontend development team to integrate with the forecasting system.

## General Information

- **Base URL**: `/api/v1` (unless otherwise specified)
- **Version**: 2.0.0
- **Format**: All request and response bodies are in JSON.

## Authentication

The API uses **Clerk** for authentication. All protected routes require a Bearer token in the `Authorization` header.

```http
Authorization: Bearer <clerk_jwt_token>
```

---

## Error Responses

The API uses standard HTTP status codes to indicate the success or failure of a request.

| Status Code | Description |
| :--- | :--- |
| `200 OK` | The request was successful. |
| `201 Created` | The resource was successfully created. |
| `400 Bad Request` | The request was invalid or could not be processed. |
| `401 Unauthorized` | Authentication failed or the user lacks permissions. |
| `403 Forbidden` | The user does not have access to the requested resource. |
| `404 Not Found` | The requested resource was not found. |
| `422 Unprocessable Entity` | Validation error (e.g., missing fields, wrong types). |
| `500 Internal Server Error` | An unexpected error occurred on the server. |

### Validation Error Detail (422)
When a `422` error occurs, the response body contains details about the failure:
```json
{
  "detail": [
    {
      "loc": ["body", "file_url"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## 1. Datasets

Endpoints for dataset management, validation, and previewing.

### List Datasets
Retrieves all datasets for the authenticated user.

- **URL**: `GET /datasets`
- **Status Code**: `200 OK`
- **Response**: `list[DatasetRead]`
  ```json
  [
    {
      "id": "string",
      "name": "string",
      "file_url": "string",
      "uploadthing_key": "string",
      "user_id": "string",
      "validation_status": "PENDING" | "VALID" | "INVALID" | "WARNING",
      "validation_report": {},
      "created_at": "string",
      "row_count": number,
      "deleted_at": "string"
    }
  ]
  ```

### Create Dataset Record
Registers a new dataset (after it has been uploaded to storage).

- **URL**: `POST /datasets`
- **Status Code**: `201 Created`
- **Request Body**: `DatasetCreate`
  ```json
  {
    "name": "string",
    "file_url": "string",
    "uploadthing_key": "string",
    "id": "string" (optional)
  }
  ```
- **Response**: `DatasetRead` (same as above)

### Delete Dataset
Soft-deletes a dataset.

- **URL**: `DELETE /datasets/{dataset_id}`
- **Status Code**: `200 OK`
- **Response**:
  ```json
  {
    "status": "deleted"
  }
  ```

### Validate Dataset
Validates a dataset file against required columns and format.

- **URL**: `POST /datasets/{dataset_id}/validate`
- **Status Code**: `200 OK`
- **Request Body**:
  ```json
  {
    "file_url": "string",
    "required_columns": ["string"]
  }
  ```
- **Response**: `DatasetValidationReport`
  ```json
  {
    "status": "PENDING" | "VALID" | "INVALID" | "WARNING",
    "missing_columns": ["string"],
    "columns": ["string"],
    "row_count": number,
    "warnings": ["string"],
    "details": {}
  }
  ```

### Preview Dataset
Retrieves a preview (first N rows) of the dataset.

- **URL**: `GET /datasets/{dataset_id}/preview`
- **Status Code**: `200 OK`
- **Query Parameters**:
  - `file_url` (string, optional): URL of the dataset. If not provided, the URL from the database is used.
  - `rows` (number, default: 100): Number of rows to return (max 500).
- **Response**: `DatasetPreviewResponse`
  ```json
  {
    "columns": ["string"],
    "rows": [{}],
    "row_count": number
  }
  ```

### List Weather Datasets
Retrieves all weather datasets for the authenticated user.

- **URL**: `GET /datasets/weather`
- **Status Code**: `200 OK`
- **Response**: `list[WeatherDataset]`
  ```json
  [
    {
      "id": "string",
      "dataset_id": "string",
      "file_url": "string",
      "uploadthing_key": "string",
      "created_at": "string"
    }
  ]
  ```

### Create Weather Dataset Record
Registers a new weather dataset linked to a consumption dataset.

- **URL**: `POST /datasets/weather`
- **Status Code**: `201 Created`
- **Request Body**: `WeatherDatasetCreate`
  ```json
  {
    "dataset_id": "string",
    "file_url": "string",
    "uploadthing_key": "string",
    "id": "string" (optional)
  }
  ```
- **Response**: `WeatherDataset` (same as above)

### Delete Weather Dataset
Removes a weather dataset record.

- **URL**: `DELETE /datasets/weather/{weather_dataset_id}`
- **Status Code**: `200 OK`
- **Response**:
  ```json
  {
    "status": "deleted"
  }
  ```

---

## 2. Preprocessing

Endpoints for managing data preprocessing jobs.

### Run Preprocessing
Starts a preprocessing job to prepare data for training.

- **URL**: `POST /preprocessing/run`
- **Status Code**: `201 Created`
- **Request Body**:
  ```json
  {
    "job_id": "string",
    "dataset_id": "string",
    "dataset_url": "string",
    "weather_url": "string" (optional),
    "splits": {
      "train": 0.7,
      "val": 0.15,
      "test": 0.15
    } (optional)
  }
  ```
- **Response**: `PreprocessingStatusResponse`
  ```json
  {
    "job_id": "string",
    "status": "PENDING" | "RUNNING" | "COMPLETE" | "FAILED",
    "progress": number,
    "error": "string" (optional),
    "processed_file_path": "string" (optional),
    "result_summary": {
      "feature_columns": ["string"],
      "train_rows": number,
      "val_rows": number,
      "test_rows": number,
      "processed_file_path": "string"
    },
    "eda_charts": {}
  }
  ```

### List Preprocessing Jobs
Retrieves all preprocessing jobs for the authenticated user.

- **URL**: `GET /preprocessing/jobs`
- **Status Code**: `200 OK`
- **Response**: `list[PreprocessingStatusResponse]`

### Get Preprocessing Status
Retrieves the status and result of a preprocessing job.

- **URL**: `GET /preprocessing/{job_id}/status`
- **Status Code**: `200 OK`
- **Response**: `PreprocessingStatusResponse` (same as above)

---

## 3. Models

Endpoints for training, evaluating, and managing machine learning models.

### List Models
Retrieves all trained models for the authenticated user.

- **URL**: `GET /models`
- **Status Code**: `200 OK`
- **Response**: `list[ModelTrainResponse]`

### Train Model
Starts a training job for a specific model type.

- **URL**: `POST /models/train`
- **Status Code**: `201 Created`
- **Request Body**:
  ```json
  {
    "job_id": "string",
    "preprocess_job_id": "string",
    "model_type": "RANDOM_FOREST" | "SVR",
    "hyperparams": {},
    "processed_file_path": "string" (optional)
  }
  ```
- **Response**: `ModelTrainResponse`
  ```json
  {
    "job_id": "string",
    "status": "PENDING" | "RUNNING" | "COMPLETE" | "FAILED",
    "model_file_path": "string",
    "scaler_file_path": "string",
    "training_time_secs": number,
    "error": "string"
  }
  ```

### Get Training Status
Retrieves the status of a training job.

- **URL**: `GET /models/jobs/{job_id}/status`
- **Status Code**: `200 OK`
- **Response**: `ModelTrainResponse` (same as above)

### Evaluate Model
Calculates performance metrics for a trained model on a test set.

- **URL**: `POST /models/{model_id}/evaluate`
- **Status Code**: `200 OK`
- **Request Body**:
  ```json
  {
    "model_run_id": "string",
    "model_file_path": "string",
    "processed_file_path": "string"
  }
  ```
- **Response**: `ModelEvaluateResponse`
  ```json
  {
    "rmse": number,
    "mae": number,
    "mape": number,
    "r2": number,
    "test_set_size": number,
    "actual": [{}],
    "predicted": [{}],
    "feature_importance": [{}]
  }
  ```

---

## 4. Forecast

Endpoints for generating and streaming predictions.

### List Forecasts
Retrieves all previous forecasts for the authenticated user.

- **URL**: `GET /forecasts`
- **Status Code**: `200 OK`
- **Response**: `list[ForecastRead]`
  ```json
  [
    {
      "id": "string",
      "user_id": "string",
      "engine": "RF" | "SVR" | "AGENT",
      "status": "PENDING" | "COMPLETE" | "FAILED",
      "start_date": "string",
      "horizon_days": number,
      "resolution": "HOURLY" | "DAILY" | "WEEKLY",
      "created_at": "string",
      "predictions": [{}],
      "reasoning": "string"
    }
  ]
  ```

### Create Forecast
Generates a forecast using a trained model or the AI Agent.

- **URL**: `POST /forecast`
- **Status Code**: `201 Created`
- **Request Body**:
  ```json
  {
    "engine": "RF" | "SVR" | "AGENT",
    "preprocess_job_id": "string",
    "model_run_id": "string" (optional),
    "start_date": "string" (YYYY-MM-DD),
    "horizon_days": number,
    "resolution": "HOURLY" | "DAILY" | "WEEKLY",
    "model_override": "string" (optional),
    "processed_file_path": "string" (optional)
  }
  ```
- **Response**: `ForecastResponse`
  ```json
  {
    "forecast_id": "string",
    "engine": "RF" | "SVR" | "AGENT",
    "predictions": [{}],
    "agent_reasoning": "string",
    "confidence": "string",
    "agent_run_id": "string"
  }
  ```

### Stream Agent Forecast (SSE)
Streams real-time updates from the AI Agent during forecast generation.

- **URL**: `GET /forecast/stream/{agent_run_id}`
- **Status Code**: `200 OK`
- **Response**: `text/event-stream`
  - Each event is a JSON string: `data: {"type": "step", "step": "string", "details": "string"}`

### Get Agent Status
Retrieves the final result or status of an agentic run.

- **URL**: `GET /agents/{agent_run_id}/status`
- **Status Code**: `200 OK`
- **Response**:
  ```json
  {
    "status": "PENDING" | "COMPLETE",
    "result": {} (only if COMPLETE)
  }
  ```

---

## 5. System

### Health Check
System health and version information.

- **URL**: `GET /health` (No `/api/v1` prefix)
- **Response**:
  ```json
  {
    "status": "ok",
    "version": "2.0.0"
  }
  ```
