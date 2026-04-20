# Frontend Integration Guide

This guide details how to connect your Windows/React frontend to the Wedding Image Culling API.

## API Endpoint
**Base URL**: `http://localhost:8000` (Default)

## Core Workflows

### 1. Health Check
Ping the server to ensure it's ready.
-   **GET** `/ping`
-   **Response**: `{"status": "ok"}`

### 2. Enrolling VIPs (Bride/Groom)
Before filtering, upload reference photos of the VIPs.
-   **POST** `/upload-profiles`
-   **Form Data**:
    -   `person`: Name (e.g., "Bride", "Groom").
    -   `zipfile_upload`: A standard ZIP file containing face images of that person.
-   **Response**: `{"enrolled": "Bride"}`

### 3. Running the Filter
Start the culling process on a local folder.
-   **POST** `/filter`
-   **JSON Body**:
    ```json
    {
      "input_folder": "C:\\Users\\Photographer\\Wedding2024",
      "workers": 4
    }
    ```
    *(Note: On Windows, use double backslashes `\\` for paths)*
-   **Response**:
    ```json
    {
      "output_folder": "C:\\Users\\Photographer\\Wedding2024\\output\\20240116_120000"
    }
    ```

### 4. Downloading Results
After filtering, retrieve the CSV report.
-   **GET** `/download-log?folder=C:\\Path\\To\\Output`
-   **Response**: A downloadable CSV file containing scores and rejection reasons for every image.

## Windows Specifics
-   **File Paths**: Ensure you send absolute Windows paths (e.g., `D:\Photos`). The Python backend handles them correctly using `os.path`.
-   **Performance**: If the user has a GPU (NVIDIA), ensure they have installed the CUDA toolkit so `torch` can use it. If not, the backend will automatically run on CPU (slower but functional).

## Error Handling
-   **400 Bad Request**: Typically invalid folder paths or ZIP files.
-   **500 Internal Server Error**: Check the Python console logs for stack traces.
