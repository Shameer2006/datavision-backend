"""
DataVision AI - FastAPI Backend
Handles file uploads, schema extraction, AI chat, and Plotly chart generation.
"""

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.data_processor import extract_schema
from services.ai_service import generate_chat_response

# 50 MB limit
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB in bytes

app = FastAPI(
    title="DataVision AI Backend",
    description="Backend API for DataVision AI chat with file analysis and Plotly chart generation.",
    version="1.0.0",
)

# CORS configuration — allow local development and production frontends
allowed_origins = ["http://localhost:3000", "http://localhost:3001"]
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    # Ensure it doesn't have a trailing slash
    allowed_origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "DataVision AI Backend is running"}


@app.post("/api/chat")
async def chat(
    message: str = Form(default=""),
    file: UploadFile | None = File(default=None),
    cached_schema: str = Form(default=""),
    cached_df_json: str = Form(default=""),
):
    """
    Chat endpoint that processes file uploads and messages,
    returning a JSON response with AI-generated insights and optional Plotly charts.
    Accepts cached schema/data for follow-up messages in the same chat session.
    """
    schema = None
    df_json = None

    if file and file.filename:
        # New file uploaded — extract fresh schema and data
        filename = file.filename or "unknown"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ("csv", "xls", "xlsx"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: .{ext}. Please upload a .csv, .xls, or .xlsx file.",
            )

        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum allowed size is 50 MB. Your file is {len(file_bytes) / (1024 * 1024):.1f} MB.",
            )

        try:
            schema, df = extract_schema(file_bytes, filename)
            df_json = df.head(50).fillna("").to_json(orient="records", default_handler=str)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    elif cached_schema:
        # No new file — reuse cached data from previous upload
        import json
        try:
            schema = json.loads(cached_schema)
            df_json = cached_df_json if cached_df_json else None
        except json.JSONDecodeError:
            pass  # Ignore invalid cached data, proceed without it

    # Determine the message to send
    current_message = message
    if not current_message.strip() and not schema:
        raise HTTPException(status_code=400, detail="Please provide a message or upload a file.")

    if not current_message.strip() and schema:
        current_message = "Describe this dataset — what columns and data does it contain?"

    # Generate AI response
    result = await generate_chat_response(
        user_message=current_message,
        schema=schema,
        df_json=df_json,
    )

    # Include schema and df_json in response so the frontend can cache them
    if schema:
        import json
        result["cached_schema"] = json.dumps(schema, default=str)
        result["cached_df_json"] = df_json or ""

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

