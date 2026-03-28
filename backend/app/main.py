from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import meetings
from app.routers import transcribe
from app.routers import extract
from app.routers import compose

# Create FastAPI app
app = FastAPI(
    title="MeetMemo API",
    version="1.0.0",
    description="Personal productivity tool for transcribing meetings and generating follow-up emails",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
# In production, replace "*" with specific frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(meetings.router)
app.include_router(transcribe.router)
app.include_router(extract.router)
app.include_router(compose.router)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "service": "meetmemo-api"}
