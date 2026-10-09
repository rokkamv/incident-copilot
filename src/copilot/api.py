from fastapi import FastAPI

app = FastAPI(title="Incident Copilot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Stage 3 adds POST /triage here.
