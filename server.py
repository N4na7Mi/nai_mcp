from fastapi import FastAPI
import os
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"pong": True}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
