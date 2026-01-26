import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from agent import initialize_agent

# --- 1. Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources on startup."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print(" CRITICAL ERROR: OPENAI_API_KEY not found.")
        sys.exit(1)
    
    # Initialize Agent
    app.state.agent = initialize_agent()
    print(" NIFTY 50 AI Analyst Agent Initialized.")
    
    yield
 

# --- 2. Setup FastAPI ---
app = FastAPI(title="NIFTY 50 AI Analyst", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Data Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str

# --- 4. Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """Handle chat messages asynchronously without logging steps."""
    user_input = payload.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        agent = app.state.agent

        # Run the agent
        result = await agent.ainvoke({"input": user_input})
        
        # Extract output safely
        final_output = result["output"] if isinstance(result, dict) else str(result)

        return {"response": final_output}

    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"response": "⚠️ Internal Error: Unable to process request."}
        )

# --- 5. Run Server ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
