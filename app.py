import os
import sys
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain.callbacks.base import BaseCallbackHandler

# Import our modular agent
from agent import initialize_agent

# 1. Load Environment
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found.")
    sys.exit(1)

# 2. Setup FastAPI
app = FastAPI(title="NIFTY 50 AI Analyst")
templates = Jinja2Templates(directory="templates")

# 3. Initialize Agent
# We initialize it once at startup
executor = initialize_agent()

# 4. Define Data Models (Pydantic)
class ChatRequest(BaseModel):
    message: str

# 5. Logging Handler
class ThinkingLogHandler(BaseCallbackHandler):
    """Captures tool usage to show 'Thinking' logs."""
    def __init__(self):
        self.logs = []
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.logs.append(f"🛠️ Using tool: {serialized.get('name')}...")

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the Chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """Handle chat messages."""
    user_input = payload.message
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    log_handler = ThinkingLogHandler()
    
    try:
        # FastAPI runs synchronous code (like our agent) in a threadpool automatically,
        # so this won't block the server.
        response = executor.invoke(
            {"input": user_input},
            config={"callbacks": [log_handler]}
        )
        
        return {
            "response": response["output"],
            "logs": log_handler.logs
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}", 
            "logs": []
        }

# 6. Run Server (Dev Mode)
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)