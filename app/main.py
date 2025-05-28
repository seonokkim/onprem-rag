from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from app.database.vector_store import VectorStore
from app.services.synthesizer import Synthesizer

# Define Pydantic model for chat request body
class ChatRequest(BaseModel):
    query: str

app = FastAPI(title="Stock Trading FAQ Assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Initialize VectorStore
vec = VectorStore()

# Categories
CATEGORIES = [
    "All",
    "Trading Strategies",
    "Trading Strategy",
    "Trading Basics",
    "Trading Concepts",
    "Trading Tools",
    "Trading Options",
    "Technical Analysis",
    "Fundamental Analysis",
    "Market Indicators",
    "Market Structure",
    "Market Function",
    "Market Theory",
    "Market Benchmarks",
    "Market Cycles",
    "Market Anomalies",
    "Risk Management",
    "Order Types",
    "Brokerage Accounts",
    "Options Strategy",
    "Portfolio Management",
    "Investment Options",
    "Investment Types",
    "Company Analysis",
    "Financial Metrics",
    "Price Metrics",
    "Earnings Analysis",
    "Economic Impact",
    "Industry Basics",
    "Corporate Finance",
    "Derivatives",
    "Dividend Types",
    "International Investing",
    "International Risk",
    "International Access",
    "Account Types",
    "Advanced Trading",
    "Advanced Strategy",
    "Advanced Risk",
    "High-Risk Investing",
    "Employee Benefits",
    "Tax Rules",
    "Wealth Building",
    "Research Tools",
    "Analysis Methods"
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": CATEGORIES
        }
    )

@app.post("/api/search")
async def search(
    query: str,
    category: str = "All",
    use_date_filter: bool = False,
    start_date: datetime = None,
    end_date: datetime = None
):
    # Prepare search parameters
    search_params = {
        "limit": 3,
        "time_range": (start_date, end_date) if use_date_filter else None
    }
    
    # Add category filter if selected
    if category != "All":
        search_params["metadata_filter"] = {"category": category}
    
    # Perform search
    results = vec.search(query, return_dataframe=False, **search_params)
    
    # Generate response
    response = Synthesizer.generate_response(question=query, context=results)
    
    return {
        "answer": response.answer,
        "thought_process": response.thought_process,
        "enough_context": response.enough_context,
        "results": [
            {
                "content": result[2],
                "metadata": result[1],
                "similarity": result[4]
            }
            for result in results
        ]
    }

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    # Prepare search parameters
    search_params = {
        "limit": 3
    }
    
    # Perform search
    results = vec.search(request.query, return_dataframe=False, **search_params)
    
    # Generate response
    response = Synthesizer.generate_response(question=request.query, context=results)
    
    return {
        "answer": response.answer,
        "thought_process": response.thought_process,
        "enough_context": response.enough_context,
        "results": [
            {
                "content": result[2],
                "metadata": result[1],
                "similarity": result[4]
            }
            for result in results
        ]
    } 