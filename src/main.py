from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import router as main_router
from src.settings import settings
from src.middlewares import handle_exceptions_middleware

app = FastAPI(title=settings.TITLE)
app.middleware("http")(handle_exceptions_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(main_router)