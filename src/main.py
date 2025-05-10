from fastapi import FastAPI
from src.api import router as main_router
from src.settings import settings

app = FastAPI(title=settings.TITLE)
app.include_router(main_router)