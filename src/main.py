from fastapi import FastAPI
from src.api import router as main_router
from src.settings import settings
from src.middlewares import handle_exceptions_middleware

app = FastAPI(title=settings.TITLE)
app.middleware("http")(handle_exceptions_middleware)
app.include_router(main_router)