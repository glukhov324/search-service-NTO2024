import uvicorn
from src.settings import settings

def run():
    uvicorn.run(
        app="src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEV_MODE
    )

if __name__ == "__main__":
    run()