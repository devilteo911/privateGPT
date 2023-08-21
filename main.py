from fastapi import FastAPI
import uvicorn
from routes import overloadPDF


def app():
    app = FastAPI(debug=True, docs_url="/docs", redoc_url=None, root_path="/")
    app.include_router(overloadPDF.router, prefix="/api")
    uvicorn.run(app, host="0.0.0.0", port=7700)


if __name__ == "__main__":
    app()
