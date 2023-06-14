from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import uvicorn
from routes import overloadPDF
import streamlit as st


def app():
    app = FastAPI(debug=True, docs_url="/docs", redoc_url=None, root_path=f"/")
    app.include_router(overloadPDF.router, prefix="/api")
    uvicorn.run(app, host="0.0.0.0", port=7700)


if __name__ == "__main__":
    app()
