"""API routes and addresses"""
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


app = FastAPI()

@app.get("/index/")
def main():
    """display the HtML UI for the home page"""
    content = """
    """

    return HTMLResponse(content=content)


@app.post("/upload/")
def upload_files_and_params(
    language: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload the language choice and the Audio file or recording"""
    pass