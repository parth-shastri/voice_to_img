"""API routes and addresses"""
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import librosa
from utils.whisper_init import whisper_tokenizer, speech_feature_extractor, model_for_generation
from utils.diffusion_model_init import pipeline

templates = Jinja2Templates(directory="templates")

app = FastAPI(title="Voice to Image API")

@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    """display the HtML UI for the home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
def upload_files_and_params(
    language: str = Form(...),
    my_file: UploadFile = File(...)
) -> dict:
    """Upload the language choice and the Audio file or recording"""
    audio, rate = librosa.load(my_file.file, sr=16000)
    features = speech_feature_extractor(
        audio,
        return_tensors="pt",
        truncation=True,
        sampling_rate=rate
    )
    mel_features = features.input_features
    # define task
    task = "translate" if language != "english" else "transcribe"
    decoder_input_ids = whisper_tokenizer.get_decoder_prompt_ids(task=task, language="language")
    logits = model_for_generation.generate(
        mel_features,
        forced_decoder_ids=decoder_input_ids
    )

    transcription = whisper_tokenizer.batch_decode(logits, skip_special_tokens=True)

    prompt = transcription[0]
    return {"prompt": prompt}


@app.get("/predict/")
def generate_img():
    pass


if __name__ == "__main__":
    uvicorn.run(app, port=8080)
