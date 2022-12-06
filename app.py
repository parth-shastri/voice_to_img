"""API routes and addresses"""
import numpy as np
import uvicorn
from enum import Enum
from fastapi import FastAPI, Form, File, UploadFile, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
import librosa
from utils.whisper_init import whisper_tokenizer, speech_feature_extractor, model_for_generation
from utils.diffusion_model_init import pipeline, DEVICE

templates = Jinja2Templates(directory="templates")

app = FastAPI(title="Voice to Image API")

class selectLanguages(str, Enum):
    english = "english"
    spanish = "spanish"
    french = "french"
    Hindi = "hindi"
    Marathi = "marathi"


@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    """display the HtML UI for the home page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "choices": [e.value for e in selectLanguages]}
        )


@app.post("/upload/")
def upload_files_and_params(
    my_file: UploadFile = File(...),
    language: selectLanguages = Form(selectLanguages.english),
):
    """Upload the language choice and the Audio file or recording"""
    file_like_audio = my_file.file
    audio, rate = librosa.load(file_like_audio, sr=16000)
    response = generate_img(audio, rate, language)
    return response


@app.get("/output/", response_class=HTMLResponse)
def display_output(
    request: Request,
    response: dict,
):
    """Display the generated image"""
    templates.TemplateResponse(
        "output.html",
        {"request": request, "prompt": response["prompt"], "image": "data/response_artifact.png"}
        )


def generate_img(
    audio: np.ndarray,
    rate: int,
    language: str):
    """Generate image using stable diffusion and whisper"""
    features = speech_feature_extractor(
        audio,
        return_tensors="pt",
        truncation=True,
        sampling_rate=rate
    )

    # Convert the audio array into a speech transcription using whisper
    mel_features = features.input_features
    # define task -> translate when audio is in any language other than "english"
    task = "translate" if language != "english" else "transcribe"
    # decoder_input_ids = whisper_tokenizer.get_decoder_prompt_ids(task=task, language="language")
    logits = model_for_generation.generate(
        mel_features,
        max_length=448,
        # forced_decoder_ids=decoder_input_ids
    )

    transcription = whisper_tokenizer.batch_decode(logits, skip_special_tokens=True)

    prompt = transcription[0]
    pipeline.enable_attention_slicing()
    sd_pipeline = pipeline.to(DEVICE)
    # image = sd_pipeline(prompt, height=512, width=512).images[0]
    # image.save("data/response_artifact.png")

    return {"prompt": prompt}


if __name__ == "__main__":
    uvicorn.run(app, port=8080)
