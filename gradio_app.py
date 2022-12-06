import gradio as gr
import librosa
from utils.whisper_init import whisper_tokenizer, speech_feature_extractor, model_for_generation
from utils.diffusion_model_init import pipeline, DEVICE


def generate_image(audio_file, language):
    """Generate image using stable diffusion with thw audio file received"""
    audio, rate = librosa.load(audio_file, sr=16000)
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
    images = sd_pipeline(prompt, height=512, width=512).images
    return images
    