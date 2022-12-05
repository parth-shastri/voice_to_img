"""Initialization of the speech-2-text model "whisper" by openai"""
import os
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import utils.config as config

LOCAL_DIR = "models/" + config.SPEECH_MODEL_NAME

# initialize the model
model_for_generation = WhisperForConditionalGeneration.from_pretrained(
    config.SPEECH_MODEL_NAME if not os.path.exists(LOCAL_DIR) else LOCAL_DIR
    )

# initialize the model feature extractor
speech_feature_extractor = WhisperFeatureExtractor.from_pretrained(
    config.SPEECH_MODEL_NAME
    )

# initialize the tokenizer
whisper_tokenizer = WhisperTokenizer.from_pretrained(
    config.SPEECH_MODEL_NAME
    )

if config.MAKE_LOCAL and not os.path.exists(LOCAL_DIR):
    model_for_generation.save_pretrained("models/" + config.SPEECH_MODEL_NAME)


if __name__ == "__main__":
    print(model_for_generation)
    
