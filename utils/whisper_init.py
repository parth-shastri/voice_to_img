"""Initialization of the speech-2-text model "whisper" by openai"""
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import config

# initialize the model
model_for_generation = WhisperForConditionalGeneration.from_pretrained(config.SPEECH_MODEL_NAME)

# initialize the model feature extractor
speech_feature_extractor = WhisperFeatureExtractor.from_pretrained(config.SPEECH_MODEL_NAME)

# initialize the tokenizer
whisper_tokenizer = WhisperTokenizer.from_pretrained(config.SPEECH_MODEL_NAME)

if config.MAKE_LOCAL:
    model_for_generation.save_pretrained("models/" + config.SPEECH_MODEL_NAME)


if __name__ == "__main__":
    print(model_for_generation)
