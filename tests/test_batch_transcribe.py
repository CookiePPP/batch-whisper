import os
import glob
import pytest
from jiwer import cer
import batch_whisper
import time


def normalize_text(text: str) -> str:
    return text.lower()

def mean(l: list):
    return sum(l) / len(l)

@pytest.mark.parametrize('model_name', ['small.en'])
def test_batch_transcribe(model_name: str):
    model = batch_whisper.load_model(model_name).cuda()
    audio_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "audios", "*.opus")))
    caption_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "audios", "*.txt")))
    
    # load whisper captions
    start_time = time.time()
    language = "en" if model_name.endswith(".en") else None
    result_list = model.transcribe(audio_paths, language=language, temperature=0.0)
    pred_captions = [result["text"] for result in result_list]
    time_taken = time.time() - start_time
    
    # load real captions
    real_captions = [open(path, "r", encoding="utf-8").read() for path in caption_paths]
    
    # calculate CER
    pred_captions = [normalize_text(caption) for caption in pred_captions]
    real_captions = [normalize_text(caption) for caption in real_captions]
    avg_cer = mean([cer(real_caption, pred_caption) for real_caption, pred_caption in zip(real_captions, pred_captions)])
    
    assert avg_cer < 0.12, f'Average CER is {avg_cer}'
    
    speed = len(audio_paths) / time_taken
    print(f'{model_name} speed: {speed:.2f} audios per second')


def num_punctuations(text: str) -> int:
    return sum([1 for char in text if char in [".", ",", "!", "?"]])

@pytest.mark.parametrize('model_name', ['small.en'])
def test_batch_transcribe_initial_prompt(model_name: str):
    model = batch_whisper.load_model(model_name).cuda()
    audio_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "audios", "*.opus")))
    
    language = "en" if model_name.endswith(".en") else None
    
    # run with prompt that has no punctuation
    initial_prompt = "hmmm so uhmm yeah thats the idea"
    result_list = model.transcribe(audio_paths, language=language, temperature=0.0, initial_prompt=initial_prompt, condition_on_previous_text=False)
    pred_captions_1 = [result["text"] for result in result_list]
    
    # run with prompt that has punctuation
    initial_prompt = "Hmmm. So... uhmm, yeah? That's the idea!"
    result_list = model.transcribe(audio_paths, language=language, temperature=0.0, initial_prompt=initial_prompt, condition_on_previous_text=False)
    pred_captions_2 = [result["text"] for result in result_list]
    
    # calculate number of punctuations for each run
    avg_num_puncs_1 = mean([num_punctuations(caption) for caption in pred_captions_1])
    avg_num_puncs_2 = mean([num_punctuations(caption) for caption in pred_captions_2])
    
    print(pred_captions_1)
    print(pred_captions_2)
    print(avg_num_puncs_2, avg_num_puncs_1)
    
    # We expect that when the model is told to use punctuation, it will use more punctuation
    assert avg_num_puncs_2 > avg_num_puncs_1, f'Average number of punctuations is {avg_num_puncs_2} with prompt and {avg_num_puncs_1} without prompt'