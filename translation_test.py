import sys, os
from functools import partial
# Needed for exllamav2 lib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav2.generator import (
    ExLlamaV2Sampler
)
from exllama_generator_wrapper import load_model, generate_response_fold

from translation_library import japanese_three_shot_to_english, multishot_detect_language, multishot_is_japanese_binary_response

abs_path = os.path.abspath(sys.argv[1])
config, tokenizer, cache, generator = load_model(sys.argv[1])

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 1.0
settings.top_k = 1
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

combined_names = [
    ("裕司", "Hiroshi"),  # Hiroshi
    ("隆", "Takashi or Taro"),    # Takashi
    ("裕介", "Yusuke"),   # Yusuke
    ("健二", "Kenji"),    # Kenji
    ("翔太", "Shota"),    # Shota
    ("結衣", "Yui"),      # Yui
    ("桜", "Sakura"),     # Sakura
    ("愛子", "Aiko"),     # Aiko
    ("遥", "Haruka"),     # Haruka
    ("咲希", "Saki")      # Saki
]

generate = partial(generate_response_fold, generator=generator, settings=settings, max_length = 500)

for (jap_name, eng_name) in combined_names:
    print(jap_name)
    input = jap_name

    binary_resp = multishot_is_japanese_binary_response(generate, input)
    print(f"Is language japanese: {binary_resp}")
    
    language_resp = multishot_detect_language(generate, input)
    print(f"Predicted language: {language_resp}")
    
    resp = japanese_three_shot_to_english(generate, input)
    print(f"Predicted text: {resp}")
    
    print("\n\n")
    