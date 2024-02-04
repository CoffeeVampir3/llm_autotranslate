import sys, os
# Needed for exllamav2 lib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav2.generator import (
    ExLlamaV2Sampler
)

from exllama_generator_wrapper import load_model, generate_response_fold

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

for (jap_name, eng_name) in combined_names:
    #print(f"{jap_name} should be translated as {eng_name}")
    print(jap_name)
    input = jap_name
    prompt = (
    f"""
    ::JAPANESE NAME:: さくら ::END NAME::
    ::ENGLISH NAME:: Sakura ::END NAME::

    ::JAPANESE NAME:: たけし ::END NAME::
    ::ENGLISH NAME:: Takeshi ::END NAME::

    ::JAPANESE NAME:: ゆき ::END NAME::
    ::ENGLISH NAME:: Yuki ::END NAME::

    ::JAPANESE NAME:: {input} ::END NAME::
    """
    )

    resp = generate_response_fold(prompt, generator, settings, max_length = 100, stop_sequences=["::END NAME"])
    name = resp.split("::")[2].strip()
    print(name)
    