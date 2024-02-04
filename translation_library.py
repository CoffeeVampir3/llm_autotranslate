def multishot_detect_language(generate_fn, input):
    prompt = (
f"""::INPUT TEXT::
さくら
::END TEXT::
::PREDICTED LANGUAGE::
Japanese
::END PREDICTION::

::INPUT TEXT::
ゆき
::END TEXT::
::ENGLISH TEXT::
Japanese
::END PREDICTION::

::INPUT TEXT::
1月下旬、大手カレーチェーン「カレーハウスCoCo (ココ）壱(いち)番屋」のキッチンカーがやってきたのは、24人が避難する石川県七尾市の能登島地区コミュニティセンターだ。量や辛さの希望を社員が聞き取り、ご飯をよそった容器のふたの上に温かいレトルトパウチを載せて渡す。
::END TEXT::
::PREDICTED LANGUAGE::
Japanese
::END PREDICTION::

::INPUT TEXT::
The goat
::END TEXT::
::PREDICTED LANGUAGE::
English
::END PREDICTION::

::INPUT TEXT::
안녕하세요
::END TEXT::
::PREDICTED LANGUAGE::
Korean
::END PREDICTION::

::INPUT TEXT::
Wie geht es Ihnen?
::END TEXT::
::PREDICTED LANGUAGE::
German
::END PREDICTION::

::INPUT TEXT::
{input}
::END TEXT::
::PREDICTED LANGUAGE::""")
    
    resp = generate_fn(prompt=prompt, stop_sequences=["::END PREDICTION"])
    text = resp.strip()
    return text

def multishot_is_language_binary_response(generate_fn, language_name, affirmative_example, input):
    prompt = (
f"""{affirmative_example}

::TEXT::
Hello, how are you?
::END TEXT::
::IS TEXT {language_name}::
No
::END::

::TEXT::
안녕하세요
::END TEXT::
::IS TEXT {language_name}::
No
::END::

::TEXT::
def fn(prompt):
    return True
::END TEXT::
::IS TEXT {language_name}::
No
::END::

::TEXT::
{input}
::END TEXT::
::IS TEXT {language_name}::""")
    
    resp = generate_fn(prompt=prompt, stop_sequences=["::END"])
    text = resp.strip()
    return text

def multishot_is_japanese_binary_response(generate_fn, input):
    affirmative_example = (
f"""::TEXT::
さくら
::END TEXT::
::IS TEXT JAPANESE::
Yes
::END::""")
    return multishot_is_language_binary_response(generate_fn, "JAPANESE", affirmative_example, input)
    
def japanese_three_shot_to_english(generate_fn, input):
    prompt = (
f"""::JAPANESE TEXT::
さくら
::END TEXT::
::ENGLISH TEXT::
Sakura
::END TEXT::

::JAPANESE TEXT::
ゆき
::END TEXT::
::ENGLISH TEXT::
Yuki
::END TEXT::

::JAPANESE TEXT::
1月下旬、大手カレーチェーン「カレーハウスCoCo (ココ）壱(いち)番屋」のキッチンカーがやってきたのは、24人が避難する石川県七尾市の能登島地区コミュニティセンターだ。量や辛さの希望を社員が聞き取り、ご飯をよそった容器のふたの上に温かいレトルトパウチを載せて渡す。
::END TEXT::
::ENGLISH TEXT::
In late January, the kitchen car from the major curry chain "Curry House CoCo Ichibanya" arrived at a community center in the Notojima area of Nanao City, Ishikawa Prefecture, where 24 people were taking shelter. The staff took orders for the desired amount and spiciness of the curry, then served it by placing a warm retort pouch on top of a lid covering a container filled with rice.
::END TEXT:

::JAPANESE TEXT::
{input}
::END TEXT::
::ENGLISH TEXT::""")
    
    resp = generate_fn(prompt=prompt, stop_sequences=["::END TEXT"])
    text = resp.strip()
    return text