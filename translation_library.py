def japanese_binary_response(generate_fn):
    generate_fn()
    
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