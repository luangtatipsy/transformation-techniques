import re

from emot.emo_unicode import EMOTICONS
from th_preprocessor.preprocess import preprocess as simple_preprocess
from th_preprocessor.preprocess import remove_emoji

RE_EMOTICONS = re.compile(u"(" + u"|".join(k for k in EMOTICONS) + u")")


def remove_emoticons(text: str) -> str:
    return RE_EMOTICONS.sub(r"", text)


def preprocess(text: str) -> str:
    _text = remove_emoticons(text)
    _text = remove_emoji(_text)
    _text = simple_preprocess(_text)

    return _text
