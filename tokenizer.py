from copy import copy
from typing import List

from newmm_tokenizer.tokenizer import word_tokenize


def tokenize(text: str, min_char: int = 2, remove_placeholder: bool = False) -> List[str]:
    tokens = word_tokenize(text, keep_whitespace=False)
    
    if remove_placeholder:
        tokens = [token for token in tokens if not token.startswith("WS")]
    
    return [token.strip() for token in tokens if len(token.strip()) >= min_char]
