# Exact copy of torchtext's basic_english tokenizer
# Source: https://github.com/pytorch/text/blob/main/torchtext/data/utils.py
import re

_patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]
_replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]
_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

def _basic_english_normalize(line):
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()

def basic_english_tokenizer(text):
    """Tokenizer matching torchtext's basic_english implementation"""
    return _basic_english_normalize(text)