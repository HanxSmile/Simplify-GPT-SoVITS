from gpt_sovits.text.chinese import TextNormalizer
from gpt_sovits.text.chinese.g2pw import G2PWPinyin, correct_pronunciation
from gpt_sovits.text.chinese.tone_sandhi import ToneSandhi
from gpt_sovits.text.base_converter import BaseConverter
import jieba_fast.posseg as psg
from pypinyin import lazy_pinyin, Style
import re
import os
from typing import List
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
from gpt_sovits.common.registry import registry
from builtins import str as unicode
from g2p_en.expand import normalize_numbers
import unicodedata
import LangSegment
import wordsegment


@registry.register_converter("english_converter")
class Converter(BaseConverter):
    LANG = "en"

    def __init__(
            self,
    ):
        pass

    def normalize(self, text):
        # todo: eng text normalize
        # 适配中文及 g2p_en 标点
        LangSegment.setfilters(["en"])
        text = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        rep_map = {
            "[;:：，；]": ",",
            '["’]': "'",
            "。": ".",
            "！": "!",
            "？": "?",
        }
        for p, r in rep_map.items():
            text = re.sub(p, r, text)

        # 来自 g2p_en 文本格式化处理
        # 增加大写兼容
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = re.sub("[^ A-Za-z'.,?!\-]", "", text)
        text = re.sub(r"(?i)i\.e\.", "that is", text)
        text = re.sub(r"(?i)e\.g\.", "for example", text)

        # 避免重复标点引起的参考泄露
        text = Converter.replace_consecutive_punctuation(text)
        return text

    def g2p(self, text):
        pattern = r"(?<=[{0}])\s*".format("".join(self.PUNCTUATIONS))
        sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
        phones, word2ph = self._g2p(sentences)
        return phones, word2ph



    @classmethod
    def build_from_cfg(cls, cfg):
        g2p_model_dir = cfg.get("g2p_model_dir", "GPT_SoVITS/text/G2PWModel")
        g2p_tokenizer_dir = cfg.get("g2p_tokenizer_dir", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
        converter = cls(
            g2p_model_dir=g2p_model_dir,
            g2p_tokenizer_dir=g2p_tokenizer_dir,
        )
        return converter


if __name__ == '__main__':
    converter = Converter(g2p_model_dir="/Users/hanxiao/Downloads/G2PWModel_1.1")
    text = "啊——但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏"
    norm_text = converter.normalize(text)
    print(norm_text)
    print(converter.g2p(norm_text))
