from gpt_sovits.text.chinese import TextNormalizer
from gpt_sovits.text.chinese.tone_sandhi import ToneSandhi
from gpt_sovits.text.base_converter import BaseConverter
import re
from gpt_sovits.common.registry import registry
from gpt_sovits.text.chinese.chn_text_norm.text import Text


@registry.register_converter("chinese_fs_converter")
class Converter(BaseConverter):
    REP_MAP = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "$": ".",
        "/": ",",
        "—": "-",
        "~": "…",
        "～": "…",
        '"': '',
        "'": '',
        '“': '',
        '”': '',
    }

    LANG = "zh"

    def __init__(
            self,
    ):
        self.normalizer = TextNormalizer()
        self.tone_modifier = ToneSandhi()

    def normalize(self, text):
        if re.search(r'[A-Za-z]', text):
            text = re.sub(r'[a-z]', lambda x: x.group(0).upper(), text)
        sentences = self.normalizer.normalize(text)
        dest_text = ""
        for sentence in sentences:
            dest_text += self.replace_punctuation(sentence)

        # 避免重复标点引起的参考泄露
        dest_text = self.replace_consecutive_punctuation(dest_text)
        dest_text = self.replace_punctuation(dest_text)
        dest_text = Text(dest_text).normalize()
        return dest_text

    @staticmethod
    def replace_punctuation(text):
        text = text.replace("嗯", "恩").replace("呣", "母")
        pattern = re.compile("|".join(re.escape(p) for p in Converter.REP_MAP.keys()))

        replaced_text = pattern.sub(lambda x: Converter.REP_MAP[x.group()], text)

        replaced_text = re.sub(
            r"[^\u4e00-\u9fa5" + "".join(Converter.PUNCTUATIONS) + r"]+", "", replaced_text
        )

        return replaced_text

    @classmethod
    def build_from_cfg(cls, cfg, *args, **kwargs):
        converter = cls()
        return converter


if __name__ == '__main__':
    converter = Converter()
    text = "啊——但是《原神》是由,米哈\游自主，“研发”的一款全.新开放世界.冒险游戏"
    print(Text(text).normalize())
    norm_text = converter.normalize(text)
    print(norm_text)
