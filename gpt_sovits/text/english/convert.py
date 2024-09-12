from gpt_sovits.text.base_converter import BaseConverter
import re
from gpt_sovits.common.registry import registry
from builtins import str as unicode
from g2p_en.expand import normalize_numbers
import unicodedata
import LangSegment
from gpt_sovits.text.english.utils import en_G2p


@registry.register_converter("english_converter")
class Converter(BaseConverter):
    LANG = "en"

    def __init__(
            self,
    ):
        self._g2p = en_G2p()

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
        phone_list = self._g2p(text)
        phones = [ph if ph != "<unk>" else "UNK" for ph in phone_list if ph not in [" ", "<pad>", "UW", "</s>", "<s>"]]
        return phones, None

    @classmethod
    def build_from_cfg(cls, cfg):
        return cls()


if __name__ == '__main__':
    converter = Converter()
    print(converter.g2p("hello"))
    print(converter.g2p(converter.normalize("e.g. I used openai's AI tool to draw a picture.")))
    print(converter.g2p(converter.normalize("In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.")))
