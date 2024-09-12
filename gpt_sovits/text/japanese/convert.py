from gpt_sovits.text.base_converter import BaseConverter
from gpt_sovits.common.registry import registry
from gpt_sovits.text.japanese.utils import preprocess_jap, post_replace_ph


@registry.register_converter("japanese_converter")
class Converter(BaseConverter):
    LANG = "ja"

    def __init__(
            self,
            with_prosody=True
    ):
        self.with_prosody = with_prosody

    def normalize(self, text):
        # todo: jap text normalize
        text = Converter.replace_consecutive_punctuation(text)
        return text

    def g2p(self, text):
        phones = preprocess_jap(text, self.with_prosody)
        phones = [post_replace_ph(i) for i in phones]
        return phones, None

    @classmethod
    def build_from_cfg(cls, cfg):
        with_prosody = cfg.get("with_prosody", True)
        return cls(with_prosody)


if __name__ == '__main__':
    converter = Converter(True)
    phones = converter.g2p("こんにちは, hello, AKITOです,よろしくお願いしますね！")
    print(phones)
