from gpt_sovits.text.base_converter import BaseConverter
import re
import os
from gpt_sovits.common.registry import registry
from .cv_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number
import inflect

try:
    import ttsfrd

    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer

    use_ttsfrd = False


@registry.register_converter("chinese_cv_converter")
class Converter(BaseConverter):
    LANG = "zh"

    def __init__(self):
        self.inflect_parser = inflect.engine()
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyin')
            self.frd.enable_pinyin_mix(True)
            self.frd.set_breakmodel_index(1)
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False)
            self.en_tn_model = EnNormalizer()

    def normalize(self, text):
        text = text.strip()
        if contains_chinese(text):
            if self.use_ttsfrd:
                text = self.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，,、]+$', '。', text)
        else:
            if self.use_ttsfrd:
                text = self.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
        return text

    @classmethod
    def build_from_cfg(cls, cfg, *args, **kwargs):
        converter = cls()
        return converter
