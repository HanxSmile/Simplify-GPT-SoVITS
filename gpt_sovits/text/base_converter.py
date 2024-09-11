import re


class BaseConverter:
    PUNCTUATIONS = ["!", "?", "…", ",", "."]  # @是SP停顿

    def normalize(self, text):
        raise NotImplementedError

    def g2p(self, text):
        raise NotImplementedError

    @staticmethod
    def replace_consecutive_punctuation(text):
        punctuations = ''.join(re.escape(p) for p in BaseConverter.PUNCTUATIONS)
        pattern = f'([{punctuations}])([{punctuations}])+'
        result = re.sub(pattern, r'\1', text)
        return result

    @classmethod
    def build_from_cfg(cls, cfg):
        raise NotImplementedError
