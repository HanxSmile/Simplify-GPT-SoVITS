class BaseConverter:

    def normalize(self, text):
        raise NotImplementedError

    def g2p(self, text):
        raise NotImplementedError

    @classmethod
    def build_from_cfg(cls, cfg):
        raise NotImplementedError
