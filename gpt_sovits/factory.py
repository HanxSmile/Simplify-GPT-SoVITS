from gpt_sovits.common import registry
from omegaconf import OmegaConf


class Factory:

    @staticmethod
    def read_config(cfg_path):
        return OmegaConf.load(cfg_path)

    @staticmethod
    def build_model(cfg):
        model_cls = registry.get_model_class(cfg.model_cls)
        return model_cls.build_from_cfg(cfg)
