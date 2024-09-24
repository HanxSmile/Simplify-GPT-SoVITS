import logging


class Registry:
    mapping = {
        "converter_name_mapping": {},
        "model_name_mapping": {},
    }

    @classmethod
    def register_model(cls, name):
        def wrap(model_cls):
            if name in cls.mapping["model_name_mapping"]:
                logging.warning(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def register_converter(cls, name):
        def wrap(converter_cls):
            from gpt_sovits.text.base_converter import BaseConverter

            assert issubclass(
                converter_cls, BaseConverter
            ), "All converters must inherit BaseConverter class"
            if name in cls.mapping["converter_name_mapping"]:
                logging.warning(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["converter_name_mapping"][name]
                    )
                )
            cls.mapping["converter_name_mapping"][name] = converter_cls
            return converter_cls

        return wrap

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_converter_class(cls, name):
        return cls.mapping["converter_name_mapping"].get(name, None)


registry = Registry()
