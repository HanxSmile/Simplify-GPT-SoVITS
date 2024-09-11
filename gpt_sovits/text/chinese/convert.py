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

special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}


@registry.register_converter("chinese_converter")
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
    }

    LANG = "zh"

    def __init__(
            self,
            *args,
            g2p_model_dir="GPT_SoVITS/text/G2PWModel",
            g2p_tokenizer_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            **kwargs
    ):
        self.normalizer = TextNormalizer()
        self.tone_modifier = ToneSandhi()

        try:
            g2pw = G2PWPinyin(
                model_dir=g2p_model_dir,
                model_source=g2p_tokenizer_dir,
                v_to_u=False,
                neutral_tone_with_five=True
            )
        except Exception as e:
            print(e)
            g2pw = None
        self.g2pw = g2pw

    def normalize(self, text):
        if re.search(r'[A-Za-z]', text):
            text = re.sub(r'[a-z]', lambda x: x.group(0).upper(), text)
        sentences = self.normalizer.normalize(text)
        dest_text = ""
        for sentence in sentences:
            dest_text += self.replace_punctuation(sentence)

        # 避免重复标点引起的参考泄露
        dest_text = self.replace_consecutive_punctuation(dest_text)
        return dest_text

    def g2p(self, text):
        pattern = r"(?<=[{0}])\s*".format("".join(self.PUNCTUATIONS))
        sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
        phones, word2ph = self._g2p(sentences)
        return phones, word2ph

    @staticmethod
    def replace_punctuation(text):
        text = text.replace("嗯", "恩").replace("呣", "母")
        pattern = re.compile("|".join(re.escape(p) for p in Converter.REP_MAP.keys()))

        replaced_text = pattern.sub(lambda x: Converter.REP_MAP[x.group()], text)

        replaced_text = re.sub(
            r"[^\u4e00-\u9fa5" + "".join(Converter.PUNCTUATIONS) + r"]+", "", replaced_text
        )

        return replaced_text

    @staticmethod
    def _get_initials_finals(word):
        initials = []
        finals = []

        orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
        orig_finals = lazy_pinyin(
            word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
        )

        for c, v in zip(orig_initials, orig_finals):
            initials.append(c)
            finals.append(v)
        return initials, finals

    @staticmethod
    def _merge_erhua(
            initials: List[str],
            finals: List[str],
            word: str,
            pos: str
    ) -> List[List[str]]:
        """
        Do erhub.
        """
        must_erhua = {
            "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"
        }
        not_erhua = {
            "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
            "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
            "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
            "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
            "狗儿", "少儿"
        }
        # fix er1
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn == 'er1':
                finals[i] = 'er2'

        # 发音
        if word not in must_erhua and (word in not_erhua or
                                       pos in {"a", "j", "nr"}):
            return initials, finals

        # "……" 等情况直接返回
        if len(finals) != len(word):
            return initials, finals

        assert len(finals) == len(word)

        # 与前一个字发同音
        new_initials = []
        new_finals = []
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn in {
                "er2", "er5"
            } and word[-2:] not in not_erhua and new_finals:
                phn = "er" + new_finals[-1][-1]

            new_initials.append(initials[i])
            new_finals.append(phn)

        return new_initials, new_finals

    def _g2p(self, segments):
        phones_list = []
        word2ph = []
        for seg in segments:
            pinyins = []
            # Replace all English words in the sentence
            seg = re.sub("[a-zA-Z]+", "", seg)
            seg_cut = psg.lcut(seg)
            seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)
            initials = []
            finals = []

            if self.g2pw is None:
                for word, pos in seg_cut:
                    if pos == "eng":
                        continue
                    sub_initials, sub_finals = self._get_initials_finals(word)
                    sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
                    # 儿化
                    sub_initials, sub_finals = self._merge_erhua(sub_initials, sub_finals, word, pos)
                    initials.append(sub_initials)
                    finals.append(sub_finals)
                    # assert len(sub_initials) == len(sub_finals) == len(word)
                initials = sum(initials, [])
                finals = sum(finals, [])
                print("pypinyin结果", initials, finals)
            else:
                # g2pw采用整句推理
                pinyins = self.g2pw.lazy_pinyin(seg, neutral_tone_with_five=True, style=Style.TONE3)

                pre_word_length = 0
                for word, pos in seg_cut:
                    sub_initials = []
                    sub_finals = []
                    now_word_length = pre_word_length + len(word)

                    if pos == 'eng':
                        pre_word_length = now_word_length
                        continue

                    word_pinyins = pinyins[pre_word_length:now_word_length]

                    # 多音字消歧
                    word_pinyins = correct_pronunciation(word, word_pinyins)

                    for pinyin in word_pinyins:
                        if pinyin[0].isalpha():
                            sub_initials.append(to_initials(pinyin))
                            sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                        else:
                            sub_initials.append(pinyin)
                            sub_finals.append(pinyin)

                    pre_word_length = now_word_length
                    sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
                    # 儿化
                    sub_initials, sub_finals = self._merge_erhua(sub_initials, sub_finals, word, pos)
                    initials.append(sub_initials)
                    finals.append(sub_finals)

                initials = sum(initials, [])
                finals = sum(finals, [])
                # print("g2pw结果",initials,finals)

            for c, v in zip(initials, finals):
                raw_pinyin = c + v
                # NOTE: post process for pypinyin outputs
                # we discriminate i, ii and iii
                if c == v:
                    assert c in Converter.PUNCTUATIONS, f"'{c}' should be a punctuation"
                    phone = [c]
                    word2ph.append(1)
                else:
                    v_without_tone = v[:-1]
                    tone = v[-1]

                    pinyin = c + v_without_tone
                    assert tone in "12345"

                    if c:
                        # 多音节
                        v_rep_map = {
                            "uei": "ui",
                            "iou": "iu",
                            "uen": "un",
                        }
                        if v_without_tone in v_rep_map.keys():
                            pinyin = c + v_rep_map[v_without_tone]
                    else:
                        # 单音节
                        pinyin_rep_map = {
                            "ing": "ying",
                            "i": "yi",
                            "in": "yin",
                            "u": "wu",
                        }
                        if pinyin in pinyin_rep_map.keys():
                            pinyin = pinyin_rep_map[pinyin]
                        else:
                            single_rep_map = {
                                "v": "yu",
                                "e": "e",
                                "i": "y",
                                "u": "w",
                            }
                            if pinyin[0] in single_rep_map.keys():
                                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                    assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                    new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                    new_v = new_v + tone
                    phone = [new_c, new_v]
                    word2ph.append(len(phone))

                phones_list += phone
        return phones_list, word2ph

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
