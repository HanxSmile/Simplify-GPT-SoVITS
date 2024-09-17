import re
import torch
from gpt_sovits.text.symbols import symbols
from gpt_sovits.text.cutter import Cutter

_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols)}


class TextProcessor:
    SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
    PUNCTUATIONS = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}

    def __init__(
            self,
            converter,
            bert_model,
            tokenizer,
            cut_method,
    ):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.converter = converter
        self.cutter = Cutter(cut_method)

    def process(self, text, device):
        texts = self.segment_text(text)
        result = []
        for text in texts:
            norm_text = self.normalize_text(text)
            phones, word2ph = self.g2p_text(norm_text)
            phones = self.cleaned_text_to_sequence(phones)
            bert_feature = self.get_bert_feature(norm_text, phones, word2ph, device)
            item = {
                "phones": phones,
                "bert_feature": bert_feature,
                "norm_text": norm_text,
            }
            result.append(item)
        return result

    def process_single(self, text, device):
        text = "".join(self.segment_text(text))
        norm_text = self.normalize_text(text)
        phones, word2ph = self.g2p_text(norm_text)
        phones = self.cleaned_text_to_sequence(phones)
        bert_feature = self.get_bert_feature(norm_text, phones, word2ph, device)
        return norm_text, phones, bert_feature

    def normalize_text(self, text):
        return self.converter.normalize(text)

    def g2p_text(self, text):
        phones, word2ph = self.converter.g2p(text)
        return phones, word2ph

    def segment_text(self, text):
        """
        切分长文本为子文本列表
        :param text:
        :return:
        """
        text = text.strip()
        text = self.replace_consecutive_punctuation(text)
        text_segs = self.cutter(text)
        text_segs = self.merge_short_text_in_array(text_segs, 5)
        result = []
        for text_seg in text_segs:
            if not re.sub("\W+", "", text_seg):
                # 检测一下，如果是纯符号，就跳过。
                continue
            if text_seg[-1] not in self.SPLITS:
                text_seg += "。"
            if len(text_seg) > 510:
                result.extend(self.split_big_text(text_seg))
            else:
                result.append(text_seg)
        return result

    def get_bert_feature(self, text: str, phones: list, word2ph: list, device) -> torch.Tensor:
        if word2ph is None:
            return torch.zeros((1024, len(phones)), dtype=torch.float32).to(device)

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T.to(device)

    @staticmethod
    def merge_short_text_in_array(texts: list, threshold: int) -> list:
        if len(texts) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if len(text) > 0:
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result

    @staticmethod
    def split_big_text(text, max_len=510):
        # 定义全角和半角标点符号
        punctuation = "".join(TextProcessor.SPLITS)

        # 切割文本
        segments = re.split('([' + punctuation + '])', text)

        # 初始化结果列表和当前片段
        result = []
        current_segment = ''

        for segment in segments:
            # 如果当前片段加上新的片段长度超过max_len，就将当前片段加入结果列表，并重置当前片段
            if len(current_segment + segment) > max_len:
                result.append(current_segment)
                current_segment = segment
            else:
                current_segment += segment

        # 将最后一个片段加入结果列表
        if current_segment:
            result.append(current_segment)

        return result

    @staticmethod
    def replace_consecutive_punctuation(text):
        punctuations = ''.join(re.escape(p) for p in TextProcessor.PUNCTUATIONS)
        pattern = f'([{punctuations}])([{punctuations}])+'
        result = re.sub(pattern, r'\1', text)
        return result

    @staticmethod
    def cleaned_text_to_sequence(cleaned_text):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
          Args:
            text: string to convert to a sequence
          Returns:
            List of integers corresponding to the symbols in the text
        '''

        phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]

        return phones
