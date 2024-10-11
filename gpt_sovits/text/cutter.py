class Cutter:
    PUNCTUATION = {'!', '?', '…', ',', '.', '-', " "}
    SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

    def __init__(self, method):
        self.all_methods = [_ for _ in dir(self) if callable(getattr(self, _)) and _.startswith("cut")]
        assert method in self.all_methods, f"Invalid method '{method}'. You can only choose in '{self.all_methods}'!"
        self._method = method

    def __call__(self, *args, **kwargs):
        return getattr(self, self._method)(*args, **kwargs)

    @staticmethod
    def split(todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in Cutter.SPLITS:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if todo_text[i_split_head] in Cutter.SPLITS:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    @staticmethod
    def cut0(inp):
        """
        不切分
        :param inp:
        :return:
        """
        if not set(inp).issubset(Cutter.PUNCTUATION):
            if inp.strip():
                return [inp.strip()]
            return []
        else:
            return []

    @staticmethod
    def cut1(inp: str):
        """
        凑四句一切
        :param inp:
        :return:
        """
        inp = inp.strip("\n")
        inps = Cutter.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
        else:
            opts = [inp]
        opts = [item.strip() for item in opts if not set(item).issubset(Cutter.PUNCTUATION)]
        opts = [_ for _ in opts if _]
        return opts

    @staticmethod
    def cut2(inp):
        """
        凑50字一切
        :param inp:
        :return:
        """
        inp = inp.strip("\n")
        inps = Cutter.split(inp)
        if len(inps) < 2:
            return inp
        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
        if tmp_str != "":
            opts.append(tmp_str)
        # print(opts)
        if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
        opts = [item.strip() for item in opts if not set(item).issubset(Cutter.PUNCTUATION)]
        opts = [_ for _ in opts if _]
        return opts

    @staticmethod
    def cut3(inp):
        """
        按中文句号。切
        :param inp:
        :return:
        """
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip("。").split("。")]
        opts = [item.strip() for item in opts if not set(item).issubset(Cutter.PUNCTUATION)]
        opts = [_ for _ in opts if _]
        return opts

    @staticmethod
    def cut4(inp):
        """
        按英文句号.切
        :param inp:
        :return:
        """
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip(".").split(".")]
        opts = [item.strip() for item in opts if not set(item).issubset(Cutter.PUNCTUATION)]
        opts = [_ for _ in opts if _]
        return opts

    @staticmethod
    def cut5(inp):
        """
        按标点符号切
        :param inp:
        :return:
        """
        inp = inp.strip("\n")
        punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
        mergeitems = []
        items = []

        for i, char in enumerate(inp):
            if char in punds:
                if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)

        if items:
            mergeitems.append("".join(items))

        opt = [item.strip() for item in mergeitems if not set(item).issubset(punds)]
        opt = [_ for _ in opt if _]
        return opt

    @staticmethod
    def cut6(inp):
        """
        按标点符号切
        :param inp:
        :return:
        """
        inp = inp.strip("\n")
        punds = {"。", "？", "！", ".", "?", "!", "~", "…"}
        quotes = {'"', "'", "’", "‘", "”", "“"}
        mergeitems = []
        items = []

        for i, char in enumerate(inp):
            if char in punds:
                if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)

        if items:
            mergeitems.append("".join(items))

        opt = [item.strip("".join(quotes)).strip() for item in mergeitems if
               not set(item).issubset(punds.union(quotes))]
        opt = [_.strip() for _ in opt if _.strip()]
        return opt


if __name__ == '__main__':
    text = "你好，我是小明。你好，我是小红。你好，我是小刚。你好，我是小张。"
    print(Cutter("cut0")(text))
    print(Cutter("cut1")(text))
    print(Cutter("cut2")(text))
    print(Cutter("cut3")(text))
    print(Cutter("cut4")(text))
    print(Cutter("cut5")(text))
