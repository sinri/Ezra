import numpy as np
import torch as torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration

from ezra.Ezra import Ezra


class Randeng(Ezra):
    def __init__(self, model: str, max_knowledge_length: int = 10240):
        super().__init__()

        self.__tokenizer = T5Tokenizer.from_pretrained(model)
        self.__model = MT5ForConditionalGeneration.from_pretrained(model)

        self.__max_knowledge_length = max_knowledge_length
        self.__max_token_length = 10240
        self.__max_answer_length = 1024

    def read_and_answer(self, content: str, question: str):
        sample = {
            "context": content,
            "question": question,
            "idx": 1
        }

        # plain_text = 'question:' + sample['question'] + 'knowledge:' + sample['context'][:self.__max_knowledge_length]

        plain_text = f"""
        f{sample['context'][:self.__max_knowledge_length]}
        question:
        f{sample['question']}
        """

        # plain_text = '''
        # 老虎是什么？
        # 根据进化论，老虎是人类变来的。作为一种高级生物，人类因为互相吃而发生了进化，变成了老虎。
        # 老虎吃什么？
        # 老虎和人类一样吃人，因为老虎是人类变来的。
        # 老虎和人类有什么区别？
        # 老虎有一条和猫一样细的尾巴，长度和人类的手臂相当。人类没有尾巴。
        # 问：
        # 人类的尾巴有多长？
        # '''

        res_prefix = self.__tokenizer.encode('answer', add_special_tokens=False)
        res_prefix.append(self.__tokenizer.convert_tokens_to_ids('<extra_id_0>'))
        res_prefix.append(self.__tokenizer.eos_token_id)
        l_rp = len(res_prefix)

        tokenized = self.__tokenizer.encode(
            plain_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.__max_token_length - 2 - l_rp
        )
        tokenized += res_prefix
        batch = [tokenized] * 2
        input_ids = torch.tensor(np.array(batch), dtype=torch.long)

        # Generate answer
        max_target_length = self.__max_answer_length
        pred_ids = self.__model.generate(input_ids=input_ids, max_new_tokens=max_target_length, do_sample=True,
                                         top_p=0.9)
        pred_tokens_list = self.__tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        # print(pred_tokens_list)
        answers = []
        for pred_tokens in pred_tokens_list:
            res = pred_tokens.replace('<extra_id_0>', '').replace('有答案', '')
            if res.startswith(':'):
                res = res[1:]
            answers.append(res)

        # pred_tokens = pred_tokens_list[0]
        # res = pred_tokens.replace('<extra_id_0>', '').replace('有答案', '')
        # if res.startswith(':'):
        #     res=res[1:]

        return answers


if __name__ == '__main__':
    model_path = 'E:\\sinri\\ezra\\models\\Randeng-T5-784M-QA-Chinese'
    randeng = Randeng(model_path, 1024)
    x = randeng.read_and_answer(
        '''
         约书亚吩咐窥探地的两个人说：“你们进那妓女的家，照着你们向她所起的誓，将那女人和她所有的都从那里带出来。”
         当探子的两个少年人就进去，将喇合与她的父母、弟兄和她所有的，并她一切的亲眷，都带出来，安置在以色列的营外。
         众人就用火将城和其中所有的焚烧了。惟有金子、银子和铜铁的器皿，都放在耶和华殿的库中。
         约书亚却把妓女喇合与她父家，并她所有的，都救活了，因为她隐藏了约书亚所打发窥探耶利哥的使者，她就住在以色列中，直到今日。
         ''',
        '''
        文中出现了几个人物？
        '''
    )
    print(x)
