from typing import Optional

from ezra.Ezra import Ezra
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline


class LuhuaExtQA(Ezra):
    """
    :see https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large
    :model-name luhua/chinese_pretrain_mrc_roberta_wwm_ext_large
    :model-clone git clone https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large
    """
    model_name = 'luhua/chinese_pretrain_mrc_roberta_wwm_ext_large'

    def __init__(self, model: Optional[str] = None):
        super().__init__()

        if model is None:
            model = self.model_name

        self.__tokenizer = AutoTokenizer.from_pretrained(model)
        self.__model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.__nlp = QuestionAnsweringPipeline(model=self.__model, tokenizer=self.__tokenizer)

    def read_and_answer(self, content: str, question: str):

        res = self.__nlp({
            'question': question,
            'context': content
        })
        if res is None:
            return None
        else:
            return [res['answer']]


if __name__ == '__main__':
    model_name = 'luhua/chinese_pretrain_mrc_roberta_wwm_ext_large'
    model_path = 'E:\\sinri\\ezra\\models\\chinese_pretrain_mrc_roberta_wwm_ext_large'
    qa = LuhuaExtQA(model_path)
    x = qa.read_and_answer(
        '''
         约书亚吩咐窥探地的两个人说：“你们进那妓女的家，照着你们向她所起的誓，将那女人和她所有的都从那里带出来。”
         当探子的两个少年人就进去，将喇合与她的父母、弟兄和她所有的，并她一切的亲眷，都带出来，安置在以色列的营外。
         众人就用火将城和其中所有的焚烧了。惟有金子、银子和铜铁的器皿，都放在耶和华殿的库中。
         约书亚却把妓女喇合与她父家，并她所有的，都救活了，因为她隐藏了约书亚所打发窥探耶利哥的使者，她就住在以色列中，直到今日。
         ''',
        '''
        约书亚和喇合有什么关系？
        '''
    )
    print(x)
