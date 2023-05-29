import numpy as np
import torch as torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration

if __name__ == '__main__':
    # pretrain_path = 'IDEA-CCNL/Randeng-T5-784M-QA-Chinese'
    # tokenizer=T5Tokenizer.from_pretrained(pretrain_path)
    # model=MT5ForConditionalGeneration.from_pretrained(pretrain_path)

    model_path = 'E:\\sinri\\ezra\\models\\Randeng-T5-784M-QA-Chinese'
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)

    max_knowledge_length = 1024

    sample = {
        "context": "在柏林,胡格诺派教徒创建了两个新的社区:多罗西恩斯塔特和弗里德里希斯塔特。到1700年,这个城市五分之一的人口讲法语。柏林胡格诺派在他们的教堂服务中保留了将近一个世纪的法语。他们最终决定改用德语,以抗议1806-1807年拿破仑占领普鲁士。他们的许多后代都有显赫的地位。成立了几个教会,如弗雷德里夏(丹麦)、柏林、斯德哥尔摩、汉堡、法兰克福、赫尔辛基和埃姆登的教会。",
        "question": "除了多罗西恩斯塔特,柏林还有哪个新的社区?",
        "idx": 1
    }
    plain_text = 'question:' + sample['question'] + 'knowledge:' + sample['context'][:max_knowledge_length]

    print(plain_text)

    res_prefix = tokenizer.encode('answer', add_special_tokens=False)
    res_prefix.append(tokenizer.convert_tokens_to_ids('<extra_id_0>'))
    res_prefix.append(tokenizer.eos_token_id)
    l_rp = len(res_prefix)

    tokenized = tokenizer.encode(plain_text, add_special_tokens=False, truncation=True, max_length=1024 - 2 - l_rp)
    tokenized += res_prefix
    batch = [tokenized] * 2
    input_ids = torch.tensor(np.array(batch), dtype=torch.long)

    # Generate answer
    max_target_length = 128
    pred_ids = model.generate(input_ids=input_ids, max_new_tokens=max_target_length, do_sample=True, top_p=0.9)
    pred_tokens = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    res = pred_tokens.replace('<extra_id_0>', '').replace('有答案:', '')
    print(res)
