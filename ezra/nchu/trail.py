# -*- coding: utf-8 -*-

import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, pipeline, QuestionAnsweringPipeline

# model_name = "NchuNLP/Chinese-Question-Answering"
# tokenizer = BertTokenizerFast.from_pretrained(model_name)
# model = BertForQuestionAnswering.from_pretrained(model_name)

model_path = "E:\\sinri\\ezra\\models\\Chinese-Question-Answering"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)


def get_predictions(content, question):
    # a) Get predictions
    # nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    nlp=QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

    QA_input = {
        'question': question,
        'context': content
    }
    res = nlp(QA_input)

    # {'score': 1.0, 'start': 21, 'end': 23, 'answer': '臺中'}
    return res


def inside_pipeline(text, query):
    # b) Inside the Question answering pipeline

    inputs = tokenizer(query, text, return_tensors="pt", padding=True, truncation=True, max_length=512, stride=256)
    outputs = model(**inputs)

    sequence_ids = inputs.sequence_ids()
    # Mask everything apart from the tokens of the context
    mask = [i != 1 for i in sequence_ids]
    # Unmask the [CLS] token
    mask[0] = False
    mask = torch.tensor(mask)[None]

    start_logits = torch.tensor()
    end_logits = torch.tensor()

    start_logits[mask] = -10000
    end_logits[mask] = -10000

    start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
    end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

    scores = start_probabilities[:, None] * end_probabilities[None, :]

    max_index = scores.argmax().item()
    start_index = max_index // scores.shape[1]
    end_index = max_index % scores.shape[1]

    inputs_with_offsets = tokenizer(query, text, return_offsets_mapping=True)
    offsets = inputs_with_offsets["offset_mapping"]

    start_char, _ = offsets[start_index]
    _, end_char = offsets[end_index]
    answer = text[start_char:end_char]

    result = {
        "answer": answer,
        "start": start_char,
        "end": end_char,
        "score": scores[start_index, end_index],
    }
    print(result)


if __name__ == '__main__':
    question = '中興大學在哪裡？'
    context = '國立中興大學（簡稱興大、NCHU），是位於臺中的一所高等教育機構。中興大學以農業科學、農業經濟學、獸醫、生命科學、轉譯醫學、生醫工程、生物科技、綠色科技等研究領域見長 。近年中興大學與臺中榮民總醫院、彰化師範大學、中國醫藥大學等機構合作，聚焦於癌症醫學、免疫醫學及醫學工程三項領域，將實驗室成果逐步應用到臨床上，未來「衛生福利部南投醫院中興院區」將改為「國立中興大學醫學院附設醫院」。興大也與臺中市政府合作，簽訂合作意向書，共同推動數位文化、智慧城市等面相帶動區域發展。'

    res = get_predictions(context, question)
    print(res)

    # res = inside_pipeline(context, question)
    # print(res)
