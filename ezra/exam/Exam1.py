# -*- coding: utf-8 -*-

from ezra.luhua.LuhuaExtQA import LuhuaExtQA
from ezra.nchu.NchuQA import NchuQA
from ezra.randeng.Randeng import Randeng

model_mapping = {
    'luhua': 'E:\\sinri\\ezra\\models\\chinese_pretrain_mrc_roberta_wwm_ext_large',
    'nchu': 'E:\\sinri\\ezra\\models\\Chinese-Question-Answering',
    'randeng': 'E:\\sinri\\ezra\\models\\Randeng-T5-784M-QA-Chinese',
}


def exam_qa(ezra):
    context = '''
    异构计算（Heterogeneous Computing）是指使用不同类型指令集和体系架构的计算单元组成系统的计算方式，目前主要包括GPU云服务器、FPGA云服务器和弹性加速计算实例EAIS等。异构计算能够让最适合的专用硬件去服务最适合的业务场景，在特定场景下，异构计算产品比普通的云服务器高出一个甚至更多数量级的性价比和效率。异构计算的显著优势在于实现了让性能、成本和功耗三者均衡的技术，通过让最合适的专用硬件去做最适合的事来调节功耗，从而达到性能和成本的最优化。
    随着以深度学习为代表的人工智能技术的飞速发展，AI计算模型越来越复杂和精确，人们对于算力和性能的需求也大幅度增加，因此，越来越多的AI计算都采用异构计算来实现性能加速。阿里云异构计算云服务研发了云端AI加速器，通过统一的框架同时支持了TensorFlow、PyTorch、MXNet和Caffe四种主流AI计算框架的性能加速，并且针对以太网和异构加速器本身进行了深入的性能优化。
    '''

    question = '什么是异构计算？'

    for answer in ezra.read_and_answer(context, question):
        print(f"ANSWER: {answer}")


if __name__ == '__main__':

    print(LuhuaExtQA.model_name)
    qa = LuhuaExtQA(model_mapping['luhua'])
    exam_qa(qa)

    print(NchuQA.model_name)
    qa = NchuQA(model_mapping['nchu'])
    exam_qa(qa)

    print(Randeng.model_name)
    qa = Randeng(model_mapping['randeng'])
    exam_qa(qa)

