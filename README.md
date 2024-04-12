书生浦语-第二期实战营12班 

【第一课】《书生浦语大模型全链路开源体系》 
===============
By 人工智能实验室陈凯老师 https://www.bilibili.com/video/BV1Vx421X72D/

# 一、大模型成为发展通用人工智能的重要途径（专用模型→通用大模型）

* 【专用模型】特定任务，一个模型解决一个问题（2006-2021）语音/图像/人脸识别/德扑...

* 【通用大模型】一个模型解决多个问题，多种模态（chatgpt/gpt4）

# 二、书生浦语大模型开源历程（2023.6起)

* 2023.6.7  InternLM 千亿参数LM发布

* 2023.7.6 InternLM 升级（8k语境，26种语言） 全面开源、免费商用  InternLM-7B 全链条开源工具体系

* 2023.8.14 书生万卷1.0--多模态训练语料库开源；

* 2023.8.21 升级版对话模型InternLM-chat-7B V1.1，开源智能体框架Lagent（支持语言模型到智能体升级转换）

* 2023.8.28 InternLM千亿参数量升级到123B

* 2023.9 增强版InternLM-20B开源，开源工具链全栈升级

* 2024.1  InternLM 2开源

# 三、书生浦语2.0（InternLM2）的体系

【7B】轻量级 + 【20B】综合级（解决复杂问题）

每个规格鱼油3个版本：IntermLM2-Base、InternLM2（推荐使用）、InternLM2-Chat（共情聊天）

# 四、InternLM2做了什么事？  回归语言建模的本质

第一代数据清洗过滤技术（多维度数据价值评估→高质量语料推动的数据富集→有针对性地数据补齐）

随着训练数据语料升级，下游任务性能增强

# 五、InternLM2的主要两点

1. 超长上下文（20万token上下文）
2. 综合性能全面提升（推理、数学、代码 InternLM-20B 重点测评比肩ChatGPT）
3. 对话和创作体验好（AlpacaEval2超越Gpt3.5和Gemini Pro）：流量地球3的创作
4. 工具调用能力整体升级（支持复杂智能体搭建）：路线规划、餐厅查询、邮件发送任务
5. 突出的数理能力和实用的数据分析功能（GSM8K和Math达到和GPT4相仿水平）：数学计算，1000以内达到80%的准确率，大学微积分。。。；分析数据，画出曲线图，数据预测
6. ****【代码解释器】：可进一步提升数学成绩

# 六、从模型到应用

书生浦语→ （多环节gap）→ 智能客服 / 个人助手 / 行业应用

1. 模型评测选型（参考经典评测集/榜单）
2. 业务场景是否复杂？ 复杂+算力足 → 续训/全参数微调； 复杂+算力不足 → 部分参数微调
3. 是否需环境交互？ 是 → 构建智能体 → 模型评测 → 模型部署； 否→模型评测→模型部署

# 七、书生浦语全链条开源开放体系：数据集获取https://opendatalab.org.cn

1. 数据（书生万卷1.0<图像/文本/视频>>→书生万卷CC<2013-2023>2024/3/6发布） 2TB数据，多模态
2. 预训练（InternLM-Train） 速度达3600 token/sec/gpu
3. 微调（Xtuner）支持全参数微调、LoRA等低成本部署：(1）增量续训<让模型学新知识：文章、书籍、代码。。。> （2）有监督的微调<让模型学会理解各种指令进行对话：高质量的对话、问答数据>
   适配多种生态、多种硬件（训练方案覆盖NVIDIA20系以上所有显卡、最低只需8GB显存即可微调7B模型）

4. 评测（OpenCompass）性能可复现100套评测集、50万道题目：24/1/30 OpenCompass2.0司南大模型评测体系发布
   1）CompassRank评测性能榜单  2）CompassKit全栈工具链（数据污染检查/跟更富的模型推理接入/长文本能力评测/中英文双语主观评测） 3）CompassHub评测基准社区
   Meta官方推荐唯一国产大模型评测体系，已适配了100+评测集 50万+题目
   主观评测对战胜率：GPT4、质谱AI、阿里
5. 部署（LMDeploy）智能生成2000+token/秒：
6. 应用（Lagent AgentLego智能体工具箱）支持多种智能体，支持【代码解释器】等多种工具，多模态AI工具使用（零样本泛化Zero-shot generalization：新型疾病预测）

   支持多种类型的智能体能力：ReAct、ReWoo、AutoGPT

L2：轻松玩转书生浦语趣味demo（任宇鹏）

Demo实战的任务内容

1. 实战部署InternLM2-Chat-1.8B
创建Intern Studio开发机（https://github.com/InternLM/Tutorial），通过Modelscope下载InternLM2-Chat-1.8B模型，完成Client Demo的部署和交互
* 1）创建开发机
* 2）下载InternLM2-Chat-1.8B模型
* 3）部署本地Client Demo

2. 实战部署“八戒-Chat-1.8B”
通过部署OpenXLab部署XiYou系列的八戒-Chat-1.8B模型，完成Web Demo的部署和交互

3. 实战进阶 运行Lagent智能体Demo
实战算力升级后，以InternLM2-Chat-7B为基础，运行开源框架Lagent的智能体Demo
* 1)了解Lagent智能体
* 2）部署InternLM2-Chat-7B
* 3）体验与智能体Demo的聊天互动

4. 实战进阶 灵笔InternLM-XComposer2
浅尝多模态实践，通过internLM-XComposer2模型实现更强大的视觉问答和图文生成式写作

[Lizzy的markdown库][[1]
! [1]:http://-/300字小故事作业 lizzy 4.6.png at main · LizzyDuan/- (github.com)

【第三课】R茴豆：搭建RAG智能助理（by 北辰）
=============
1. RAG:是什么、原理、RAG VS Fine-tune、架构、向量数据库、评估和测试

   * 茴香豆InternLM2-Chat-7B RAG助手
   * RAG技术概述 Retrieval Augmented Generation: 检索+生成，通过利用外部知识库LlLMs的性能。
     - 回答更准确、成本低（无训练过程）、实现外部记忆
     - 适合做“问答系统、文本生成、信息检索、照片描述”
    
   * RAG 工作原理
   - 1. Indexing 索引（将知识源（如文档或网页）分割成chunk，编码或向量，存储在向量数据库中。
   - 2. Retrieval 检索 (接收到用户的问题后，将问题也编码成向量，并在向量数据库中找到与之最相关的文档快 top-k chunks)
   - 3. Generation 生成 （将检索到的文档块与原始问题一起作为prompt输入LLM中，生成最终的回答。）

    > **VECTOR DATABASE: 向量数据库（存储外部数据）**
    --------------
      >> 1. 数据存储（将文本即其他数据通过其他预训练模型转换为固定长度的向量表示，这些向量能捕捉文本的语义信息。）
      >> 2. 相似性检查 （根据用户的查询向量，使用向量数据库快速找出最相关的向量的过程。通常通过计算余弦相似度或其他相似度量来完成。检索结果根据相似度得分排序，最相关的文档将被用于后续文本生成。）
      >> 3. 向量表示的优化（包括使用更高级的文本编码技术，如句子嵌入或段落嵌入，以及对数据库进行优化以支持大规模向量搜索。）
    
    * RAG 工作流程
    * ![549045950014864942](https://github.com/LizzyDuan/LZ-notebook/assets/165522321/fc55d96c-eca7-48c5-8216-d8183d683cae)

    * RAG 发展历程
      > Naive RAG（简单问答，信息检索）
      > Advance RAG （摘要生成，内容推荐） GOOD!!!!!
      > Module RAG （多模态任务，对话系统）

    * RAG 常见优化方法
      > 嵌入优化 Embedding Optimization ：结合稀疏和密集检索，多任务
      > 索引优化 Indexing Optimization ： 细粒度分割chunk，元数据
      > 查询优化 Query Optimization ：Advance RAG 前检索 ---查询扩展、转换，多查询
      > 上下文管理 Context Curation ： Advance RAG 后检索 ---重排rerank，上下文选择/压缩
      > 迭代检索 Interative Retrieval ： 根据初始查询和迄今为止生成的文本进行重复搜索
      > 递归检索 Recusive Retrieval ：迭代细化搜索查询，链式推理Chain-of-Thought指导检索过程
      > 自适应检索 Adaptive Retrieval ：Flare，Self-RAG， 使用LLMs主动决定检索的最佳实际和内容
      > LLM微调 LLM fine-tuning ：检索/生成/双重微调
      
     * RAG vs 微调Fine-tuning
       > RAG:小数据样本的开放问答，动态数据库更新，实时新闻；高度依赖基础大模型性能，数据库品质很重要。
       > Fine-tuning：参数记忆，通过在特定任务数据上训练，需要大量标注数据来进行有效微调；需要特别多的数据量，每次信息更新都要新微调。
       
    * LLM模型优化方法
       >
    * 评估框架和基准测试
       > [经典评估指标]
       >> 准确率、召回率、F1分数、BLEU分数（用于机器翻译和文本生成）、ROUGE分数（用于文本生成的评估）
       > [RAG评测框架]
       >> 基准测试、RGB、RECALL、CRUD
       >> 评测工具：RAGAS、ARES、TruLens
       
    * RAG 总结
    * ![373732025928785431](https://github.com/LizzyDuan/LZ-notebook/assets/165522321/5bcc35cb-2f72-4c96-8732-64658a0fb5a8)
       > **RAG Ecosystem**
       >> Downstream Tasks : Dialogue, Question answering, Summarizatioin, Fact verification
       >> Technology Stakes : Langchain, LlamaIndex, FlowiseAI, AutoGen
       * > **The RAG Paradigm**
         >> Naive RAG / Advanced RAG / Modular RAG
         >>> The techniques for Better RAG
         >>>> Chunk optimization, iterative retrieval, retriever fine-tuning, query transformation, recusive retrieval, generator fine-turning, context selection, adpative retrieval, dual fine-turning
      * > **RAG Porspect**
       >> Challenges: RAG in Long Context length, Hybrid, Robustness, Scaling-laws for RAG, Production-ready RAG
       >> Modality Extension :  Image, Video, Audio, Code
       >> Ecosystem : Customization, Simplification, Specialization
      * > **Evaluation of RAG**
       >> Evaluation Target: Retrieval Quality, Generation Quality
       >> Evaluation Aspect: Answer Retrieval, Noise Robustness, Context Relevance, Negation Rejection, Info Integration, Answer Faithfulness, Couterfactual Robustness
       >> Evaluation Framework:
       >>> Benchmarks: CRUD, RGB, RECALL
       >>> Tools: TruLens, RAGAS, ARES
  
3. 茴香豆：介绍、特点、架构、构建步骤
 ======================
* **应用场景**：智能客服：技术支持、领域知识对话
  > IM即时通讯工具中创建用户群组、讨论、解答相关问题
* **场景难点**：群聊中信息量巨大，内容多样，技术讨论和闲聊都有
* **核心特性**：
  > 开源免费：BSD-3-Clasue免费商用
  > 高效准确：Hybrid LLMs专为群聊优化
  > 领域知识：应用RAG技术专业知识快速获取
  > 部署成本低：无需额外训练，可利用云端模型api，本地算力需求少
  > 安全：可完全本地部署，信息不上传，保护数据和用户隐私
  > 扩展性强：兼容多种IM软件，支持多种开源LLMs和云端api
* **构建**
  > 专属知识库资料：pdf word等
  > 前端：微信群、飞书群
  > LLM后端：本地大模型（书生浦语/通义千问的模型格式；远端kimi）
  > 豆哥：打通工作流
  >> chat group（sentense） → preprocess(query) → rejection pipeline(query) → response pipeline（reply） → chat group
  >> 完整工作流
  >>> * 多来源检索 ：向量数据库、网络搜索结果、知识图谱
  >>> * 混合大模型 ：本地LLM，远程LLM
  >>> * 多重评分拒答工作流 ：回答有效、避免信息泛滥
  >>> * 安全检查 ：多种手段、确保回答合规
  
4. 实践演示：茴香豆Web版、Intern Studio部署茴香豆知识助手
   ==========================
   作业截图
   <img width="956" alt="image" src="https://github.com/LizzyDuan/LZ-notebook/assets/165522321/8d063c1d-feb8-4ecf-a758-aa71c23371ad">









