书生浦语-第二期实战营12班 

【第一课】《书生浦语大模型全链路开源体系》 By 人工智能实验室陈凯老师 https://www.bilibili.com/video/BV1Vx421X72D/
==================

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

   







