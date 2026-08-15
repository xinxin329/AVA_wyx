# AVA_wyx

本仓库是 AVA（Agentic Video Analytics）在 AVA-100 数据集上的单卡复现与诊断版本，包含 Wildlife1 和 Traffic1 两段长视频的核心流程代码、问答评测脚本、Debug 工具以及实验图表。

本仓库不包含模型权重、原始视频、抽帧缓存、虚拟环境和完整运行缓存。运行前需要自行准备数据与本地模型，并按照说明配置路径。

## 已完成内容

- 长视频抽帧与基础片段组织
- Qwen2.5-VL 片段描述与事件摘要
- DeBERTa/BERTScore 语义事件合并
- JinaCLIP 图文向量编码
- 事件知识图谱构建
- SA 智能体式检索问答
- CA 原始视频帧复核
- Wildlife1 摘要边界与语义分块诊断
- Traffic1 Oracle Time 诊断
- 准确率、投票分布和局部知识图谱可视化

## 使用说明

完整的环境配置、模型分工、运行命令、复现流程和结果路径见：

[说明文档.md](./说明文档.md)

## 图表

- Wildlife1：`outputs/figures/`
- Traffic1：`outputs/traffic1_figures/`

## 说明

当前结果属于单张 RTX 4090 条件下的局部复现。由于 SA/CA 模型和视觉帧数配置与论文完整实验不同，仓库中的准确率用于验证流程、分析错误和研究方法适用边界，不应直接视为对论文完整 AVA-100 总结果的严格复刻。
