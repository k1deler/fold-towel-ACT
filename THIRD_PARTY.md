# 来源与新增部分

- `training/upstream`来自[FuFutheBest/yam_train](https://github.com/FuFutheBest/yam_train)，固定commit见`training/source.json`，保留ACT/DETR许可证和来源文件头。
- `inference`的`yam_infer`、驱动与工具来自[FuFutheBest/yam_infer](https://github.com/FuFutheBest/yam_infer)，固定commit`bb2e8aba33d6bd9bcba43c2ffca129ad90c3feb5`；原第三方清单保留在`inference/VENDORED.md`。
- I2RT SDK与ACT/DETR已有许可证保留。上游应用代码没有在本项目中重新授权，不声称第三方实现全部原创。
- 本次新增：fold_towel任务配置、完整性审计、独立训练管线、有限训练进程管理、原始数据SHA验证、有限值检查、训练集start_pose／归一化、选模包装、训练与部署输出一致性检查及使用说明。
- `fit.py`使用上游网络、数据、loss、AdamW参数分组；不修改上游网络源码。相对于原训练循环新增日志、梯度检查和原子checkpoint。启动不加载黄块权重，只有标准ImageNet视觉骨干初始化。
