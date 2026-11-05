# fold-towel-ACT — ACT 折毛巾

本项目以用户提供的51段`fold_towel`真实遥操作录制训练折毛巾模型，复用已验证黄块项目的ACT参考网络与YAM推理结构。不是复用黄块任务权重，也不将随机测试模型作为成品。

**2026-11-02更新：训练与离线选模已完成。** 5000epoch、25000更新，选出epoch3499，验证窗口平均两臂关节MAE约0.07333rad。336MB真实模型已导出并核验SHA；实机折毛巾尚未验证。部署完整步骤见[部署步骤.md](部署步骤.md)，独立部署ZIP自带权重，使用`bash install_bundle.sh`进行不连接硬件的安装与真实观测自检。

## 仓库地址与安装目录

```bash
git clone git@github.com:k1deler/fold-towel-ACT.git
cd fold-towel-ACT
```

已有本地克隆可在其目录执行：

```bash
git remote set-url origin git@github.com:k1deler/fold-towel-ACT.git
```

改名不改变模型中的`fold_towel`任务标识、权重SHA或训练/推理合同。此前交付ZIP与现场安装目录沿用原文件名；按[部署步骤.md](部署步骤.md)使用，不要仅为仓库改名移动正在使用的机器人目录。已有训练记录中的历史机器路径按原样保留。

## 结构

| 目录 | 用途 |
|---|---|
| `training/upstream` | 已注明来源的ACT训练网络、原数据转换器、训练评估工具 |
| `training/fit.py` | 训练入口：有限值检查、独立输出、完整episode划分、训练集归一化、原ACT优化方式 |
| `training/pipeline.py` | 源数据SHA校验、转换、等空闲GPU、小训练验证、正式训练与选模 |
| `training/check_contract.py` | 同一权重／真实观测下比较训练网络与部署网络逐元素输出 |
| `training/select_candidate.py` | 固定验证窗口的物理误差选模与导出；折毛巾左右臂均计入 |
| `inference` | 已注明来源的YAM worker、SDK、相机、动作调度、录制和归位流程 |
| `configs` | 任务、动作模式、相机顺序、训练和运行配置 |
| `scripts` | 注册真实训练权重、GPU无硬件验证、现场启动 |
| `tests` | 训练／部署合同、delta动作、episode划分和数据错误检查 |
| `reports` | 版本、数据摘要和验证结果；不存储密码／录制画面 |

## 训练合同

- 任务标识`fold_towel`。原录制metadata的任务名是通用“测试”，本项目根据用户指定目录使用fold_towel；不修改原始录制。
- 左6关节＋左夹爪＋右6关节＋右夹爪；从臂为状态，主臂为监督目标。
- 两路腕部RGB JPEG，按`right_wrist, left_wrist`顺序；480×640、30Hz。head为黑图占位，不参与模型。
- ACT参考网络：ResNet18＋Transformer，hidden512，encoder4、decoder7，chunk100。
- 14维delta目标均相对同一次观测从臂状态，包括两个夹爪；action_shift0。训练归一化mean/std与推理反归一化必须保持一致。
- 新seed42任务参数，视觉骨干使用标准ImageNet初始化，不读取黄块任务checkpoint。
- 51段按完整episode80/20划分，正常为40训练／11验证；只用训练episode计算归一化与start_pose。
- 首组5000epoch、batch8、AdamW lr1e-5、KL10。数据或小训练验证失败就停，不触发正式训练。
- 验证选择使用z=0、512固定窗口的两臂平均物理关节MAE，不把随机后验验证loss最低值当作唯一交付依据。

## 训练机运行

本次独立服务器目录：`~/output/personal/k1deler/fold_towel_20261101`。原始录制从本机`C:\Users\lenovo\Downloads\fold_towel`上传至该目录，按文件SHA复核。使用现有CUDA环境，只将缺少的读取依赖装在该实验的`dependencies/`；不升级原环境、不覆盖原实验、不停止其他人的作业。

默认GPU忙时等待空闲卡，最多等待6小时。用户在2026-11-02明确允许共用GPU2，因此本次运行使用`--allow-shared-gpu 2`：仍优先空闲卡，共用卡须至少剩余32GiB显存；不停止其他作业。训练完成后其他卡空闲时可用于选择评估。不是未经验证的动态多卡同步训练。单组正式训练最多12小时、选择评估最多2小时，失败或超时均记录，不能当作完成。查看`reports/pipeline_status.json`和各阶段日志。

数据转换与全部训练阶段只访问文件和GPU，不连接机器人、实时相机或CAN。

## 成品和现场操作

训练完成后输出`artifacts/fold_towel_act_delta_seed42.pt`、`model_card.json`和候选选择记录。这是离线候选，必须在实验室另做真实模型GPU自检及现场折叠试验，不能保证折叠成功。

将整个代码项目及导出权重分别复制到实验室新目录后，运行：

```bash
bash scripts/setup_robot.sh /absolute/path/fold_towel_act_delta_seed42.pt
# 离线自检通过、现场准备好、原程序停止后：
bash scripts/run_robot.sh
```

启动入口支持交互SSH终端，保留START确认、互斥和硬件占用检查。**START后就会初始化驱动并移动到模型start_pose**，随后Enter开始／暂停，`r`＋回车结束一轮并回起始姿态，`q`＋回车归位退出。`q`和Ctrl+C会触发归位运动，不能作为即时急停；现场操作员负责观察和物理急停。默认两臂都不冻结，单轮上限60秒可在验证配置后调整。

相机左右身份、CAN、RPC沿用同一台实验室YAM配置；换机器人时须重新核对。折毛巾场景与起始姿态必须按新示范布置，不能照搬黄块的摆放。

## 数据和仓库

目标仓库`k1deler/fold-towel-ACT`。代码、配置、来源、数据SHA摘要、训练曲线和报告可以入库；2.61GB原始录制、模型大权重、虚拟环境、密码和私钥不直接提交Git。权重另行下载并用SHA识别，原始数据保留在授权训练目录。

训练记录是本次实验产物；ACT架构、参考训练代码、YAM推理程序和SDK保留来源。说明见`THIRD_PARTY.md`。
