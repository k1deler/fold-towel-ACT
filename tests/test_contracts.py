import json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'training'),str(ROOT/'training/upstream'),str(ROOT/'inference')]
from audit import raw_audit,converted_audit

def test_delta_actions_include_grippers_and_use_one_anchor():
    from yam_train.data.dataset import chunk_targets
    q=np.arange(140,dtype=np.float32).reshape(10,14)/100
    a=q+.1
    values,pad=chunk_targets(q,a,3,6,0,'delta')
    np.testing.assert_allclose(values,a[3:9]-q[3])
    assert not pad.any()
    np.testing.assert_allclose(values[1,[6,13]],a[4,[6,13]]-q[3,[6,13]])

def test_split_is_episode_disjoint_deterministic():
    from yam_train.data.dataset import split_episodes
    t,v=split_episodes(51,.8,1)
    assert len(t)==40 and len(v)==11 and not set(t)&set(v)
    np.testing.assert_array_equal(t,split_episodes(51,.8,1)[0])

def test_normalization_does_not_see_validation():
    from types import SimpleNamespace
    from yam_train.data.dataset import norm_stats
    q=np.zeros((8,14));a=np.zeros((8,14));q[4:]=50;a[4:]=100
    st=norm_stats(SimpleNamespace(qpos=q,action=a),[dict(start=0,length=4)],2,0,'delta')
    np.testing.assert_array_equal(st['qpos_mean'],np.zeros(14))
    np.testing.assert_array_equal(st['action_mean'],np.zeros(14))

def test_raw_recorder_error_is_rejected(tmp_path):
    for i in range(5):
        e=tmp_path/f'episode_{i:04d}';e.mkdir();(e/'metadata.json').write_text('{}')
        (e/'episode_stats.json').write_text(json.dumps(dict(writer_error='failed')))
    with pytest.raises(ValueError):raw_audit(tmp_path)

def test_training_scripts_do_not_shadow_standard_library():
    import sys
    files={p.stem for p in (ROOT/'training').glob('*.py')}
    assert not files & sys.stdlib_module_names

def test_training_and_inference_use_identical_model_structure():
    import torch
    from yam_train.policy import ACTPolicy
    from yam_train.detr.models import backbone
    from yam_infer.policy.act_ref import ACTRefPolicy
    torch.set_num_threads(4);torch.manual_seed(42)
    cfg=dict(chunk_size=8,kl_weight=10,hidden_dim=64,dim_feedforward=128,enc_layers=1,
        dec_layers=1,nheads=8,dropout=.1,pre_norm=False,backbone='resnet18',position_embedding='sine',
        dilation=False,masks=False,lr=1e-5,lr_backbone=1e-5,weight_decay=1e-4)
    original=backbone.is_main_process;backbone.is_main_process=lambda:False
    try:model=ACTPolicy(cfg,['right_wrist','left_wrist'])
    finally:backbone.is_main_process=original
    q=torch.zeros(1,14);img=torch.rand(1,2,3,64,64);a=torch.zeros(1,8,14);pad=torch.zeros(1,8,dtype=torch.bool)
    model.train();loss=model(q,img,a,pad)['loss'];assert torch.isfinite(loss)
    optimizer=model.configure_optimizers();optimizer.zero_grad();loss.backward();optimizer.step()
    model.eval();deploy=ACTRefPolicy(cfg,['right_wrist','left_wrist']);deploy.load_state_dict(model.state_dict());deploy.eval()
    with torch.inference_mode():torch.testing.assert_close(model.predict(q,img),deploy(q,img),rtol=0,atol=0)
