"""Verify that reset-hc changes only neural state, including checkpoint rebuild."""
import argparse
from pathlib import Path
import torch
from config import CaseConfig
from nets.EPINN_Net import EPINN_PhyLSTM_NetBody
from nets.EPINN_Loss import EPINN_MDOFSys_DisIncrement_PhyLoss
from utils.DataPreProcess import load_case_data, as_torch_case
from EPINN_MDOFSys_Test import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260730)
    cfg = CaseConfig(data_root=args.data_root, time_truncation=40)
    data = load_case_data(cfg)
    t = as_torch_case(data, torch.device('cpu'))
    kw = dict(nLoad=5,nLoadNL=5,influence_kernel=t['kernel'],stiffness=t['stiffness'],
              fiber=t['fiber'],steel=t['steel'],hidden_size=8,fc_size=8)
    carry = EPINN_PhyLSTM_NetBody(**kw).double()
    reset = EPINN_PhyLSTM_NetBody(**kw,reset_lstm_state=True).double()
    reset.load_state_dict(carry.state_dict())
    load = torch.as_tensor(data.load[:1,:64].transpose(0,2,1)[:,None],dtype=torch.float64)
    a, state = carry.forward_chunk(load[...,:32])
    b, state_b = reset.forward_chunk(load[...,:32])
    torch.testing.assert_close(a['dis_nl'], b['dis_nl'], rtol=0,atol=0)
    # Feeding the SAME complete physical state isolates the effect of h,c.
    expected, _ = carry.forward_chunk(load[...,32:], {**state, 'temporal':None})
    actual, next_state = reset.forward_chunk(load[...,32:], state)
    continued, _ = carry.forward_chunk(load[...,32:], state)
    for key in ('dis_increment_nl','dis_nl','dis','force_internal','elastic_dis_increment'):
        torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    torch.testing.assert_close(actual['dis_nl'], state['displacement_nl']+actual['dis_increment_nl'].cumsum(1))
    assert not torch.equal(actual['dis_increment_nl'],continued['dis_increment_nl'])
    criterion = EPINN_MDOFSys_DisIncrement_PhyLoss(increment_scale=.1,displacement_scale=.5,local_cumsum_window=32)
    for model in (carry, reset):
        model.zero_grad()
        pred, st = model.forward_chunk(load[...,:32])
        (criterion(pred)*.5).backward()
        pred, st = model.forward_chunk(load[...,32:],st)
        (criterion(pred)*.5).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(not h.requires_grad and not c.requires_grad for h,c in st['temporal'])
    ckpt = dict(n_load=5,n_dof=5,hidden_size=8,fc_size=8,reset_lstm_state=True,model_state_dict=reset.state_dict())
    rebuilt=build_model('increment',ckpt,t,cfg,torch.device('cpu'))
    assert rebuilt.reset_lstm_state
    print('[PASS] Identical initialization and first chunk; only h,c reset in later chunks.')
    print('[PASS] Physical history retained; cumulative displacement continuity; finite gradients; checkpoint rebuild.')


if __name__ == '__main__':
    main()
