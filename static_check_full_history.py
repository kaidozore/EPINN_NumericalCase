"""Check full-history physics ordering, gradients and carry/reset parity."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
from config import CaseConfig
from nets.EPINN_Net import EPINN_PhyLSTM_NetBody
from nets.EPINN_Loss import EPINN_MDOFSys_DisIncrement_PhyLoss
from utils.DataPreProcess import load_case_data, as_torch_case, fixed_dof_response_scales
from utils.utils_fit_EPINN import _run_epoch
from EPINN_MDOFSys_Test import build_model, chunked_batched_predict


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=70)
    p.add_argument('--chunk-length',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--hidden-size',type=int,default=8)
    p.add_argument('--dof-scaling', choices=('uniform','stiffness-profile'), default='uniform')
    args=p.parse_args()
    torch.set_num_threads(2)
    device=torch.device(args.device)
    cfg=CaseConfig(data_root=args.data_root,time_truncation=min(1000,args.steps))
    data=load_case_data(cfg)
    inc_scale, dis_scale = (.1, .5)
    if args.dof_scaling == 'stiffness-profile':
        inc_scale, dis_scale = fixed_dof_response_scales(data.stiffness)
    print('Fixed increment/displacement scales:',inc_scale,dis_scale)
    t=as_torch_case(data,device)
    load=torch.as_tensor(data.load[:args.batch_size,:args.steps].transpose(0,2,1)[:,None],dtype=torch.float64,device=device)
    target_dis=torch.as_tensor(data.displacement[:args.batch_size,:args.steps],dtype=torch.float64,device=device)
    target={'dis':target_dis,'dis_increment':torch.cat([torch.zeros_like(target_dis[:,:1]),target_dis[:,1:]-target_dis[:,:-1]],1),
            'labelled':torch.ones(args.batch_size,dtype=torch.bool,device=device)}
    criterion=EPINN_MDOFSys_DisIncrement_PhyLoss(increment_scale=inc_scale,displacement_scale=dis_scale).to(device)
    for reset in (False,True):
        torch.manual_seed(20260730)
        model=EPINN_PhyLSTM_NetBody(nLoad=5,nLoadNL=5,influence_kernel=t['kernel'],stiffness=t['stiffness'],
            fiber=t['fiber'],steel=t['steel'],hidden_size=args.hidden_size,fc_size=args.hidden_size,
            input_increment_scale=inc_scale,output_increment_scale=inc_scale,input_displacement_scale=dis_scale,
            reset_lstm_state=reset,physics_evaluation='full-history').double().to(device)
        events=[]
        handles=[module.register_forward_hook(lambda m,i,o,name=name: events.append(name))
                 for name,module in [('neural',model.LSTM_Module),('material',model.Constitutive_Module),('scl',model.SCL_Module)]]
        pred=model.forward_full_history(load,args.chunk_length)
        expected_chunks=(args.steps+args.chunk_length-1)//args.chunk_length
        assert events==['neural']*expected_chunks+['material','scl'],events
        assert pred['dis_increment_nl'].requires_grad
        assert all(not pred[k].requires_grad for k in ['dis_nl','dis','force_internal','dis_increment_scl'])
        torch.testing.assert_close(pred['dis_nl'],pred['dis_increment_nl'].detach().cumsum(1))
        # Forward physics equals history-preserving chunk evaluation (no filtering).
        with torch.no_grad():
            st=None
            old=[]
            for start in range(0,args.steps,args.chunk_length):
                q,st=model.forward_chunk(load[...,start:start+args.chunk_length],st)
                old.append(q)
            for key in ['dis_increment_nl','dis_nl','dis','force_internal']:
                torch.testing.assert_close(pred[key],torch.cat([q[key] for q in old],1),rtol=1e-6,atol=1e-6)
        for h in handles: h.remove()
        opt=torch.optim.Adam(model.parameters(),lr=2e-4)
        with patch('torch.autograd.backward',wraps=torch.autograd.backward) as backward, patch.object(opt,'step',wraps=opt.step) as step:
            _run_epoch(model,criterion,[(load,target)],device,opt,args.chunk_length,.2)
            assert backward.call_count==1 and step.call_count==1
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        ckpt=dict(n_load=5,n_dof=5,hidden_size=args.hidden_size,fc_size=args.hidden_size,reset_lstm_state=reset,
                  input_increment_scale=inc_scale,output_increment_scale=inc_scale,input_displacement_scale=dis_scale,
                  physics_evaluation='full-history',model_state_dict=model.state_dict())
        rebuilt=build_model('increment',ckpt,t,cfg,device)
        assert rebuilt.physics_evaluation=='full-history' and rebuilt.reset_lstm_state==reset
        with torch.no_grad():
            a=model.forward_full_history(load,args.chunk_length)
        b=chunked_batched_predict(rebuilt,data.load[:args.batch_size,:args.steps],__import__('numpy').arange(args.batch_size),args.batch_size,args.chunk_length,device)
        torch.testing.assert_close(a['dis_nl'].cpu(),torch.as_tensor(b['dis_nl']))
        print(f'[PASS] reset_hc={reset}: {expected_chunks} neural chunks, one material/SCL call, one backward/step, detached physics, test parity.')
    if device.type=='cuda':
        print('Peak CUDA allocated GiB:',torch.cuda.max_memory_allocated()/1024**3)


if __name__=='__main__':
    main()
