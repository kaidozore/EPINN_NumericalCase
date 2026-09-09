"""Paired incremental LSTM experiment: carry versus reset h,c per chunk."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=200)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shared = ['--data-root', str(args.data_root.resolve()), '--epochs', str(args.epochs),
              '--sequence-variant', 'lstm-hidden', '--batch-size', '10',
              '--hidden-size', '120', '--fc-size', '120', '--learning-rate', '2e-4',
              '--gradient-clip', '0.2', '--tbptt-length', '1000', '--time-truncation', '1000',
              '--labelled-samples', '10', '--increment-loss-weight', '1',
              '--local-cumsum-loss-weight', '0.05', '--local-cumsum-window', '32',
              '--label-increment-loss-weight', '0.2', '--label-local-cumsum-loss-weight', '0.01',
              '--device', 'cuda']
    state = dict(created=datetime.now().isoformat(), pid=os.getpid(),
                 git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
                 shared_arguments=shared, experiments={k:dict(status='queued') for k in ['carry-hc','reset-hc']})
    def save():
        tmp = out/'status.tmp.json'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out/'status.json')
    save()
    for name in state['experiments']:
        item = state['experiments'][name]
        logroot = root/'logs'/('EPINN_PhyLSTM' + ('_reset_hc' if name == 'reset-hc' else ''))
        before = set(logroot.glob('loss_*'))
        cmd = [sys.executable,'-u',str(root/'EPINN_MDOFSys_Train.py'),*shared]
        if name == 'reset-hc':
            cmd.append('--reset-lstm-state')
        item.update(status='training', command=cmd, started=datetime.now().isoformat())
        with (out/(name+'.log')).open('w') as f:
            proc = subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
            item['pid'] = proc.pid
            save()
            code = proc.wait()
        item['training_exit_code'] = code
        candidates = set(logroot.glob('loss_*')) - before
        if code != 0 or len(candidates) != 1:
            item['status'] = 'training_failed' if code else 'run_directory_ambiguous'
            save()
            return
        run = candidates.pop()
        item.update(status='testing',run_dir=str(run))
        save()
        cmd = [sys.executable,'-u',str(root/'EPINN_MDOFSys_Test.py'),'--variant','increment',
               '--data-root',str(args.data_root.resolve()),'--run-dir',str(run),
               '--chunk-length','1000','--batch-size','10','--device','cuda']
        with (out/(name+'_test.log')).open('w') as f:
            code = subprocess.call(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
        item.update(status='complete' if code == 0 else 'test_failed',test_exit_code=code,
                    finished=datetime.now().isoformat())
        save()
        if code:
            return
    state['finished'] = datetime.now().isoformat()
    save()


if __name__ == '__main__':
    main()
