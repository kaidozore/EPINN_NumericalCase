"""Run three stitching comparisons sequentially (full or increment response)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--response-form', choices=('full', 'increment'), default='full')
    parser.add_argument('--variants', nargs='+', choices=('transformer-hidden', 'lstm-explicit-overlap', 'transformer-explicit-overlap'), default=None)
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry only failed training jobs, preserving completed results.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.retry_failed)
    variants = ('transformer-hidden', 'lstm-explicit-overlap', 'transformer-explicit-overlap')
    if args.variants is not None:
        variants = tuple(args.variants)
    shared = ['--data-root', str(args.data_root.resolve()), '--epochs', str(args.epochs),
              '--batch-size', '10', '--hidden-size', '120', '--fc-size', '120',
              '--learning-rate', '2e-4', '--gradient-clip', '0.2',
              '--tbptt-length', '1000', '--time-truncation', '1000',
              '--labelled-samples', '10', '--label-weight', '0.1',
              '--continuity-loss-weight', '1.0', '--transformer-layers', '3',
              '--transformer-heads', '4', '--transformer-memory-length', '128',
              '--device', 'cuda']
    if args.response_form == 'increment':
        label_index = shared.index('--label-weight')
        del shared[label_index:label_index + 2]
        shared += ['--increment-loss-weight', '1.0',
                   '--local-cumsum-loss-weight', '0.05', '--local-cumsum-window', '32',
                   '--label-increment-loss-weight', '0.2',
                   '--label-local-cumsum-loss-weight', '0.01']
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    state = {'created': datetime.now().isoformat(), 'pid': os.getpid(),
             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
             'response_form': args.response_form,
             'shared_arguments': shared, 'experiments': {v: {'status': 'queued'} for v in variants}}
    if args.retry_failed:
        prior = json.loads((output / 'status.json').read_text(encoding='utf-8'))
        if prior.get('response_form', 'full') != args.response_form:
            raise ValueError('Retry must keep the original response form.')
        if prior['shared_arguments'] != shared:
            raise ValueError('Retry must keep the original experiment arguments.')
        prior['retry_git_commit'] = state['git_commit']
        prior['pid'] = os.getpid()
        state = prior
        state.pop('finished', None)
        variants = tuple(v for v in variants if state['experiments'][v]['status'] == 'training_failed')

    def save():
        temporary = output / 'status.tmp.json'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(output / 'status.json')

    save()
    for variant in variants:
        item = state['experiments'][variant]
        if args.retry_failed:
            state.setdefault('previous_attempts', []).append({'variant': variant, **item})
        train_script = ('EPINN_MDOFSys_Full_Train.py' if args.response_form == 'full'
                        else 'EPINN_MDOFSys_Train.py')
        command = [sys.executable, '-u', str(root / train_script), *shared,
                   '--sequence-variant', variant]
        item.update(status='training', command=command, started=datetime.now().isoformat())
        prefix = 'EPINN_Full_' if args.response_form == 'full' else 'EPINN_'
        log_root = root / 'logs' / (prefix + variant.replace('-', '_', 1))
        before = set(log_root.glob('loss_*'))
        with (output / (variant + '.log')).open('a' if args.retry_failed else 'w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            item['pid'] = process.pid
            save()
            code = process.wait()
        item['training_exit_code'] = code
        if code != 0:
            item['status'] = 'training_failed'
            save()
            return
        candidates = set(log_root.glob('loss_*')) - before
        if len(candidates) != 1:
            item['status'] = 'run_directory_ambiguous'
            save()
            return
        run_dir = candidates.pop()
        item.update(status='testing', run_dir=str(run_dir))
        save()
        test_command = [sys.executable, '-u', str(root / 'EPINN_MDOFSys_Test.py'),
                        '--variant', args.response_form, '--data-root', str(args.data_root.resolve()),
                        '--run-dir', str(run_dir), '--chunk-length', '1000',
                        '--batch-size', '10', '--device', 'cuda']
        with (output / (variant + '_test.log')).open('w', encoding='utf-8') as stream:
            result = subprocess.run(test_command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        item.update(status='complete' if result.returncode == 0 else 'test_failed',
                    test_exit_code=result.returncode, finished=datetime.now().isoformat())
        save()
        if result.returncode != 0:
            return
    state['finished'] = datetime.now().isoformat()
    save()


if __name__ == '__main__':
    main()
