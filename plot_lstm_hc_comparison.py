"""Compare the same held-out sample for LSTM cumulative and SCL displacement."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=152, help='MATLAB one-based sample ID.')
    args = parser.parse_args()
    out = args.comparison_dir
    status = json.loads((out/'status.json').read_text())
    arrays, configs, report = {}, {}, {}
    for name in ('carry-hc', 'reset-hc'):
        entry = status['experiments'][name]
        assert entry['status'] == 'complete' and entry['test_exit_code'] == 0
        run = Path(entry['run_dir'])
        arrays[name] = loadmat(run/'test_results/test_results.mat', variable_names=[
            'time','sampleIndexMATLAB','U_pred','U_scl','U_true','checkpointEpoch'])
        configs[name] = json.loads((run/'training_config.json').read_text())
        report[name] = {'checkpoint_epoch': int(arrays[name]['checkpointEpoch'].item())}
    assert configs['carry-hc']['data_split'] == configs['reset-hc']['data_split']
    for name in arrays:
        assert configs[name]['model_and_loss']['physics_evaluation'] == 'full-history'
        assert configs[name]['model_and_loss']['reset_lstm_state'] == (name == 'reset-hc')
    np.testing.assert_array_equal(arrays['carry-hc']['sampleIndexMATLAB'], arrays['reset-hc']['sampleIndexMATLAB'])
    np.testing.assert_array_equal(arrays['carry-hc']['U_true'], arrays['reset-hc']['U_true'])
    np.testing.assert_array_equal(arrays['carry-hc']['time'], arrays['reset-hc']['time'])
    indices = arrays['carry-hc']['sampleIndexMATLAB'].ravel()
    found = np.flatnonzero(indices == args.sample)
    if len(found) != 1:
        raise ValueError('Requested sample is not in the held-out test set.')
    idx = int(found[0])
    t = arrays['carry-hc']['time'].ravel()
    ref = arrays['carry-hc']['U_true']
    sample_mat = {'time': t, 'sampleIndexMATLAB': args.sample, 'U_true': ref[:,:,idx]}
    for key, title in [('U_pred','LSTM increments accumulated to displacement'), ('U_scl','SCL displacement')]:
        fig, axes = plt.subplots(ref.shape[0], 1, figsize=(13,11), sharex=True, squeeze=False)
        for dof, ax in enumerate(axes[:,0]):
            ax.plot(t, ref[dof,:,idx], color='black', lw=1, label='Reference')
            for name, color in [('carry-hc','tab:blue'),('reset-hc','tab:orange')]:
                ax.plot(t, arrays[name][key][dof,:,idx], color=color, lw=.8, label=name)
            ax.set_ylabel('DOF %d (m)'%(dof+1)); ax.grid(alpha=.25)
        axes[0,0].legend(ncol=3)
        axes[0,0].set_title('MATLAB sample %d: %s'%(args.sample,title))
        axes[-1,0].set_xlabel('Time (s)')
        fig.tight_layout()
        fig.savefig(out/('comparison_%s_sample%d.png'%(key,args.sample)), dpi=160)
        plt.close(fig)
        for name in arrays:
            u = arrays[name][key]
            assert np.isfinite(u).all()
            error = u-ref
            p = u-u.mean(axis=1,keepdims=True)
            r = ref-ref.mean(axis=1,keepdims=True)
            den = np.sqrt(np.sum(p*p,axis=1)*np.sum(r*r,axis=1))
            valid = den > 0
            corr = np.sum(p*r,axis=1)[valid]/den[valid]
            report[name][key] = {'overall_rmse_m': float(np.sqrt(np.mean(error**2))),
                'overall_mae_m': float(np.mean(np.abs(error))),
                'mean_sample_dof_correlation': float(np.mean(corr)),
                'sample_rmse_m': float(np.sqrt(np.mean(error[:,:,idx]**2)))}
            sample_mat[name.replace('-','_')+'_'+key] = u[:,:,idx]
    savemat(out/('comparison_sample%d.mat'%args.sample),sample_mat)
    (out/'comparison_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
