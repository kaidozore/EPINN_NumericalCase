# Increment-response stitching comparison

Run the three non-baseline variants without changing the physical case or
overwriting previous full-response results:

```bash
python run_full_stitch_comparison.py --response-form increment --data-root .. --output-dir comparison_increment_20260909
```

The queue runs Transformer memory, LSTM explicit initial displacement with
overlap, and Transformer explicit initial displacement with overlap. It stops
on a failed job. The original LSTM-hidden baseline is not rerun.

All runs use 200 epochs, 10 labelled samples, batch size 10, hidden/FC size 120,
learning rate 2e-4, gradient clipping 0.2, TBPTT length 1000 and SCL kernel
truncation 1000. Each batch processes the complete 5000-point history with
detached chunk states, accumulating gradients before one optimizer step.
Training data shuffling uses an independent seeded generator across variants.

The input is elastic displacement increment / 0.1 m, and the network output
is multiplied by 0.1 m to obtain the total elastoplastic displacement increment
(not a plastic correction). Displacement is reconstructed by cumulative sum
with the preceding chunk terminal displacement. Steel02/SCL histories persist
across chunks and reset for each new batch.

The loss weights are: increment MSE 1, 32-step local cumulative MSE 0.05,
labelled increment MSE 0.2, labelled local cumulative MSE 0.01. Increment and
local displacement errors use fixed scales 0.1 m and 0.5 m respectively.
SCL targets remain detached. Explicit variants additionally use normalized
boundary increment MSE with weight 1 (increment scale 0.1 m), with the initial
displacement INPUT token scaled by 0.5 m. ALL output tokens are increments.
The extra output repeats the preceding chunk's terminal increment (zero at
the global start); its detached target is carried separately from the initial
displacement. It is used only in the overlap loss, never accumulated twice.
The following 1000 outputs are the new physical time increments. Previous
mixed displacement/increment explicit checkpoints must not be reused.

Tests automatically export both U_pred (integrated network output) and U_scl,
the reference responses, sample indices, metrics and plots. Select checkpoints
by validation loss, not test errors. All configuration and timing is saved.
