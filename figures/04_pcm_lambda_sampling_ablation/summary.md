# Experiment 04: spectral-parameter sampling-cadence ablation

This experiment repeats the three principal-chiral-model hard-parametrization loss constructions under two choices for the spectral-parameter batch.

- `fixed`: the 64 spectral-parameter values are sampled once and kept for the entire run; the exponential moving average in the adaptive method follows a fixed spectral-parameter value.
- `resampled`: 64 new spectral-parameter values are drawn at every optimization step; the exponential moving average follows a batch index rather than a persistent point of the spectral-parameter plane.

The collaborator note specifies uniform spectral-parameter sampling and an exponential moving average indexed by lambda_n, but it does not explicitly specify the resampling cadence. The two conventions are therefore treated as an implementation ablation rather than as two claims about the collaborator code.

| sampling | method | c relative L2 | gap mean | gap max |
| --- | --- | ---: | ---: | ---: |
| fixed | Bare flatness loss | 2.261760e-02 | 6.956254e-03 | 9.249240e-02 |
| fixed | Analytic per-spectral-parameter weight | 1.690099e-02 | 5.810227e-03 | 7.644533e-02 |
| fixed | Component normalization + adaptive weight | 1.414548e-02 | 6.192462e-03 | 6.215910e-02 |
| resampled | Bare flatness loss | 2.655439e-02 | 9.630971e-03 | 1.032098e-01 |
| resampled | Analytic per-spectral-parameter weight | 9.261880e-03 | 5.681147e-03 | 3.722395e-02 |
| resampled | Component normalization + adaptive weight | 1.712904e-02 | 1.441959e-02 | 7.835089e-02 |

## Collaborator-note values at 10000 steps

| method | c relative L2 | gap mean | gap max |
| --- | ---: | ---: | ---: |
| Bare flatness loss | 1.700000e-02 | 5.900000e-03 | 7.000000e-02 |
| Analytic per-spectral-parameter weight | 7.600000e-03 | 5.900000e-03 | 3.000000e-02 |
| Component normalization + adaptive weight | 1.100000e-02 | 9.700000e-03 | 5.500000e-02 |

The purpose of this ablation is to determine which conclusions are stable under an implementation detail that is not fixed by the available note. In particular, the relative ordering of the analytic weighting and the adaptive weighting should not be treated as reproduced unless it is stable or the original sampling cadence is known.
