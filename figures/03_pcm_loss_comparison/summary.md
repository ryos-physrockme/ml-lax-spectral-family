# Experiment 03: principal chiral model loss comparison

The coefficient a(lambda) is fixed to a(lambda)=lambda. A neural network learns c(lambda) on the complex rectangle Re(lambda) in [-2,0], Im(lambda) in [-1,1].
The analytic solution c(lambda)=lambda/(2 lambda-1) is used only for evaluation.

The 3000-step entries below are checkpoints of the 10000-step run and therefore do not reproduce a separately scheduled 3000-step optimization.

| method | step | c relative L2 | gap mean | gap max |
| --- | ---: | ---: | ---: | ---: |
| Bare flatness loss | 3000 | 4.786500e-02 | 1.999248e-02 | 1.851271e-01 |
| Bare flatness loss | 10000 | 2.261760e-02 | 6.956254e-03 | 9.249240e-02 |
| Analytic per-lambda weight | 3000 | 3.291873e-02 | 1.683805e-02 | 1.118963e-01 |
| Analytic per-lambda weight | 10000 | 1.690099e-02 | 5.810227e-03 | 7.644533e-02 |
| Component normalization + adaptive weight | 3000 | 2.513771e-02 | 1.711435e-02 | 1.017569e-01 |
| Component normalization + adaptive weight | 10000 | 1.414548e-02 | 6.192446e-03 | 6.215913e-02 |

## Values reported in the collaborator note at 10000 steps

| method | c relative L2 | gap mean | gap max |
| --- | ---: | ---: | ---: |
| Bare flatness loss | 1.700000e-02 | 5.900000e-03 | 7.000000e-02 |
| Analytic per-lambda weight | 7.600000e-03 | 5.900000e-03 | 3.000000e-02 |
| Component normalization + adaptive weight | 1.100000e-02 | 9.700000e-03 | 5.500000e-02 |

The source note mentions additional smoothness and parameter-L2 regularization terms with weight 1e-4, but does not specify the smoothness functional in enough detail to reproduce it independently. This implementation therefore compares the three reported flatness-loss constructions without those small regularizers.
