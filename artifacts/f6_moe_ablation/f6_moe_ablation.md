# F6 MoE balancing ablation results

| arm | granularity | val_ce | perplexity | entropy | balance_score |
| --- | --- | --- | --- | --- | --- |
| bias_free | coarse | 1.9282 | 6.88 | 2.7000 | 0.9738 |
| bias_free | fine | 2.2956 | 9.93 | 4.0942 | 0.9844 |
| seq_aux | coarse | 1.9265 | 6.87 | 2.6888 | 0.9698 |
| seq_aux | fine | 2.2995 | 9.97 | 4.0558 | 0.9752 |
| none | coarse | 1.9268 | 6.87 | 2.6918 | 0.9709 |
| none | fine | 2.2999 | 9.97 | 4.0568 | 0.9755 |
