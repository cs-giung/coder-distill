# Distillation Results (gpt2 student, gpt2-xl teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | vicuna_gpt4_en |
|---|---|---|---|---|---|
| SFT | 9 | 0.2373 | 0.2942 | 0.1001 | 0.1011 |
| KD_FKL | 0 | 0.2405 | 0.3142 | 0.1016 | 0.1084 |
| KD_RKL | 0 | 0.2446 | 0.3126 | 0.1057 | 0.1129 |
| OD_FKL | 0 | 0.2423 | 0.3036 | 0.1029 | 0.1068 |
| OD_RKL | 2 | 0.2446 | 0.3117 | 0.1082 | 0.1248 |

# Distillation Results (gpt2 student, gpt2-large teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | vicuna_gpt4_en |
|---|---|---|---|---|---|
| SFT | 9 | 0.2373 | 0.2942 | 0.1001 | 0.1011 |
| KD_FKL | 0 | 0.2351 | 0.2990 | 0.0977 | 0.1122 |
| KD_RKL | 3 | 0.2422 | 0.3030 | 0.1082 | 0.1079 |
| OD_FKL | 0 | 0.2413 | 0.3006 | 0.1013 | 0.1205 |
| OD_RKL | 0 | 0.2506 | 0.3125 | 0.1047 | 0.1063 |

