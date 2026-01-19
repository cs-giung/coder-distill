# Distillation Results (gpt2 student, gpt2-xl teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | snat_instruct_en | tinyalpaca_en | vicuna_gpt4_en |
|---|---|---|---|---|---|---|---|
| Teacher | 6 | 0.3100 | 0.4178 | 0.1625 | | | 0.1046 |
| SFT | 9 | 0.2369 | 0.2942 | 0.1002 | | | 0.1011 |
| KD_FKL | 0 | 0.2405 | 0.3142 | 0.1016 | | | 0.1084 |
| KD_RKL | 0 | 0.2446 | 0.3126 | 0.1057 | | | 0.1129 |
| OD_FKL | 0 | 0.2423 | 0.3036 | 0.1029 | | | 0.1068 |
| OD_RKL | 2 | 0.2446 | 0.3117 | 0.1082 | | | 0.1248 |
| JSD_L1.0_B0.9 | 1 | 0.2428 | 0.3107 | 0.1069 | | | 0.1236 |

# Distillation Results (gpt2 student, gpt2-large teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | snat_instruct_en | tinyalpaca_en | vicuna_gpt4_en |
|---|---|---|---|---|---|---|---|
| Teacher | 9 | 0.2959 | 0.4130 | 0.1533 | | | 0.1129 |
| SFT | 9 | 0.2369 | 0.2942 | 0.1002 | | | 0.1011 |
| KD_FKL | 0 | 0.2351 | 0.2990 | 0.0977 | | | 0.1122 |
| KD_RKL | 3 | 0.2422 | 0.3030 | 0.1082 | | | 0.1079 |
| OD_FKL | 0 | 0.2413 | 0.3006 | 0.1013 | | | 0.1205 |
| OD_RKL | 0 | 0.2506 | 0.3125 | 0.1047 | | | 0.1063 |
| JSD_L1.0_B0.9 | 4 | 0.2536 | 0.3421 | 0.1132 | | | 0.1102 |

# Distillation Results (gpt2 student, gpt2-medium teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | snat_instruct_en | tinyalpaca_en | vicuna_gpt4_en |
|---|---|---|---|---|---|---|---|
| Teacher | 8 | 0.2689 | 0.3359 | 0.1250 | | | 0.0967 |
| SFT | 9 | 0.2369 | 0.2942 | 0.1002 | | | 0.1011 |
| KD_FKL | 2 | 0.2347 | 0.2866 | 0.0881 | | | 0.0953 |
| KD_RKL | 1 | 0.2439 | 0.2911 | 0.0947 | | | 0.1028 |
| OD_FKL | 4 | 0.2405 | 0.2943 | 0.0983 | | | 0.0930 |
| OD_RKL | 3 | 0.2521 | 0.3117 | 0.0943 | | | 0.0964 |
| JSD_L1.0_B0.9 | 2 | 0.2498 | 0.3046 | 0.0948 | | | 0.0966 |