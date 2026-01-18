# Distillation Results (gpt2-medium student, gpt2-xl teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | vicuna_gpt4_en |
|---|---|---|---|---|---|
| Teacher | 6 | 0.3100 | 0.4178 | 0.1625 | 0.1046 |
| SFT | 8 | 0.2689 | 0.3359 | 0.1250 | 0.0967 |
| KD_FKL | 9 | 0.2732 | 0.3651 | 0.1239 | 0.1045 |
| KD_RKL | 8 | 0.2841 | 0.3943 | 0.1290 | 0.1182 |
| OD_FKL | 7 | 0.2825 | 0.3518 | 0.1303 | 0.1264 |
| OD_RKL | 7 | 0.2895 | 0.3860 | 0.1259 | 0.1069 |
| JSD_L1.0_B0.9 | 5 | 0.2878 | 0.3831 | 0.1285 | 0.1176 |

# Distillation Results (gpt2-medium student, gpt2-large teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | vicuna_gpt4_en |
|---|---|---|---|---|---|
| Teacher | 9 | 0.2959 | 0.4130 | 0.1533 | 0.1129 |
| SFT | 8 | 0.2689 | 0.3359 | 0.1250 | 0.0967 |
| KD_FKL | 7 | 0.2718 | 0.3588 | 0.1219 | 0.1044 |
| KD_RKL | 9 | 0.2819 | 0.3756 | 0.1307 | 0.1028 |
| OD_FKL | 6 | 0.2819 | 0.3687 | 0.1339 | 0.1011 |
| OD_RKL | 6 | 0.2878 | 0.3851 | 0.1360 | 0.1036 |
| JSD_L1.0_B0.9 | 7 | 0.2882 | 0.3902 | 0.1309 | 0.1064 |

# Distillation Results (gpt2-medium student, gpt2 teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | vicuna_gpt4_en |
|---|---|---|---|---|---|
| Teacher | 9 | 0.2373 | 0.2942 | 0.1001 | 0.1011 |
| SFT | 8 | 0.2689 | 0.3359 | 0.1250 | 0.0967 |
| KD_FKL | 8 | 0.2563 | 0.3245 | 0.1302 | 0.0827 |
| KD_RKL | 9 | 0.2609 | 0.3350 | 0.1226 | 0.0965 |
| OD_FKL | 5 | 0.2573 | 0.3423 | 0.1309 | 0.0865 |
| OD_RKL | 6 | 0.2629 | 0.3682 | 0.1335 | 0.0930 |
| JSD_L1.0_B0.9 | 5 | 0.2608 | 0.3407 | 0.1324 | 0.0900 |







