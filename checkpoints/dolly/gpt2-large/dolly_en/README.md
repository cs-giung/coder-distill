# Distillation Results (gpt2-large student, gpt2-xl teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | snat_instruct_en | tinyalpaca_en | vicuna_gpt4_en |
|---|---|---|---|---|---|---|---|
| Teacher | 6 | 0.3100 | 0.4178 | 0.1625 | | | 0.1046 |
| SFT | 9 | 0.2959 | 0.4130 | 0.1533 | | | 0.1129 |
| KD_FKL | 1 | 0.2936 | 0.4201 | 0.1469 | | | 0.1139 |
| KD_RKL | 5 | 0.2986 | 0.4215 | 0.1501 | | | 0.1222 |
| OD_FKL | 3 | 0.3020 | 0.4332 | 0.1532 | | | 0.1123 |
| OD_RKL | 9 | 0.3050 | 0.4383 | 0.1498 | | | 0.1206 |
| JSD_L1.0_B0.9 | 6 | 0.3055 | 0.4247 | 0.1567 | | | 0.1256 |

# Distillation Results (gpt2-large student, gpt2-medium teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | snat_instruct_en | tinyalpaca_en | vicuna_gpt4_en |
|---|---|---|---|---|---|---|---|
| Teacher | 8 | 0.2689 | 0.3359 | 0.1250 | | | 0.0967 |
| SFT | 9 | 0.2959 | 0.4130 | 0.1533 | | | 0.1129 |
| KD_FKL | 5 | 0.2768 | 0.3793 | 0.1181 | | | 0.1065 |
| KD_RKL | 3 | 0.2832 | 0.3907 | 0.1371 | | | 0.1049 |
| OD_FKL | 8 | 0.2825 | 0.3769 | 0.1268 | | | 0.1004 |
| OD_RKL | 3 | 0.2868 | 0.3734 | 0.1393 | | | 0.1008 |
| JSD_L1.0_B0.9 | 2 | 0.2866 | 0.3716 | 0.1393 | | | 0.1147 |

# Distillation Results (gpt2-large student, gpt2 teacher)

| Method | Epoch | dolly_en | mlqa_en | self_instruct_en | snat_instruct_en | tinyalpaca_en | vicuna_gpt4_en |
|---|---|---|---|---|---|---|---|
| Teacher | 9 | 0.2373 | 0.2942 | 0.1001 | | | 0.1011 |
| SFT | 9 | 0.2959 | 0.4130 | 0.1533 | | | 0.1129 |
| KD_FKL | 3 | 0.2590 | 0.3384 | 0.1139 | | | 0.0958 |
| KD_RKL | 1 | 0.2632 | 0.3610 | 0.1207 | | | 0.1098 |
| OD_FKL | 1 | 0.2696 | 0.3639 | 0.1231 | | | 0.1051 |
| OD_RKL | 6 | 0.2743 | 0.3373 | 0.1204 | | | 0.0971 |
| JSD_L1.0_B0.9 | 0 | 0.2731 | 0.3627 | 0.1226 | | | 0.1064 |