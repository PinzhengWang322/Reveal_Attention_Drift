Namespace(metric='accuracy', base_path='./results/gpt-j-6B', tojson=False, num=100000, mode='scores')

| Metric | pre | rome-ADR | rome-origin |
|---|---|---|---|
| rewrite | 20.86(1.9) | 99.76(0.2) | 99.88(0.2) |
| p-reasoning | 33.09(1.8) | 50.42(2.0) | 52.39(2.0) |
| p-subject_alias | 23.79(1.7) | 56.40(2.0) | 62.17(2.0) |
| p-paraphrase | 17.70(1.8) | 96.36(0.9) | 99.58(0.3) |
| l-relation | 79.73(2.1) | 27.75(2.3) | 11.94(1.7) |
| l-relation_aft_edit_text | 30.72(2.4) | 14.76(1.8) | 8.76(1.5) |
| l-relation-wrong | 0.00(0.0) | 0.00(0.0) | 0.00(0.0) |
| l-relation_aft_edit_text-wrong | 0.00(0.0) | 0.00(0.0) | 0.00(0.0) |
| l-neighbor | 82.43(1.0) | 80.86(1.0) | 80.26(1.0) |
| l-neighbor_aft_edit_text | 61.99(1.2) | 49.35(1.3) | 30.43(1.2) |
| l-neighbor-wrong | 0.00(0.0) | 0.00(0.0) | 0.00(0.0) |
| l-neighbor_aft_edit_text-wrong | 0.00(0.0) | 0.00(0.0) | 0.00(0.0) |
| ngram_entropy | 621.96(0.9) | 623.00(1.0) | 620.58(1.2) |
| reference_score | 29.99(0.7) | 39.80(0.8) | 42.04(0.9) |
| o-commonsense | 74.73(2.1) | 74.73(2.1) | 74.73(2.1) |
| o-commonsense_aft_edit_text | 74.08(2.1) | 74.49(2.1) | 74.37(2.1) |
