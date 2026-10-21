# Data

Run `python scripts/prepare_data.py` from the repository root. Existing files
are retained; downloaded data are excluded from Git.

| File | Source | Use |
| --- | --- | --- |
| MATH_rldf.json | Official Drift processed MATH | RL and offline POS estimation |
| GSM8K_train.json | openai/gsm8k, main, train | Offline POS estimation |
| MATH500.json | Official Drift benchmark copy | Evaluation |
| gsm8k_test.json | Official Drift benchmark copy | Evaluation |

Drift files are pinned to commit `787d875d7e992e92f883049cef58bcd49ef34844`.
JSON entries contain `question` and `ground_truth_answer` strings. Dataset
licenses apply; see [Drift](https://github.com/ant-research/Drift),
[MATH](https://github.com/hendrycks/math) and
[GSM8K](https://github.com/openai/grade-school-math).
