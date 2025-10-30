# SWE-bench-experimentation

This repo contains various experimentations around agentic coding evals.

## Simple instructions for running SWE-bench locally

This is based off the great work from Epoch AI team
(https://epoch.ai/blog/swebench-docker) that publish the images for swe bench
evals.

We go a step further in adding simple utilities + instructions for anyone to be
able to run SWE bench tests on their local machine.

Example SWE Bench test validation on a run result for astropy__astropy-13236
test. In this example, we copy the golden solution for astropy__astropy-13236
from https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified (field
`patch`.)

Prerequisites:
- Docker installed
- Python 3.10+
- swebench installed by following https://github.com/SWE-bench/SWE-bench?tab=readme-ov-file#-set-up

```bash
$ docker pull ghcr.io/epoch-research/swe-bench.eval.x86_64.astropy__astropy-13236:latest
$ python run_swe_bench_with_existing_image.py \
     --image_name "ghcr.io/epoch-research/swe-bench.eval.x86_64.astropy__astropy-13236:latest" \
     --instance_id "astropy__astropy-13236" \
     --predictions_path "astropy-13236-golden-solution.jsonl" \
     --run_id "my_astropy_test_run" \
     --timeout 300
```

TODO: Add option to run an agent harness with the existing image.