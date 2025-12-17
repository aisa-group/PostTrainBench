#!/bin/bash

condor_submit_bid 50 -a "eval=humaneval" -a "eval_dir=..." dev_utils/test_evaluation/single_evaluation.sub
