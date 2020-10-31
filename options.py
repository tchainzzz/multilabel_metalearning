from argparse import ArgumentParser

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--support-size", type=int, default=8, help="# of examples to load for each support set. For meta-validation, query set size is equal to support size.")
    psr.add_argument("--inner-update-lr", type=float, default=0.4, help="Inner optimization learning rate initialization. Constant if --learn-inner-lr not specified.")
    psr.add_argument("--num_inner_updates", type=int, default=1, help="Number of inner optimization steps (MAML).")
    psr.add_argument("--iterations", type=int, default=4000, help="Number of outer optimization iterations (MAML).")
    psr.add_argument("--learn-inner-lr", action='store_true', help="Whether to dynamically update the inner MAML learning rate.")
    psr.add_argument("--bs", "--batch-size", type=int, default=8, help="Meta-batch size (# of size-N disjoint label subsets.")
    psr.add_argument("--label-subset-size", type=int, default=3, help="Maximum cardinality of multi-labels in each meta-example (task).")
    psr.add_argument("--multilabel-scheme", choices=['powerset', 'binary'], help="Use powerset or binary relevance label transformation", default='powerset')
    psr.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model.")
    psr.add_argument("--log-frequency", type=int, default=5, help="How often to print meta-train/val results")
    psr.add_argument("--test-log-frequency", type=int, default=25, help="How often to print meta-test results")
    return psr.parse_args()