
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "test", "ops"))

import llaisys
import torch
from test_utils import random_tensor, arrange_tensor, check_equal, benchmark

from add import test_op_add
from argmax import test_op_argmax
from embedding import test_op_embedding
from linear import test_op_linear
from rms_norm import test_op_rms_norm
from rope import test_op_rope
from self_attention import test_op_self_attention
from swiglu import test_op_swiglu

def run_all_tests(device="cpu", profile=False):
    """
    运行所有算子的测试，使用默认的形状和精度配置。
    """
    print(f"\n{'='*60}")
    print(f"Running all operator tests on device: {device}")
    print(f"{'='*60}\n")


    tests = {
        "add": {
            "func": test_op_add,
            "shapes": [(2, 3), (512, 4096)],
            "dtype_prec": [
                ("f32", 1e-5, 1e-5),
                ("f16", 1e-3, 1e-3),
                ("bf16", 1e-2, 1e-2),
            ]
        },
        "argmax": {
            "func": test_op_argmax,
            "shapes": [(4,), (256,), (1024,)],
            "dtype_prec": [
                ("f32", None, None),  
                ("f16", None, None),
                ("bf16", None, None),
            ]
        },
        "embedding": {
            "func": test_op_embedding,
            "shapes": [  # (idx_shape, embd_shape)
                ((1,), (2, 3)),
                ((16,), (1000, 768)),
            ],
            "dtype_prec": [
                ("f32", 1e-5, 1e-5),
                ("f16", 1e-3, 1e-3),
                ("bf16", 1e-2, 1e-2),
            ]
        },
        "linear": {
            "func": test_op_linear,
            "shapes": [  # (out_shape, x_shape, w_shape, bias_flag)
                ((2, 3), (2, 4), (3, 4), True),
                ((512, 4096), (512, 4096), (4096, 4096), True),
            ],
            "dtype_prec": [
                ("f32", 1e-5, 1e-5),
                ("f16", 1e-3, 1e-3),
                ("bf16", 1e-2, 1e-2),
            ]
        },
        "rms_norm": {
            "func": test_op_rms_norm,
            "shapes": [(1, 4), (128, 768)],
            "dtype_prec": [
                ("f32", 1e-5, 1e-5),
                ("f16", 1e-3, 1e-3),
                ("bf16", 1e-2, 1e-2),
            ]
        },
        "rope": {
            "func": test_op_rope,
            "shapes": [  # (shape, start_end)
                ((2, 1, 4), (0, 2)),
                ((512, 4, 4096), (512, 1024)),
            ],
            "dtype_prec": [
                ("f32", 1e-4, 1e-4),
                ("f16", 1e-3, 1e-3),
                ("bf16", 1e-2, 1e-2),
            ]
        },
        "self_attention": {
            "func": test_op_self_attention,
            "shapes": [  # (qlen, kvlen, nh, nkvh, hd)
                (2, 2, 1, 1, 4),
                (5, 11, 4, 2, 8),
            ],
            "dtype_prec": [
                ("f32", 1e-5, 1e-5),
                ("f16", 1e-3, 1e-3),
                ("bf16", 1e-2, 1e-2),
            ]
        },
        "swiglu": {
            "func": test_op_swiglu,
            "shapes": [(2, 3), (512, 4096)],
            "dtype_prec": [
                ("f32", 1e-5, 1e-5),
                ("f16", 1e-3, 1e-3),
                ("bf16", 1e-2, 1e-2),
            ]
        },
    }

    passed_all = True
    for op_name, config in tests.items():
        print(f"--- Testing {op_name} ---")
        func = config["func"]
        shapes = config["shapes"]
        dtype_prec = config["dtype_prec"]

        for shape in shapes:
            for dtype_name, atol, rtol in dtype_prec:
                try:

                    if op_name == "add":
                        func(shape, dtype_name, atol, rtol, device, profile)
                    elif op_name == "argmax":
                        func(shape, dtype_name, device, profile)
                    elif op_name == "embedding":
                        idx_shape, embd_shape = shape
                        func(idx_shape, embd_shape, dtype_name, device, profile)
                    elif op_name == "linear":
                        out_shape, x_shape, w_shape, bias = shape
                        func(out_shape, x_shape, w_shape, bias, dtype_name, atol, rtol, device, profile)
                    elif op_name == "rms_norm":
                        func(shape, dtype_name, atol, rtol, device, profile)
                    elif op_name == "rope":
                        s, start_end = shape
                        func(s, start_end, dtype_name, atol, rtol, device, profile)
                    elif op_name == "self_attention":
                        func(*shape, dtype_name, atol, rtol, device, profile)
                    elif op_name == "swiglu":
                        func(shape, dtype_name, atol, rtol, device, profile)
                    else:
                        print(f"Unknown operator {op_name}, skipping")
                        continue
                    print(f"  {op_name}: shape={shape} dtype=<{dtype_name}> PASSED")
                except AssertionError:
                    print(f"  {op_name}: shape={shape} dtype=<{dtype_name}> FAILED")
                    passed_all = False
                except Exception as e:
                    print(f"  {op_name}: shape={shape} dtype=<{dtype_name}> ERROR: {e}")
                    passed_all = False

    if passed_all:
        print("\n\033[92mAll tests passed!\033[0m\n")
    else:
        print("\n\033[91mSome tests failed. Check above logs.\033[0m\n")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "iluvatar"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    run_all_tests(device=args.device, profile=args.profile)