#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Generate repository-style kernel skeletons."""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
KERNEL_DIR = ROOT / "agent" / "example" / "kernels"
VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_name(name: str) -> str:
    base = name[:-3] if name.endswith(".py") else name
    if not VALID_NAME_RE.match(base):
        raise ValueError(f"Invalid kernel name: {name!r}. Use a valid Python identifier-style basename.")
    return base


def _formula_comment(formula: str) -> str:
    if not formula.strip():
        return "# formula:\n#   TODO: fill exact PyTorch contract"
    lines = formula.strip().splitlines()
    body = "\n".join(f"#   {line}" for line in lines)
    return f"# formula:\n{body}"


def _layout_comment(layout: str) -> str:
    if layout == "mknk":
        return "# layout:\n#   x: [M, K]\n#   y: [N, K]\n#   z: [M, N]"
    if layout == "kmkn":
        return "# layout:\n#   x: [K, M]\n#   y: [K, N]\n#   z: [M, N]"
    return "# layout:\n#   TODO: keep this explicit for the target contract"


def _grid_config(grid_mode: str, m_split: Optional[int], n_split: Optional[int]) -> Tuple[str, str, str]:
    if grid_mode == "tile-m":
        return (
            "    m_split = Var(GetCubeNum())\n    n_split = Var(1)",
            "    cube_row = Var(GetCubeIdx())\n    cube_col = Var(0)",
            "# grid mode:\n#   tile-m",
        )
    if grid_mode == "tile-n":
        return (
            "    m_split = Var(1)\n    n_split = Var(GetCubeNum())",
            "    cube_row = Var(0)\n    cube_col = Var(GetCubeIdx())",
            "# grid mode:\n#   tile-n",
        )
    if grid_mode == "mix":
        if m_split is None or n_split is None:
            raise ValueError("--grid-mode mix requires both --m-split and --n-split.")
        if m_split <= 0 or n_split <= 0:
            raise ValueError("--m-split and --n-split must be positive integers.")
        return (
            f"    m_split = Var({m_split})\n    n_split = Var({n_split})",
            "    cube_row = Var(GetCubeIdx() // n_split)\n    cube_col = Var(GetCubeIdx() % n_split)",
            "# split mode:\n#   mix (2d grid)",
        )
    raise ValueError(f"Unsupported grid mode: {grid_mode}")


def _k_loop_mode_comment(k_loop_mode: str) -> str:
    return f"# k-loop mode:\n#   {k_loop_mode}"


def _resolve_k_loop_mode(k_loop_mode: str) -> str:
    if k_loop_mode == "auto":
        return "always"
    return k_loop_mode


def _cube_stage_block(
    resolved_k_loop_mode: str, layout: str, dest_expr: str, extra_note: str = "",
) -> Tuple[str, str, str]:
    if resolved_k_loop_mode == "always":
        load_note = (
            "                    # TODO:\n"
            "                    # - confirm exact operand layout\n"
            "                    # - if this is MKNK, keep x[m0:..., k0:...] and y[n0:..., k0:...]\n"
            "                    # - if this is KMKN, switch slices and transpose only at the matmul call site"
        )
        x_load = "                    l1x[l1_cnt] <<= x[m0:m0 + valid_m, k0:k0 + valid_k]"
        y_load = "                    l1y[l1_cnt] <<= y[n0:n0 + valid_n, k0:k0 + valid_k]"
        if layout == "kmkn":
            x_load = "                    l1x[l1_cnt] <<= x[k0:k0 + valid_k, m0:m0 + valid_m]"
            y_load = "                    l1y[l1_cnt] <<= y[k0:k0 + valid_k, n0:n0 + valid_n]"
        block = "\n".join(
            [
                "                for k0 in range(0, K, TILE_K):",
                "                    valid_k = Min(TILE_K, K - k0)",
                "",
                load_note,
                x_load,
                y_load,
                "",
                "                    # TODO:",
                "                    # - choose exact cube call: plain matmul / splitn / splitk",
                "                    # - keep exact init semantics when K is tiled",
                f"                    matmul({dest_expr}, l1x[l1_cnt], l1y[l1_cnt], is_init=(k0 == 0))",
            ]
        )
        tile_k_decl = "TILE_K = 128"
        k_loop_note = (
            "# - K is currently scaffolded as a tiled loop;"
            " switch to --k-loop-mode never if the template should be one-shot"
        )
        if extra_note:
            k_loop_note += f"\n{extra_note}"
        return block, tile_k_decl, k_loop_note

    load_note = (
        "                # TODO:\n"
        "                # - confirm exact operand layout\n"
        "                # - if this is MKNK, load directly from x[m0:..., 0:K] and y[n0:..., 0:K]\n"
        "                # - if this is KMKN, switch slices and transpose only at the matmul call site\n"
        "                # - make TILE_K match the intended one-shot K width before treating this as runnable"
    )
    x_load = "                l1x[l1_cnt] <<= x[m0:m0 + valid_m, 0:K]"
    y_load = "                l1y[l1_cnt] <<= y[n0:n0 + valid_n, 0:K]"
    if layout == "kmkn":
        x_load = "                l1x[l1_cnt] <<= x[0:K, m0:m0 + valid_m]"
        y_load = "                l1y[l1_cnt] <<= y[0:K, n0:n0 + valid_n]"
    block = "\n".join(
        [
            load_note,
            x_load,
            y_load,
            "",
            "                # TODO:",
            "                # - choose exact cube call: plain matmul / splitn / splitk",
            "                # - add a K loop later only if the kernel truly needs tiled accumulation",
            f"                matmul({dest_expr}, l1x[l1_cnt], l1y[l1_cnt])",
        ]
    )
    tile_k_decl = "TILE_K = 128  # TODO: set this to the intended one-shot K width or re-enable the K loop"
    k_loop_note = (
        "# - this scaffold intentionally omits a K loop;"
        " make TILE_K match the intended one-shot K width or add the loop later"
    )
    if extra_note:
        k_loop_note += f"\n{extra_note}"
    return block, tile_k_decl, k_loop_note


def _simple_preprocess_vf_stub(extra_note: str = "") -> str:
    lines = [
        "@vf()",
        "def preprocess_vf(src: Tensor, dst: Tensor, n_loops: Var):",
        "    # TODO:",
        "    # - implement the exact vec-side preprocess",
        "    # - examples: abs+sqrt, cast, scaling, rowwise transform",
    ]
    if extra_note:
        lines.append(f"    # - {extra_note}")
    lines.append("    pass")
    return "\n".join(lines)


def _simple_postprocess_vf_stub(extra_note: str = "") -> str:
    lines = [
        "@vf()",
        "def postprocess_vf(src: Tensor, dst: Tensor, n_loops: Var):",
        "    # TODO:",
        "    # - implement the exact vec-side postprocess",
        "    # - examples: add bias, abs + 1, cast, scale",
    ]
    if extra_note:
        lines.append(f"    # - {extra_note}")
    lines.append("    pass")
    return "\n".join(lines)


def _lookahead_common_notes(first_line: str, edge_line: str, extra_lines: list[str]) -> str:
    lines = [
        first_line,
        "# - this is a pipeline skeleton with running state and a delayed consumer,"
        " not a generic four-stage fusion template",
        edge_line,
        "# - this template assumes one-tile lookahead: stage1 runs at block t, stage2 drains block t - 1",
        "# - keep stage1_cnt and stage2_cnt separate unless you can prove the lifetimes are identical",
        "# - running state lives across iterations; do not treat it like a throwaway scratch buffer",
        "# - running state is sharded by split-M vec-side row ownership; each subblock keeps only its own row slice",
        "# - this v1 scaffold keeps both cube stages one-shot;"
        " add per-stage tiled-K loops only when a real kernel needs them",
        "# - this v1 scaffold uses split-M style vec-side row ownership via `GetSubBlockIdx()` slices;"
        " fork a second profile if a real kernel wants single-owner or another ownership scheme",
        *extra_lines,
    ]
    return "\n".join(lines)


def _kmkn_main_stub(name: str) -> str:
    return f'''

SHAPE_BINDINGS = {{
    0: [2, 0],  # x[K, M]
    1: [2, 1],  # y[K, N]
    2: [0, 1],  # z[M, N]
}}

def _check_dim_alignment(m: int, n: int, k: int) -> None:
    if m % 16 != 0:
        raise ValueError(f"M must be divisible by 16 for transpose matmul path, got M={{m}}")
    if n % 16 != 0:
        raise ValueError(f"N must be divisible by 16 for transpose matmul path, got N={{n}}")
    if k % 16 != 0:
        raise ValueError(f"K must be divisible by 16 for transpose matmul path, got K={{k}}")

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # TODO:
    # - choose one aligned case and one tail case
    # - if grid mode is mix, pick a shape that actually exercises both axes
    M = 256
    N = 256
    K = 128

    _check_dim_alignment(M, N, K)

    x = torch.randn((K, M), dtype=torch.float16)
    y = torch.randn((K, N), dtype=torch.float16)
    z = torch.zeros((M, N), dtype=torch.float32)

    # TODO: keep this reference exactly aligned with the chosen layout
    z_ref = x.float().t() @ y.float()

    z_kernel = OpExec({name}_kernel, simulator=True)(x, y, z, M, N, K, shape_bindings=SHAPE_BINDINGS)

    torch.testing.assert_close(z_kernel, z_ref, rtol=1e-3, atol=1e-3)
    print(f"max_abs_diff={{torch.abs(z_kernel - z_ref).max().item():.6e}}")
'''


def _cube_vec_cube_vec_main_stub(name: str) -> str:
    return f'''

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # TODO:
    # - replace this smoke stub with the exact staged streaming reference before trusting results
    # - add aligned and tail cases once the stage semantics are fixed
    M = 16
    T = 4
    K1 = 128
    S1 = 256
    N2 = 512

    lhs1 = torch.randn((M, K1), dtype=torch.float16)
    rhs1_stream = torch.randn((T, S1, K1), dtype=torch.float16)
    rhs2_stream = torch.randn((T, N2, S1), dtype=torch.float16)
    out = torch.zeros((M, N2), dtype=torch.float32)

    out_kernel = OpExec({name}_kernel, simulator=True)(lhs1, rhs1_stream, rhs2_stream, out, M, T, K1, S1, N2)
    print(out_kernel.shape)
'''


def _vec_cube_vec_cube_main_stub(name: str) -> str:
    return f'''

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # TODO:
    # - replace this smoke stub with the exact staged streaming reference before trusting results
    # - add aligned and tail cases once the stage semantics are fixed
    T = 4
    M = 16
    K1 = 128
    S1 = 256
    N2 = 512

    src_stream = torch.randn((T, M, K1), dtype=torch.float16)
    rhs1 = torch.randn((S1, K1), dtype=torch.float16)
    rhs2 = torch.randn((N2, S1), dtype=torch.float16)
    out = torch.zeros((T, M, N2), dtype=torch.float32)

    out_kernel = OpExec({name}_kernel, simulator=True)(src_stream, rhs1, rhs2, out, T, M, K1, S1, N2)
    print(out_kernel.shape)
'''


def _main_stub(name: str, topology: str, layout: str) -> str:
    if topology == "cube->vec->cube->vec":
        return _cube_vec_cube_vec_main_stub(name)
    if topology == "vec->cube->vec->cube":
        return _vec_cube_vec_cube_main_stub(name)
    if layout == "kmkn":
        return _kmkn_main_stub(name)

    input_block = (
        "    x = torch.randn((M, K), dtype=torch.float16)\n"
        "    y = torch.randn((N, K), dtype=torch.float16)\n"
        "    z = torch.zeros((M, N), dtype=torch.float32)\n\n"
        "    # TODO: keep this reference exactly aligned with the chosen layout\n"
        "    z_ref = x.float() @ y.float().t()\n"
    )

    if topology == "cube->vec":
        input_block = input_block.replace(
            "z_ref =",
            "# TODO: replace this with the exact postprocess contract\n    z_ref =",
        )
    if topology == "vec->cube":
        input_block = input_block.replace(
            "z_ref =",
            "# TODO: replace this with the exact preprocess-then-matmul contract\n    z_ref =",
        )
    if topology == "vec->cube->vec":
        input_block = input_block.replace(
            "z_ref =",
            "# TODO: replace this with the exact preprocess-then-matmul-then-postprocess contract\n    z_ref =",
        )

    return f'''

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # TODO:
    # - choose one aligned case and one tail case
    # - if grid mode is mix, pick a shape that actually exercises both axes
    M = 256
    N = 256
    K = 128

{input_block}
    z_kernel = OpExec({name}_kernel, simulator=True)(x, y, z, M, N, K)

    torch.testing.assert_close(z_kernel, z_ref, rtol=1e-3, atol=1e-3)
    print(f"max_abs_diff={{torch.abs(z_kernel - z_ref).max().item():.6e}}")
'''


def render_cube_only(
    name: str, formula: str, layout: str, grid_mode: str,
    m_split: Optional[int], n_split: Optional[int],
    profile: str, k_loop_mode: str, with_main: bool,
) -> str:
    grid_assign, cube_axes, grid_comment = _grid_config(grid_mode, m_split, n_split)
    resolved_k_loop_mode = _resolve_k_loop_mode(k_loop_mode)
    resolved_profile = profile or ""
    if resolved_profile not in ["", "splitk", "kmkn"]:
        raise ValueError("--topology cube-only currently supports only --profile splitk|kmkn")

    if resolved_profile == "splitk":
        if resolved_k_loop_mode == "always":
            k_loop_block = '''                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)

                    # TODO:
                    # - confirm exact operand layout for the split-k path
                    # - if this is MKNK, keep x[m0:..., k0:...] and y[n0:..., k0:...]
                    # - if this is KMKN, switch slices and transpose only at the matmul call site
                    l1x[l1_cnt] <<= x[m0:m0 + valid_m, k0:k0 + valid_k]
                    l1y[l1_cnt] <<= y[n0:n0 + valid_n, k0:k0 + valid_k]

                    # TODO:
                    # - replace this placeholder with the exact split-k matmul call used by the target kernel
                    # - keep split-k legality and accumulation semantics explicit
                    # - this is not a generic plain/splitn choice point
                    matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[l1_cnt], l1y[l1_cnt], is_init=(k0 == 0))'''
            tile_k_decl = (
                "TILE_K = 256\n"
                "SPLIT_K = 2  # TODO: replace with the intended split-k factor"
            )
            k_loop_note = (
                "# - this scaffold is for split-k style cube matmul, not a generic tiled-K baseline\n"
                "# - choose SPLIT_K explicitly and keep its legality aligned with the real kernel\n"
                "# - here split-k coexists with an outer GM->L1 TILE_K loop"
            )
        else:
            k_loop_block = '''                # TODO:
                # - confirm exact operand layout for the split-k path
                # - if this is MKNK, keep x[m0:..., 0:K] and y[n0:..., 0:K]
                # - if this is KMKN, switch slices and transpose only at the matmul call site
                l1x[l1_cnt] <<= x[m0:m0 + valid_m, 0:K]
                l1y[l1_cnt] <<= y[n0:n0 + valid_n, 0:K]

                # TODO:
                # - replace this placeholder with the exact split-k matmul call used by the target kernel
                # - keep split-k legality and accumulation semantics explicit
                # - split-k here is the internal matmul computation loop, even though the outer GM->L1 path is one-shot over K
                matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[l1_cnt], l1y[l1_cnt])'''
            tile_k_decl = (
                "TILE_K = 256  # TODO: set this to the intended one-shot K width\n"
                "SPLIT_K = 2  # TODO: replace with the intended split-k factor"
            )
            k_loop_note = (
                "# - this scaffold is for split-k style cube matmul, not a generic tiled-K baseline\n"
                "# - choose SPLIT_K explicitly and keep its legality aligned with the real kernel\n"
                "# - here split-k is internal to the matmul while the outer GM->L1 path is one-shot over K"
            )
        profile_comment = "# profile:\n#   splitk\n#\n"
        profile_notes = (
            "# - keep split-k accumulation/init semantics explicit"
            " instead of treating this like plain matmul\n"
            "# - if downstream merge or postprocess depends on split-k,"
            " keep that coupling visible in comments and validation"
        )
        l1_buffer_decls = (
            "    l1x = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)\n"
            "    l1y = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)"
        )
    elif resolved_profile == "kmkn":
        if layout != "kmkn":
            raise ValueError("--topology cube-only --profile kmkn requires --layout kmkn")
        if resolved_k_loop_mode == "always":
            k_loop_block = '''                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)

                    # TODO:
                    # - KMKN path keeps x/y staged as [K, M] / [K, N] in L1
                    # - keep the transpose-at-call-site form explicit instead of hiding it in a generic comment
                    # - if the target kernel needs shape bindings or stricter alignment guards, add them near the runnable stub
                    l1x[l1_cnt][0:valid_k, 0:valid_m] <<= x[k0:k0 + valid_k, m0:m0 + valid_m]
                    l1y[l1_cnt][0:valid_k, 0:valid_n] <<= y[k0:k0 + valid_k, n0:n0 + valid_n]

                    # TODO:
                    # - keep transpose matmul semantics explicit for KMKN
                    # - pass aligned m/n/k arguments instead of assuming implicit shape inference
                    matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[l1_cnt].T, l1y[l1_cnt].T,
                           m=valid_m, n=valid_n, k=valid_k, is_init=(k0 == 0))'''
            tile_k_decl = "TILE_K = 128"
            k_loop_note = (
                "# - this scaffold is for KMKN-style staged operands,"
                " not a generic comment-only layout hint\n"
                "# - keep x/y loads as [K, M] / [K, N] and transpose only at the matmul call site\n"
                "# - if the target kernel has stricter alignment guards,"
                " keep them explicit in validation"
            )
        else:
            k_loop_block = '''                # TODO:
                # - KMKN path keeps x/y staged as [K, M] / [K, N] in L1
                # - this one-shot form only makes sense if TILE_K matches the intended full-K staging width
                # - if the target kernel needs shape bindings or stricter alignment guards, add them near the runnable stub
                l1x[l1_cnt][0:K, 0:valid_m] <<= x[0:K, m0:m0 + valid_m]
                l1y[l1_cnt][0:K, 0:valid_n] <<= y[0:K, n0:n0 + valid_n]

                # TODO:
                # - keep transpose matmul semantics explicit for KMKN
                # - pass aligned m/n/k arguments instead of assuming implicit shape inference
                matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[l1_cnt].T, l1y[l1_cnt].T, m=valid_m, n=valid_n, k=K)'''
            tile_k_decl = "TILE_K = 128  # TODO: set this to the intended one-shot K width or re-enable the K loop"
            k_loop_note = (
                "# - this scaffold is for KMKN-style staged operands,"
                " not a generic comment-only layout hint\n"
                "# - the one-shot form still keeps [K, M] / [K, N] staging"
                " plus transpose-at-call-site explicit\n"
                "# - make TILE_K match the intended full-K width before treating this as runnable"
            )
        profile_comment = "# profile:\n#   kmkn\n#\n"
        profile_notes = (
            "# - keep KMKN transpose-at-call-site semantics explicit"
            " instead of burying them in layout comments\n"
            "# - if the real kernel depends on alignment guards or shape bindings,"
            " keep that coupling visible in validation and the runnable stub"
        )
        l1_buffer_decls = (
            "    l1x = DBuff(DT.half, [TILE_K, TILE_M], Position.L1)\n"
            "    l1y = DBuff(DT.half, [TILE_K, TILE_N], Position.L1)"
        )
    else:
        k_loop_block, tile_k_decl, k_loop_note = _cube_stage_block(
            resolved_k_loop_mode, layout, "l0c[tile_cnt][:valid_m, :valid_n]",
        )
        profile_comment = ""
        profile_notes = (
            "# - decide plain matmul / splitn / splitk explicitly instead of inheriting fake defaults"
        )
        l1_buffer_decls = (
            "    l1x = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)\n"
            "    l1y = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)"
        )

    main_block = _main_stub(name, "cube-only", layout) if with_main else ""

    return f'''from easyasc.a5 import *


{_formula_comment(formula)}
#
# topology:
#   cube-only
#
{profile_comment}{grid_comment}
#
{_k_loop_mode_comment(resolved_k_loop_mode)}
#
{_layout_comment(layout)}
#
# notes:
# - this skeleton follows the repository's tiled cube template
# - choose TILE_M / TILE_N before filling the body
{k_loop_note}
{profile_notes}
# - keep accumulation in float unless there is a strong reason not to


TILE_M = 128
TILE_N = 128
{tile_k_decl}


@kernel()
def {name}_kernel(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
{l1_buffer_decls}
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    l1_cnt = Var(0)
    tile_cnt = Var(0)

    tile_m = CeilDiv(M, TILE_M)
    tile_n = CeilDiv(N, TILE_N)

{grid_assign}

    tile_m_per_core = CeilDiv(tile_m, m_split)
    tile_n_per_core = CeilDiv(tile_n, n_split)

{cube_axes}

    tile_m_begin = Var(tile_m_per_core * cube_row)
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    tile_n_begin = Var(tile_n_per_core * cube_col)
    tile_n_end = Min(tile_n_begin + tile_n_per_core, tile_n)

    with auto_sync():
        for mt in range(tile_m_begin, tile_m_end):
            m0 = Var(mt * TILE_M)
            valid_m = Min(TILE_M, M - m0)

            for nt in range(tile_n_begin, tile_n_end):
                n0 = Var(nt * TILE_N)
                valid_n = Min(TILE_N, N - n0)

{k_loop_block}

                # TODO:
                # - verify direct float writeback is the intended contract
                # - if not, stop and switch topology instead of forcing the cast here
                z[m0:m0 + valid_m, n0:n0 + valid_n] <<= l0c[tile_cnt][:valid_m, :valid_n]

                l1_cnt += 1
                tile_cnt += 1

    return z{main_block}'''


def render_cube_vec(
    name: str, formula: str, layout: str, grid_mode: str,
    m_split: Optional[int], n_split: Optional[int],
    profile: str, k_loop_mode: str, with_main: bool,
) -> str:
    grid_assign, cube_axes, grid_comment = _grid_config(grid_mode, m_split, n_split)
    resolved_k_loop_mode = _resolve_k_loop_mode(k_loop_mode)
    main_block = _main_stub(name, "cube->vec", layout) if with_main else ""
    resolved_profile = profile or "simple-post"
    if resolved_profile not in [
        "simple-post", "half-row-post", "normalize-simple", "normalize-two-pass", "dual-output",
    ]:
        raise ValueError(
            "--topology cube->vec currently supports only"
            " --profile simple-post|half-row-post|normalize-simple|normalize-two-pass|dual-output"
        )
    profile_comment = f"# profile:\n#   {resolved_profile}"
    extra_note = (
        "# - vec postprocess ownership starts after the FIX->V handoff;"
        " keep that lifetime separate from the cube stage if needed"
    )
    k_loop_block, tile_k_decl, k_loop_note = _cube_stage_block(
        resolved_k_loop_mode, layout, "l0c[tile_cnt][:valid_m, :valid_n]", extra_note=extra_note,
    )

    if resolved_profile == "simple-post":
        profile_notes = (
            "# - this is the simple vec postprocess path; use it for bias/add/abs/cast style post stages\n"
            "# - keep vec-side ownership explicit instead of assuming split-M half-row slicing by default"
        )
        helper_block = _simple_postprocess_vf_stub()
        extra_buffers = (
            "    mid_ub = DBuff(DT.float, [TILE_M, TILE_N], Position.UB)\n"
            "    out_ub = DBuff(DT.float, [TILE_M, TILE_N], Position.UB)"
        )
        postprocess_block = '''                cvmutex.lock()

                # TODO:
                # - verify the intended cube -> vec handoff tile shape
                # - keep this generic whole-tile path unless the profile explicitly changes vec-side ownership
                mid_ub[tile_cnt] <<= l0c[tile_cnt]

                cvmutex.ready()
                cvmutex.wait()

                # TODO:
                # - compute correct vec loop count from rows/cols and dtype width
                post_loops = Var(valid_m * TILE_N // 64)
                postprocess_vf(mid_ub[tile_cnt], out_ub[tile_cnt], post_loops)

                cvmutex.free()

                z[m0:m0 + valid_m, n0:n0 + valid_n] <<= out_ub[tile_cnt][0:valid_m, 0:valid_n]'''
    elif resolved_profile == "half-row-post":
        profile_notes = "# - this profile keeps the repository's common split-M half-row vec-side ownership explicit"
        helper_block = _simple_postprocess_vf_stub()
        extra_buffers = (
            "    mid_ub = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)\n"
            "    out_ub = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)"
        )
        postprocess_block = '''                cvmutex.lock()

                # TODO:
                # - verify the intended cube -> vec handoff tile shape
                # - keep split-M half-row ownership explicit in this profile
                mid_ub[tile_cnt] <<= l0c[tile_cnt]

                cvmutex.ready()
                cvmutex.wait()

                # TODO:
                # - compute correct vec loop count from rows/cols and dtype width
                post_loops = Var(TILE_M // 2 * TILE_N // 64)
                postprocess_vf(mid_ub[tile_cnt], out_ub[tile_cnt], post_loops)

                cvmutex.free()

                half_valid = CeilDiv(valid_m, 2)
                sb_idx = GetSubBlockIdx()
                row_begin = sb_idx * half_valid
                if row_begin < valid_m:
                    row_end = Min(row_begin + half_valid, valid_m)
                    row_count = row_end - row_begin
                    z[m0 + row_begin:m0 + row_end, n0:n0 + valid_n] <<= out_ub[tile_cnt][0:row_count, 0:valid_n]'''
    elif resolved_profile == "normalize-simple":
        profile_notes = (
            "# - this profile uses row-sum scratch plus immediate normalize placeholder in the same pass\n"
            "# - if the real kernel needs temporary GM store and a second pass,"
            " switch to normalize-two-pass instead"
        )
        helper_block = '''@vf()
def accumulate_row_sum_vf(src: Tensor, row_sum: Tensor, dst: Tensor, n_rows: Var):
    # TODO:
    # - accumulate row sums from src
    # - optionally write an intermediate normalized-prep or temporary copy into dst
    pass


@vf()
def normalize_rows_with_sum_vf(src: Tensor, row_sum: Tensor, dst: Tensor, n_rows: Var):
    # TODO:
    # - divide each row by its corresponding row sum
    # - keep exact zero / epsilon / tail policy explicit
    pass'''
        extra_buffers = (
            "    xbuf = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)\n"
            "    outbuf = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)\n"
            "    row_sum_ub = Tensor(DT.float, [TILE_M // 2, 8], Position.UB)"
        )
        postprocess_block = '''                half_valid = CeilDiv(valid_m, 2)
                sb_idx = Var(GetSubBlockIdx())
                row_begin = sb_idx * half_valid
                if row_begin < valid_m:
                    row_end = Min(row_begin + half_valid, valid_m)
                    row_count = row_end - row_begin

                    cvmutex.lock()
                    xbuf[tile_cnt] <<= l0c[tile_cnt]
                    cvmutex.ready()

                    cvmutex.wait()

                    # TODO:
                    # - zero row_sum scratch explicitly before accumulation when needed
                    accumulate_row_sum_vf(xbuf[tile_cnt][0:row_count, :], row_sum_ub[0:row_count, :], outbuf[tile_cnt][0:row_count, :], row_count)
                    normalize_rows_with_sum_vf(outbuf[tile_cnt][0:row_count, :], row_sum_ub[0:row_count, :], outbuf[tile_cnt][0:row_count, :], row_count)

                    z[m0 + row_begin:m0 + row_end, n0:n0 + valid_n] <<= outbuf[tile_cnt][0:row_count, 0:valid_n]
                    cvmutex.free()'''
    elif resolved_profile == "dual-output":
        profile_notes = (
            "# - this profile keeps the cube result plus a separate vec-side postprocess result\n"
            "# - do not pretend the two outputs share identical ownership or validation paths"
        )
        helper_block = _simple_postprocess_vf_stub(
            "implement the exact vec-side postprocess for the secondary output"
        )
        extra_buffers = (
            "    mid_ub = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)\n"
            "    out_vec_ub = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)"
        )
        postprocess_block = '''                out_cube[m0:m0 + valid_m, n0:n0 + valid_n] <<= l0c[tile_cnt][:valid_m, :valid_n]

                cvmutex.lock()
                mid_ub[tile_cnt] <<= l0c[tile_cnt]
                cvmutex.ready()
                cvmutex.wait()

                # TODO:
                # - compute correct vec loop count from rows/cols and dtype width
                post_loops = Var(TILE_M // 2 * TILE_N // 64)
                postprocess_vf(mid_ub[tile_cnt], out_vec_ub[tile_cnt], post_loops)
                cvmutex.free()

                half_valid = CeilDiv(valid_m, 2)
                sb_idx = Var(GetSubBlockIdx())
                row_begin = sb_idx * half_valid
                if row_begin < valid_m:
                    row_end = Min(row_begin + half_valid, valid_m)
                    row_count = row_end - row_begin
                    out_vec[m0 + row_begin:m0 + row_end, n0:n0 + valid_n] <<= out_vec_ub[tile_cnt][0:row_count, 0:valid_n]'''
    else:
        profile_notes = (
            "# - this profile is explicitly two-pass:"
            " pass 1 accumulates row sums and stores temporary tiles,"
            " pass 2 normalizes and writes final output"
        )
        helper_block = '''@vf()
def accumulate_row_sum_vf(src: Tensor, row_sum: Tensor, dst: Tensor, n_rows: Var):
    # TODO:
    # - accumulate row sums from src and write a temporary tile into dst
    pass


@vf()
def normalize_rows_with_sum_vf(src: Tensor, row_sum: Tensor, dst: Tensor, n_rows: Var):
    # TODO:
    # - divide each row by its corresponding row sum
    # - keep exact zero / epsilon / tail policy explicit
    pass'''
        extra_buffers = (
            "    xbuf = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)\n"
            "    outbuf = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)\n"
            "    row_sum_ub = Tensor(DT.float, [TILE_M // 2, 8], Position.UB)\n"
            "    pass1_cnt = Var(0)\n    pass2_cnt = Var(0)"
        )
        postprocess_block = '''                half_valid = CeilDiv(valid_m, 2)
                sb_idx = Var(GetSubBlockIdx())
                row_begin = sb_idx * half_valid
                if row_begin < valid_m:
                    row_end = Min(row_begin + half_valid, valid_m)
                    row_count = row_end - row_begin

                    # Pass 1: store temporary tile and update row sums.
                    cvmutex.lock()
                    xbuf[pass1_cnt] <<= l0c[tile_cnt]
                    cvmutex.ready()

                    cvmutex.wait()
                    accumulate_row_sum_vf(xbuf[pass1_cnt][0:row_count, :], row_sum_ub[0:row_count, :], outbuf[pass1_cnt][0:row_count, :], row_count)
                    z[m0 + row_begin:m0 + row_end, n0:n0 + valid_n] <<= outbuf[pass1_cnt][0:row_count, 0:valid_n]
                    cvmutex.free()
                    pass1_cnt += 1'''
        # append second pass after nt loop inside mt loop
    footer_after_nt = ""
    if resolved_profile == "normalize-two-pass":
        footer_after_nt = '''

            # Pass 2: read temporary tiles back and normalize them with the accumulated row sums.
            half_valid = CeilDiv(valid_m, 2)
            sb_idx = Var(GetSubBlockIdx())
            row_begin = sb_idx * half_valid
            if row_begin < valid_m:
                row_end = Min(row_begin + half_valid, valid_m)
                row_count = row_end - row_begin
                for nt in range(tile_n_begin, tile_n_end):
                    prev_n0 = Var(nt * TILE_N)
                    prev_valid_n = Min(TILE_N, N - prev_n0)
                    xbuf[pass2_cnt][0:row_count, 0:prev_valid_n] <<= z[m0 + row_begin:m0 + row_end, prev_n0:prev_n0 + prev_valid_n]
                    normalize_rows_with_sum_vf(xbuf[pass2_cnt][0:row_count, :], row_sum_ub[0:row_count, :], outbuf[pass2_cnt][0:row_count, :], row_count)
                    z[m0 + row_begin:m0 + row_end, prev_n0:prev_n0 + prev_valid_n] <<= outbuf[pass2_cnt][0:row_count, 0:prev_valid_n]
                    pass2_cnt += 1'''

    kernel_signature = f"def {name}_kernel(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):"
    return_expr = "z"
    if resolved_profile == "dual-output":
        kernel_signature = (
            f"def {name}_kernel("
            f"x: GMTensor, y: GMTensor, out_cube: GMTensor, out_vec: GMTensor, M: Var, N: Var, K: Var):"
        )
        return_expr = "out_cube, out_vec"
        if with_main:
            main_block = f'''\n\n
if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    M = 256
    N = 256
    K = 128

    x = torch.randn((M, K), dtype=torch.float16)
    y = torch.randn((N, K), dtype=torch.float16)
    out_cube = torch.zeros((M, N), dtype=torch.float32)
    out_vec = torch.zeros((M, N), dtype=torch.float32)

    out_cube_kernel, out_vec_kernel = OpExec({name}_kernel, simulator=True)(x, y, out_cube, out_vec, M, N, K)

    print(out_cube_kernel.shape)
    print(out_vec_kernel.shape)
'''

    return f'''from easyasc.a5 import *


{_formula_comment(formula)}
#
# topology:
#   cube->vec
#
{profile_comment}
#
{grid_comment}
#
{_k_loop_mode_comment(resolved_k_loop_mode)}
#
{_layout_comment(layout)}
#
# notes:
# - this skeleton is for cube main compute followed by vec-side postprocess
# - choose TILE_M / TILE_N before filling the body
{k_loop_note}
# - use CvMutex for the cube -> vec ownership edge; do not expect auto_sync() to replace it
# - keep accumulation in float unless there is a strong reason not to
{profile_notes}


TILE_M = 128
TILE_N = 128
{tile_k_decl}


{helper_block}


@kernel()
{kernel_signature}
    cvmutex = CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1x = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1y = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)
{extra_buffers}

    l1_cnt = Var(0)
    tile_cnt = Var(0)

    tile_m = CeilDiv(M, TILE_M)
    tile_n = CeilDiv(N, TILE_N)

{grid_assign}

    tile_m_per_core = CeilDiv(tile_m, m_split)
    tile_n_per_core = CeilDiv(tile_n, n_split)

{cube_axes}

    tile_m_begin = Var(tile_m_per_core * cube_row)
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    tile_n_begin = Var(tile_n_per_core * cube_col)
    tile_n_end = Min(tile_n_begin + tile_n_per_core, tile_n)

    with auto_sync():
        for mt in range(tile_m_begin, tile_m_end):
            m0 = Var(mt * TILE_M)
            valid_m = Min(TILE_M, M - m0)

            for nt in range(tile_n_begin, tile_n_end):
                n0 = Var(nt * TILE_N)
                valid_n = Min(TILE_N, N - n0)

{k_loop_block}

{postprocess_block}

                l1_cnt += 1
                tile_cnt += 1{footer_after_nt}

    return {return_expr}{main_block}'''


def render_vec_cube(
    name: str, formula: str, layout: str, grid_mode: str,
    m_split: Optional[int], n_split: Optional[int],
    profile: str, k_loop_mode: str, with_main: bool,
) -> str:
    grid_assign, cube_axes, grid_comment = _grid_config(grid_mode, m_split, n_split)
    resolved_k_loop_mode = _resolve_k_loop_mode(k_loop_mode)
    main_block = _main_stub(name, "vec->cube", layout) if with_main else ""
    resolved_profile = profile or "nd-publish"
    if resolved_profile not in ["nd-publish", "nz-publish", "half-row-pre"]:
        raise ValueError("--topology vec->cube currently supports only --profile nd-publish|nz-publish|half-row-pre")
    profile_comment = f"# profile:\n#   {resolved_profile}"

    if resolved_profile == "nz-publish":
        preprocess_helper_body = (
            "@vf()\n"
            "def preprocess_pack_nz_vf(src_nd: Tensor, dst_nz: Tensor, n_rows: Var):\n"
            "    # TODO:\n"
            "    # - implement the exact preprocess and pack-to-NZ path\n"
            "    # - examples: abs+sqrt+cast+pack, scaling+cast+pack\n"
            "    # - dst_nz is not a plain ND tensor placeholder; treat it as packed publish source\n"
            "    pass"
        )
        ub_decl = (
            "    x_nd_ub = DBuff(DT.half, [TILE_M, TILE_K], Position.UB)\n"
            "    x_nz_ub = DBuff(DT.half, [TILE_M, TILE_K], Position.UB)"
        )
        pre_counter = "    vec_stage_cnt = Var(0)\n    x_publish_cnt = Var(0)"
        half_rows_decl = ""
        notes_extra = (
            "# - preprocess output is organized for NZ-style publish to L1\n"
            "# - keep `.nz()` as the profile-defining publish step unless the profile itself changes"
        )
        if resolved_k_loop_mode == "always":
            preprocess_block = '''            vcmutex.lock()
            for k0 in range(0, K, TILE_K):
                valid_k = Min(TILE_K, K - k0)
                x_nd_ub[vec_stage_cnt][0:valid_m, 0:valid_k] <<= x[m0:m0 + valid_m, k0:k0 + valid_k]
                preprocess_pack_nz_vf(x_nd_ub[vec_stage_cnt], x_nz_ub[vec_stage_cnt], valid_m)
                l1x[x_publish_cnt][0:valid_m, k0:k0 + valid_k] <<= x_nz_ub[vec_stage_cnt][0:valid_m, 0:valid_k].nz()
            vec_stage_cnt += 1
            vcmutex.ready()

            vcmutex.wait()
            for nt in range(tile_n_begin, tile_n_end):
                n0 = Var(nt * TILE_N)
                valid_n = Min(TILE_N, N - n0)
                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)
                    # TODO:
                    # - confirm exact y layout
                    l1y[tile_cnt][0:valid_n, 0:valid_k] <<= y[n0:n0 + valid_n, k0:k0 + valid_k]
                    matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[x_publish_cnt], l1y[tile_cnt], is_init=(k0 == 0))
                z[m0:m0 + valid_m, n0:n0 + valid_n] <<= l0c[tile_cnt][0:valid_m, 0:valid_n]
                tile_cnt += 1
            vcmutex.free()
            x_publish_cnt += 1'''
            k_loop_note = (
                "# - vec preprocess and cube consume both use a tiled K loop in this scaffold\n"
                "# - vec stays outside nt but moves inside k0, matching tiled-K ownership"
            )
        else:
            preprocess_block = '''            vcmutex.lock()
            x_nd_ub[vec_stage_cnt][0:valid_m, 0:K] <<= x[m0:m0 + valid_m, 0:K]
            preprocess_pack_nz_vf(x_nd_ub[vec_stage_cnt], x_nz_ub[vec_stage_cnt], valid_m)
            l1x[x_publish_cnt][0:valid_m, 0:K] <<= x_nz_ub[vec_stage_cnt][0:valid_m, 0:K].nz()
            vec_stage_cnt += 1
            vcmutex.ready()

            vcmutex.wait()
            for nt in range(tile_n_begin, tile_n_end):
                n0 = Var(nt * TILE_N)
                valid_n = Min(TILE_N, N - n0)
                # TODO:
                # - confirm exact y layout for the one-shot cube consume path
                l1y[tile_cnt][0:valid_n, 0:K] <<= y[n0:n0 + valid_n, 0:K]
                matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[x_publish_cnt], l1y[tile_cnt])
                z[m0:m0 + valid_m, n0:n0 + valid_n] <<= l0c[tile_cnt][0:valid_m, 0:valid_n]
                tile_cnt += 1
            vcmutex.free()
            x_publish_cnt += 1'''
            k_loop_note = (
                "# - this scaffold intentionally omits a K loop;"
                " vec preprocess stays outside nt only because K is one-shot in this profile"
            )
    elif resolved_profile == "half-row-pre":
        preprocess_helper_body = _simple_preprocess_vf_stub(
            "this profile assumes split-M style subblock ownership before L1 publish"
        )
        ub_decl = (
            "    x_ub = DBuff(DT.half, [TILE_M // 2, TILE_K], Position.UB)\n"
            "    x_proc_ub = DBuff(DT.half, [TILE_M // 2, TILE_K], Position.UB)"
        )
        pre_counter = "    pre_cnt = Var(0)\n    l1x_cnt = Var(0)\n    l1y_cnt = Var(0)"
        half_rows_decl = "    half_rows = Var(TILE_M // 2)"
        notes_extra = (
            "# - this profile keeps the repository's common split-M half-row"
            " preprocess/publish shape explicit"
        )
        if resolved_k_loop_mode == "always":
            preprocess_block = '''            vcmutex.lock()
            row_start = Var(GetSubBlockIdx() * half_rows)
            if row_start < valid_m:
                row_end = Min(row_start + half_rows, valid_m)
                rows_this = Var(row_end - row_start)
                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)
                    vf_loops = Var(rows_this * valid_k // 64)
                    x_ub[pre_cnt][0:rows_this, 0:valid_k] <<= x[m0 + row_start:m0 + row_end, k0:k0 + valid_k]
                    preprocess_vf(x_ub[pre_cnt], x_proc_ub[pre_cnt], vf_loops)
                    l1x[l1x_cnt][row_start:row_end, k0:k0 + valid_k] <<= x_proc_ub[pre_cnt][0:rows_this, 0:valid_k]
            pre_cnt += 1
            vcmutex.ready()

            vcmutex.wait()
            for nt in range(tile_n_begin, tile_n_end):
                n0 = Var(nt * TILE_N)
                valid_n = Min(TILE_N, N - n0)
                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)
                    # TODO:
                    # - confirm exact y layout
                    l1y[tile_cnt][0:valid_n, 0:valid_k] <<= y[n0:n0 + valid_n, k0:k0 + valid_k]
                    matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[l1x_cnt], l1y[tile_cnt], is_init=(k0 == 0))
                z[m0:m0 + valid_m, n0:n0 + valid_n] <<= l0c[tile_cnt][:valid_m, :valid_n]
                tile_cnt += 1
            vcmutex.free()
            l1x_cnt += 1'''
            k_loop_note = (
                "# - vec preprocess and cube consume both use a tiled K loop in this scaffold\n"
                "# - half-row-pre keeps split-M subblock ownership explicit"
                " while K tiles stream through the same stage"
            )
        else:
            preprocess_block = '''            vcmutex.lock()
            row_start = Var(GetSubBlockIdx() * half_rows)
            if row_start < valid_m:
                row_end = Min(row_start + half_rows, valid_m)
                rows_this = Var(row_end - row_start)
                vf_loops = Var(rows_this * K // 64)
                x_ub[pre_cnt][0:rows_this, 0:K] <<= x[m0 + row_start:m0 + row_end, 0:K]
                preprocess_vf(x_ub[pre_cnt], x_proc_ub[pre_cnt], vf_loops)
                l1x[l1x_cnt][row_start:row_end, 0:K] <<= x_proc_ub[pre_cnt][0:rows_this, 0:K]
            pre_cnt += 1
            vcmutex.ready()

            vcmutex.wait()
            for nt in range(tile_n_begin, tile_n_end):
                n0 = Var(nt * TILE_N)
                valid_n = Min(TILE_N, N - n0)
                # TODO:
                # - confirm exact y layout for the one-shot cube consume path
                l1y[tile_cnt][0:valid_n, 0:K] <<= y[n0:n0 + valid_n, 0:K]
                matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[l1x_cnt], l1y[tile_cnt])
                z[m0:m0 + valid_m, n0:n0 + valid_n] <<= l0c[tile_cnt][:valid_m, :valid_n]
                tile_cnt += 1
            vcmutex.free()
            l1x_cnt += 1'''
            k_loop_note = (
                "# - this scaffold intentionally omits a K loop;"
                " half-row-pre still keeps split-M subblock ownership explicit"
                " in the one-shot publish path"
            )
    else:
        preprocess_helper_body = _simple_preprocess_vf_stub()
        ub_decl = (
            "    x_ub = DBuff(DT.half, [TILE_M, TILE_K], Position.UB)\n"
            "    x_proc_ub = DBuff(DT.half, [TILE_M, TILE_K], Position.UB)"
        )
        pre_counter = "    pre_cnt = Var(0)\n    l1x_cnt = Var(0)\n    l1y_cnt = Var(0)"
        half_rows_decl = ""
        notes_extra = (
            "# - this is the generic ND publish path;"
            " keep ownership explicit instead of assuming split-M subblock slicing"
        )
        if resolved_k_loop_mode == "always":
            preprocess_block = '''            vcmutex.lock()
            for k0 in range(0, K, TILE_K):
                valid_k = Min(TILE_K, K - k0)
                vf_loops = Var(valid_m * valid_k // 64)
                x_ub[pre_cnt][0:valid_m, 0:valid_k] <<= x[m0:m0 + valid_m, k0:k0 + valid_k]
                preprocess_vf(x_ub[pre_cnt], x_proc_ub[pre_cnt], vf_loops)
                l1x[l1x_cnt][0:valid_m, k0:k0 + valid_k] <<= x_proc_ub[pre_cnt][0:valid_m, 0:valid_k]
            pre_cnt += 1
            vcmutex.ready()

            vcmutex.wait()
            for nt in range(tile_n_begin, tile_n_end):
                n0 = Var(nt * TILE_N)
                valid_n = Min(TILE_N, N - n0)
                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)
                    # TODO:
                    # - confirm exact y layout
                    l1y[tile_cnt][0:valid_n, 0:valid_k] <<= y[n0:n0 + valid_n, k0:k0 + valid_k]
                    matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[l1x_cnt], l1y[tile_cnt], is_init=(k0 == 0))
                z[m0:m0 + valid_m, n0:n0 + valid_n] <<= l0c[tile_cnt][:valid_m, :valid_n]
                tile_cnt += 1
            vcmutex.free()
            l1x_cnt += 1'''
            k_loop_note = (
                "# - vec preprocess and cube consume both use a tiled K loop in this scaffold\n"
                "# - generic ND publish keeps the preprocess tile whole"
                " unless the profile explicitly changes ownership"
            )
        else:
            preprocess_block = '''            vcmutex.lock()
            vf_loops = Var(valid_m * K // 64)
            x_ub[pre_cnt][0:valid_m, 0:K] <<= x[m0:m0 + valid_m, 0:K]
            preprocess_vf(x_ub[pre_cnt], x_proc_ub[pre_cnt], vf_loops)
            l1x[l1x_cnt][0:valid_m, 0:K] <<= x_proc_ub[pre_cnt][0:valid_m, 0:K]
            pre_cnt += 1
            vcmutex.ready()

            vcmutex.wait()
            for nt in range(tile_n_begin, tile_n_end):
                n0 = Var(nt * TILE_N)
                valid_n = Min(TILE_N, N - n0)
                # TODO:
                # - confirm exact y layout for the one-shot cube consume path
                l1y[tile_cnt][0:valid_n, 0:K] <<= y[n0:n0 + valid_n, 0:K]
                matmul(l0c[tile_cnt][:valid_m, :valid_n], l1x[l1x_cnt], l1y[tile_cnt])
                z[m0:m0 + valid_m, n0:n0 + valid_n] <<= l0c[tile_cnt][:valid_m, :valid_n]
                tile_cnt += 1
            vcmutex.free()
            l1x_cnt += 1'''
            k_loop_note = (
                "# - this scaffold intentionally omits a K loop;"
                " generic ND publish keeps the preprocess tile whole"
                " unless the profile explicitly changes ownership"
            )

    tile_k_decl = (
        "TILE_K = 128"
        if resolved_k_loop_mode == "always"
        else "TILE_K = 128  # TODO: set this to the intended one-shot K width or re-enable the K loop"
    )

    return f'''from easyasc.a5 import *


{_formula_comment(formula)}
#
# topology:
#   vec->cube
#
{profile_comment}
#
{grid_comment}
#
{_k_loop_mode_comment(resolved_k_loop_mode)}
#
{_layout_comment(layout)}
#
# notes:
# - this skeleton is for vec-side preprocess followed by cube consume
# - choose TILE_M / TILE_N before filling the body
{k_loop_note}
# - use VcMutex for the vec -> cube ownership edge; do not expect auto_sync() to replace it
{notes_extra}


TILE_M = 64
TILE_N = 128
{tile_k_decl}


{preprocess_helper_body}


@kernel()
def {name}_kernel(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
    vcmutex = VcMutex(0, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)

    l1x = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1y = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)
{ub_decl}

{pre_counter}
    tile_cnt = Var(0)
{half_rows_decl}

    tile_m = CeilDiv(M, TILE_M)
    tile_n = CeilDiv(N, TILE_N)

{grid_assign}

    tile_m_per_core = CeilDiv(tile_m, m_split)
    tile_n_per_core = CeilDiv(tile_n, n_split)

{cube_axes}

    tile_m_begin = Var(tile_m_per_core * cube_row)
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    tile_n_begin = Var(tile_n_per_core * cube_col)
    tile_n_end = Min(tile_n_begin + tile_n_per_core, tile_n)

    with auto_sync():
        for mt in range(tile_m_begin, tile_m_end):
            m0 = Var(mt * TILE_M)
            valid_m = Min(TILE_M, M - m0)

{preprocess_block}

    return z{main_block}'''


def render_vec_cube_vec(
    name: str, formula: str, layout: str, grid_mode: str,
    m_split: Optional[int], n_split: Optional[int],
    profile: str, k_loop_mode: str, with_main: bool,
) -> str:
    if grid_mode != "tile-m":
        raise ValueError("v1 currently supports --topology vec->cube->vec only with --grid-mode tile-m")

    grid_assign, cube_axes, grid_comment = _grid_config(grid_mode, m_split, n_split)
    resolved_k_loop_mode = _resolve_k_loop_mode(k_loop_mode)
    main_block = _main_stub(name, "vec->cube->vec", layout) if with_main else ""
    resolved_profile = profile or "overlap-basic"
    if resolved_profile not in ["overlap-basic", "half-row-post", "delayed-post"]:
        raise ValueError(
            "--topology vec->cube->vec currently supports only"
            " --profile overlap-basic|half-row-post|delayed-post"
        )
    profile_comment = f"# profile:\n#   {resolved_profile}"

    if resolved_profile == "delayed-post":
        delay_decl = "POST_DELAY_TILES = 2"
        cvmutex_decl = (
            "    cvmutex = CvMutex(0, depth=POST_DELAY_TILES + 1,"
            " src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)"
        )
        loop_bound = "N + POST_DELAY_TILES * TILE_N"
        stage2_gate = "n0 >= POST_DELAY_TILES * TILE_N"
        prev_n0_expr = "Var(n0 - POST_DELAY_TILES * TILE_N)"
        stage2_depth_note = (
            "# - if POST_DELAY_TILES changes,"
            " keep the CvMutex depth and warmup/drain reasoning aligned with it"
        )
        pipeline_note = (
            "# - this template keeps vec2 intentionally behind stage1 by POST_DELAY_TILES tiles\n"
            "# - when the delayed drain falls behind by more than one tile,"
            " increase CvMutex depth with the delay so cube can publish safely"
        )
        delay_note = (
            "# - delayed-post uses a deeper cube -> vec queue"
            " plus a longer warmup/drain period than the one-tile overlap profile"
        )
    else:
        delay_decl = ""
        cvmutex_decl = "    cvmutex = CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)"
        loop_bound = "N + TILE_N"
        stage2_gate = "n0 > 0"
        prev_n0_expr = "Var(n0 - TILE_N)"
        stage2_depth_note = (
            "# - if this post stage ever lags by more than one tile,"
            " increase CvMutex depth with the extra delay"
        )
        pipeline_note = (
            "# - this template overlaps stage1 (vec1 + cube) with stage2 (vec2)"
            " by shifting the postprocess one tile later"
        )
        if resolved_profile == "half-row-post":
            delay_note = "# - half-row-post stays on the repository's default split-M handoff model for vec2 drain"
        else:
            delay_note = (
                "# - overlap-basic keeps the default one-tile delayed drain"
                " on the repository's split-M handoff model"
            )

    if resolved_k_loop_mode == "always":
        stage1_block = '''                if n0 < N:
                    valid_n = Min(TILE_N, N - n0)

                    vcmutex.lock()
                    row_start = Var(GetSubBlockIdx() * half_rows)
                    if row_start < valid_m:
                        row_end = Min(row_start + half_rows, valid_m)
                        rows_this = Var(row_end - row_start)
                        for k0 in range(0, K, TILE_K):
                            valid_k = Min(TILE_K, K - k0)
                            vf_loops = Var(rows_this * valid_k // 64)

                            # TODO:
                            # - confirm exact preprocess input layout
                            # - if this is MKNK, keep x[m0 + row_start:..., k0:...]
                            # - if this is KMKN, switch slices before the vec stage
                            x_ub[stage1_cnt][0:rows_this, 0:valid_k] <<= x[m0 + row_start:m0 + row_end, k0:k0 + valid_k]
                            preprocess_vf(x_ub[stage1_cnt], x_proc_ub[stage1_cnt], vf_loops)
                            l1x[stage1_cnt][row_start:row_end, k0:k0 + valid_k] <<= x_proc_ub[stage1_cnt][0:rows_this, 0:valid_k]
                    vcmutex.ready()

                    vcmutex.wait()
                    for k0 in range(0, K, TILE_K):
                        valid_k = Min(TILE_K, K - k0)
                        # TODO:
                        # - confirm exact y layout for the tiled cube consume path
                        l1y[stage1_cnt][0:valid_n, 0:valid_k] <<= y[n0:n0 + valid_n, k0:k0 + valid_k]
                        matmul(l0c[stage1_cnt][:valid_m, :valid_n], l1x[stage1_cnt], l1y[stage1_cnt], is_init=(k0 == 0))

                    cvmutex.lock()
                    mid_ub[stage1_cnt] <<= l0c[stage1_cnt]
                    cvmutex.ready()

                    vcmutex.free()
                    stage1_cnt += 1'''
        tile_k_decl = "TILE_K = 128"
        if resolved_profile == "delayed-post":
            k_loop_note = (
                "# - stage1 (vec1 + cube) currently uses a tiled K loop;"
                " vec2 drains after a longer delayed queue"
            )
        else:
            k_loop_note = "# - stage1 (vec1 + cube) currently uses a tiled K loop; stage2 drains one tile later"
    else:
        stage1_block = '''                if n0 < N:
                    valid_n = Min(TILE_N, N - n0)

                    vcmutex.lock()
                    row_start = Var(GetSubBlockIdx() * half_rows)
                    if row_start < valid_m:
                        row_end = Min(row_start + half_rows, valid_m)
                        rows_this = Var(row_end - row_start)
                        vf_loops = Var(rows_this * K // 64)

                        # TODO:
                        # - confirm exact preprocess input layout
                        # - if this is MKNK, keep x[m0 + row_start:..., 0:K]
                        # - if this is KMKN, switch slices before the vec stage
                        x_ub[stage1_cnt][0:rows_this, 0:K] <<= x[m0 + row_start:m0 + row_end, 0:K]
                        preprocess_vf(x_ub[stage1_cnt], x_proc_ub[stage1_cnt], vf_loops)
                        l1x[stage1_cnt][row_start:row_end, 0:K] <<= x_proc_ub[stage1_cnt][0:rows_this, 0:K]
                    vcmutex.ready()

                    vcmutex.wait()
                    # TODO:
                    # - confirm exact y layout for the one-shot cube consume path
                    l1y[stage1_cnt][0:valid_n, 0:K] <<= y[n0:n0 + valid_n, 0:K]
                    matmul(l0c[stage1_cnt][:valid_m, :valid_n], l1x[stage1_cnt], l1y[stage1_cnt])

                    cvmutex.lock()
                    mid_ub[stage1_cnt] <<= l0c[stage1_cnt]
                    cvmutex.ready()

                    vcmutex.free()
                    stage1_cnt += 1'''
        tile_k_decl = "TILE_K = 128  # TODO: set this to the intended one-shot K width or re-enable the K loop"
        if resolved_profile == "delayed-post":
            k_loop_note = (
                "# - this scaffold intentionally omits a K loop;"
                " vec2 still drains through a deeper delayed queue instead of a one-tile overlap"
            )
        else:
            k_loop_note = (
                "# - this scaffold intentionally omits a K loop;"
                " stage1 vec1/cube and stage2 vec2 stay aligned by tile order instead"
            )

    stage2_block = f'''                if {stage2_gate}:
                    prev_n0 = {prev_n0_expr}
                    prev_valid_n = Min(TILE_N, N - prev_n0)

                    cvmutex.wait()

                    # TODO:
                    # - compute correct vec loop count from rows/cols and dtype width
                    {stage2_depth_note}
                    post_loops = Var(TILE_M // 2 * TILE_N // 64)
                    postprocess_vf(mid_ub[stage2_cnt], out_ub[stage2_cnt], post_loops)

                    row_begin = Var(GetSubBlockIdx() * half_rows)
                    if row_begin < valid_m:
                        row_end = Min(row_begin + half_rows, valid_m)
                        row_count = Var(row_end - row_begin)
                        z[m0 + row_begin:m0 + row_end, prev_n0:prev_n0 + prev_valid_n] <<= out_ub[stage2_cnt][0:row_count, 0:prev_valid_n]

                    cvmutex.free()
                    stage2_cnt += 1'''

    return f'''from easyasc.a5 import *


{_formula_comment(formula)}
#
# topology:
#   vec->cube->vec
#
{profile_comment}
#
{grid_comment}
#
{_k_loop_mode_comment(resolved_k_loop_mode)}
#
{_layout_comment(layout)}
#
# notes:
# - this skeleton is for vec-side preprocess, cube main compute, then vec-side postprocess
# - use VcMutex for vec -> cube publish and CvMutex for cube -> vec handoff
{pipeline_note}
# - keep stage1_cnt and stage2_cnt separate unless you can prove the lifetimes are identical
{k_loop_note}
{delay_note}


TILE_M = 64
TILE_N = 128
{tile_k_decl}
{delay_decl}


@vf()
def preprocess_vf(src: Tensor, dst: Tensor, n_loops: Var):
    # TODO:
    # - implement the exact vec-side preprocess
    # - examples: abs+sqrt, scale, cast, rowwise transform
    pass


@vf()
def postprocess_vf(src: Tensor, dst: Tensor, n_loops: Var):
    # TODO:
    # - implement the exact vec-side postprocess
    # - examples: add bias, abs + 1, rowwise normalize, cast
    pass


@kernel()
def {name}_kernel(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
    vcmutex = VcMutex(0, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
{cvmutex_decl}

    l1x = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1y = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    x_ub = DBuff(DT.half, [TILE_M // 2, TILE_K], Position.UB)
    x_proc_ub = DBuff(DT.half, [TILE_M // 2, TILE_K], Position.UB)

    mid_ub = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)
    out_ub = DBuff(DT.float, [TILE_M // 2, TILE_N], Position.UB)

    stage1_cnt = Var(0)
    stage2_cnt = Var(0)
    half_rows = Var(TILE_M // 2)

    tile_m = CeilDiv(M, TILE_M)
    tile_m_per_core = CeilDiv(tile_m, GetCubeNum())
    tile_m_begin = Var(tile_m_per_core * GetCubeIdx())
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    with auto_sync():
        for mt in range(tile_m_begin, tile_m_end):
            m0 = Var(mt * TILE_M)
            valid_m = Min(TILE_M, M - m0)

            for n0 in range(0, {loop_bound}, TILE_N):
{stage1_block}

{stage2_block}

    return z{main_block}'''


def render_cube_vec_cube_vec(
    name: str, formula: str, layout: str, grid_mode: str,
    m_split: Optional[int], n_split: Optional[int],
    profile: str, k_loop_mode: str, with_main: bool,
) -> str:
    if grid_mode != "tile-m":
        raise ValueError("v1 currently supports --topology cube->vec->cube->vec only with --grid-mode tile-m")

    resolved_profile = profile or "lookahead-basic"
    if resolved_profile != "lookahead-basic":
        raise ValueError("--topology cube->vec->cube->vec currently supports only --profile lookahead-basic")

    resolved_k_loop_mode = _resolve_k_loop_mode(k_loop_mode)
    if resolved_k_loop_mode != "never":
        raise ValueError(
            "v1 currently supports --topology cube->vec->cube->vec only with one-shot cube stages;"
            " use --k-loop-mode never"
        )

    main_block = _cube_vec_cube_vec_main_stub(name) if with_main else ""
    profile_comment = f"# profile:\n#   {resolved_profile}"
    stage_layout_comment = (
        "# layout:\n#   lhs1: [M, K1]\n#   rhs1_stream: [T, S1, K1]\n"
        "#   rhs2_stream: [T, N2, S1]\n#   out: [M, N2]"
    )
    notes_block = _lookahead_common_notes(
        "# - this skeleton is for cube -> vec -> cube -> vec lookahead streaming pipelines",
        "# - use CvMutex for cube->vec ownership and VcMutex for vec->cube publish",
        [
            "# - finalize vec logic stays outside the streaming loop",
            "# - here `T` means streamed block count while `S1` is the stage1 output / stage2 reduction width",
            "# - `rhs2_stream` stores each stage2 rhs block as `[N2, S1]`"
            " so stage2 can consume it directly without inventing another transpose-side story in the scaffold",
            "# - this v1 scaffold assumes K1 <= TILE_K1, S1 <= TILE_S1, and N2 <= TILE_N2 for one-shot cube stages",
            "# - if either cube stage needs tiled-K or wider stage2 output tiling,"
            " add a dedicated profile instead of stretching this one silently",
            "# - stage2_in_ub is a plain half placeholder in v1;"
            " if the real stage2 consume path needs packed/NZ or another dtype, change the publish path explicitly",
        ],
    )

    return f'''from easyasc.a5 import *


{_formula_comment(formula)}
#
# topology:
#   cube->vec->cube->vec
#
{profile_comment}
#
# grid mode:
#   tile-m
#
# k-loop mode:
#   never
#
{stage_layout_comment}
#
# notes:
{notes_block}


# Smoke defaults only: these tile values are plausible for bring-up and preview, not recommended production tiling.
TILE_M = 16
TILE_S1 = 256
TILE_K1 = 128
TILE_N2 = 512


@vf()
def init_running_state_vf(state_ub: Tensor, accum_ub: Tensor, n_rows: Var):
    # TODO:
    # - initialize persistent running state before the streaming loop
    # - zero / seed accumulators explicitly
    pass


@vf()
def update_state_and_emit_stage2_input_vf(cube1_ub: Tensor, state_ub: Tensor, stage2_in_ub: Tensor, n_rows: Var):
    # TODO:
    # - update running state from stage1 cube output
    # - emit the stage2 input tile
    pass


@vf()
def accumulate_stage2_output_vf(cube2_ub: Tensor, state_ub: Tensor, accum_ub: Tensor, n_rows: Var):
    # TODO:
    # - consume delayed stage2 cube output
    # - update accumulated output using the running state
    pass


@vf()
def finalize_output_vf(accum_ub: Tensor, state_ub: Tensor, out_ub: Tensor, n_rows: Var):
    # TODO:
    # - finalize the accumulated output using the running state
    pass


@kernel()
def {name}_kernel(lhs1: GMTensor, rhs1_stream: GMTensor, rhs2_stream: GMTensor, out: GMTensor, M: Var, T: Var, K1: Var, S1: Var, N2: Var):
    cv_stage1 = CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    vc_stage2_in = VcMutex(1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    cv_stage2 = CvMutex(2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1_lhs1 = DBuff(DT.half, [TILE_M, TILE_K1], Position.L1)
    l1_rhs1 = TBuff(DT.half, [TILE_S1, TILE_K1], Position.L1)
    l1_stage2_in = DBuff(DT.half, [TILE_M, TILE_S1], Position.L1)
    # stage2 consumes each streamed rhs block as an `[N2, S1]` tile directly.
    l1_rhs2 = TBuff(DT.half, [TILE_N2, TILE_S1], Position.L1)

    l0c_stage1 = DBuff(DT.float, [TILE_M, TILE_S1], Position.L0C)
    l0c_stage2 = DBuff(DT.float, [TILE_M, TILE_N2], Position.L0C)

    stage1_ub = DBuff(DT.float, [TILE_M // 2, TILE_S1], Position.UB)
    # plain half placeholder: switch this publish path explicitly if stage2 needs packed/NZ or another dtype story.
    stage2_in_ub = DBuff(DT.half, [TILE_M // 2, TILE_S1], Position.UB)
    stage2_out_ub = DBuff(DT.float, [TILE_M // 2, TILE_N2], Position.UB)

    # running state is sharded by vec-side row ownership rather than shared as one global tile state.
    state_ub = Tensor(DT.float, [TILE_M // 2, 64], Position.UB)
    accum_ub = Tensor(DT.float, [TILE_M // 2, TILE_N2], Position.UB)
    out_ub = Tensor(DT.float, [TILE_M // 2, TILE_N2], Position.UB)

    stage1_cnt = Var(0)
    stage2_cnt = Var(0)
    half_rows = Var(TILE_M // 2)

    tile_m = CeilDiv(M, TILE_M)
    tile_m_per_core = CeilDiv(tile_m, GetCubeNum())
    tile_m_begin = Var(tile_m_per_core * GetCubeIdx())
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    with auto_sync():
        for mt in range(tile_m_begin, tile_m_end):
            m0 = Var(mt * TILE_M)
            valid_m = Min(TILE_M, M - m0)

            row_start = Var(GetSubBlockIdx() * half_rows)
            if row_start < valid_m:
                row_end = Min(row_start + half_rows, valid_m)
                rows_this = Var(row_end - row_start)
                init_running_state_vf(state_ub[0:rows_this, :], accum_ub[0:rows_this, 0:N2], rows_this)

            for t0 in range(0, T + 1):
                if t0 < T:
                    l1_lhs1[stage1_cnt][0:valid_m, 0:K1] <<= lhs1[m0:m0 + valid_m, 0:K1]
                    l1_rhs1[stage1_cnt][0:S1, 0:K1] <<= rhs1_stream[t0, 0:S1, 0:K1]
                    matmul(l0c_stage1[stage1_cnt][:valid_m, :S1], l1_lhs1[stage1_cnt], l1_rhs1[stage1_cnt], m=valid_m, n=S1, k=K1)

                    cv_stage1.lock()
                    cv_stage1.ready()

                    cv_stage1.wait()
                    if row_start < valid_m:
                        stage1_ub[stage1_cnt][0:rows_this, 0:S1] <<= l0c_stage1[stage1_cnt][row_start:row_end, 0:S1]
                        update_state_and_emit_stage2_input_vf(stage1_ub[stage1_cnt][0:rows_this, 0:S1], state_ub[0:rows_this, :], stage2_in_ub[stage1_cnt][0:rows_this, 0:S1], rows_this)
                    cv_stage1.free()

                    vc_stage2_in.lock()
                    if row_start < valid_m:
                        l1_stage2_in[stage1_cnt][row_start:row_end, 0:S1] <<= stage2_in_ub[stage1_cnt][0:rows_this, 0:S1]
                    vc_stage2_in.ready()

                    stage1_cnt += 1

                if t0 > 0:
                    prev_t0 = Var(t0 - 1)

                    vc_stage2_in.wait()
                    l1_rhs2[stage2_cnt][0:N2, 0:S1] <<= rhs2_stream[prev_t0, 0:N2, 0:S1]
                    matmul(l0c_stage2[stage2_cnt][:valid_m, :N2], l1_stage2_in[stage2_cnt], l1_rhs2[stage2_cnt], m=valid_m, n=N2, k=S1)
                    vc_stage2_in.free()

                    cv_stage2.lock()
                    cv_stage2.ready()

                    cv_stage2.wait()
                    if row_start < valid_m:
                        stage2_out_ub[stage2_cnt][0:rows_this, 0:N2] <<= l0c_stage2[stage2_cnt][row_start:row_end, 0:N2]
                        accumulate_stage2_output_vf(stage2_out_ub[stage2_cnt][0:rows_this, 0:N2], state_ub[0:rows_this, :], accum_ub[0:rows_this, 0:N2], rows_this)
                    cv_stage2.free()

                    stage2_cnt += 1

            if row_start < valid_m:
                finalize_output_vf(accum_ub[0:rows_this, 0:N2], state_ub[0:rows_this, :], out_ub[0:rows_this, 0:N2], rows_this)

            bar_all()
            if row_start < valid_m:
                out[m0 + row_start:m0 + row_end, 0:N2] <<= out_ub[0:rows_this, 0:N2]

    return out{main_block}'''


def render_vec_cube_vec_cube(
    name: str, formula: str, layout: str, grid_mode: str,
    m_split: Optional[int], n_split: Optional[int],
    profile: str, k_loop_mode: str, with_main: bool,
) -> str:
    if grid_mode != "tile-m":
        raise ValueError("v1 currently supports --topology vec->cube->vec->cube only with --grid-mode tile-m")

    resolved_profile = profile or "lookahead-basic"
    if resolved_profile != "lookahead-basic":
        raise ValueError("--topology vec->cube->vec->cube currently supports only --profile lookahead-basic")

    resolved_k_loop_mode = _resolve_k_loop_mode(k_loop_mode)
    if resolved_k_loop_mode != "never":
        raise ValueError(
            "v1 currently supports --topology vec->cube->vec->cube only with one-shot cube stages;"
            " use --k-loop-mode never"
        )

    main_block = _vec_cube_vec_cube_main_stub(name) if with_main else ""
    profile_comment = f"# profile:\n#   {resolved_profile}"
    stage_layout_comment = (
        "# layout:\n#   src_stream: [T, M, K1]\n#   rhs1: [S1, K1]\n"
        "#   rhs2: [N2, S1]\n#   out: [T, M, N2]"
    )
    notes_block = _lookahead_common_notes(
        "# - this skeleton is for vec -> cube -> vec -> cube lookahead streaming pipelines",
        "# - use VcMutex for vec->cube publish and CvMutex for cube->vec ownership",
        [
            "# - this mirrored topology is currently inferred from the repository's lookahead patterns"
            " rather than lifted from one canonical kernel",
            "# - here the running state is mainly for vec-side transformation between the two cube stages,"
            " not for a loop-external final epilogue",
            "# - here `T` means streamed block count while `S1` is the stage1 output / stage2 reduction width",
            "# - this v1 mirrored scaffold assumes a streamed lhs plus fixed rhs1/rhs2 tiles;"
            " fork another profile if both cube stages need streamed rhs blocks",
            "# - `rhs1` stores the first cube rhs as `[S1, K1]` and `rhs2` stores the second cube rhs as `[N2, S1]`"
            " so both cube stages can consume direct tiles"
            " without inventing another transpose-side story in the scaffold",
            "# - this v1 scaffold assumes K1 <= TILE_K1, S1 <= TILE_S1, and N2 <= TILE_N2 for one-shot cube stages",
            "# - if either cube stage needs tiled-K or wider stage2 output tiling,"
            " add a dedicated profile instead of stretching this one silently",
            "# - output is emitted per streamed block as `out[t, ...]`,"
            " not reduced across blocks into one final `[M, N2]` tensor",
            "# - the streamed vec UB staging and `stage2_in_ub` are plain half/float placeholders in v1;"
            " if a real consume path needs packed/NZ or another dtype, change the publish path explicitly",
        ],
    )

    return f'''from easyasc.a5 import *


{_formula_comment(formula)}
#
# topology:
#   vec->cube->vec->cube
#
{profile_comment}
#
# grid mode:
#   tile-m
#
# k-loop mode:
#   never
#
{stage_layout_comment}
#
# notes:
{notes_block}


# Smoke defaults only: these tile values are plausible for bring-up and preview, not recommended production tiling.
TILE_M = 16
TILE_S1 = 256
TILE_K1 = 128
TILE_N2 = 512


@vf()
def preprocess_stream_input_vf(src_ub: Tensor, dst_ub: Tensor, n_rows: Var):
    # TODO:
    # - preprocess the streamed vec input before the first cube stage
    pass


@vf()
def update_state_and_emit_stage2_input_vf(cube1_ub: Tensor, state_ub: Tensor, stage2_in_ub: Tensor, n_rows: Var):
    # TODO:
    # - update running state from the first cube output
    # - emit the second cube input tile
    pass


@vf()
def init_running_state_vf(state_ub: Tensor, n_rows: Var):
    # TODO:
    # - initialize persistent running state before the streaming loop
    pass


@kernel()
def {name}_kernel(src_stream: GMTensor, rhs1: GMTensor, rhs2: GMTensor, out: GMTensor, T: Var, M: Var, K1: Var, S1: Var, N2: Var):
    vc_stage1 = VcMutex(0, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    cv_stage1 = CvMutex(1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    vc_stage2 = VcMutex(2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)

    l1_stage1_in = DBuff(DT.half, [TILE_M, TILE_K1], Position.L1)
    l1_rhs1 = DBuff(DT.half, [TILE_S1, TILE_K1], Position.L1)
    l1_stage2_in = DBuff(DT.half, [TILE_M, TILE_S1], Position.L1)
    l1_rhs2 = DBuff(DT.half, [TILE_N2, TILE_S1], Position.L1)

    l0c_stage1 = DBuff(DT.float, [TILE_M, TILE_S1], Position.L0C)
    l0c_stage2 = DBuff(DT.float, [TILE_M, TILE_N2], Position.L0C)

    src_ub = DBuff(DT.half, [TILE_M // 2, TILE_K1], Position.UB)
    src_proc_ub = DBuff(DT.half, [TILE_M // 2, TILE_K1], Position.UB)
    stage1_ub = DBuff(DT.float, [TILE_M // 2, TILE_S1], Position.UB)
    # plain half placeholder: switch this publish path explicitly if cube2 needs packed/NZ or another dtype story.
    stage2_in_ub = DBuff(DT.half, [TILE_M // 2, TILE_S1], Position.UB)

    # running state is sharded by vec-side row ownership rather than shared as one global tile state.
    state_ub = Tensor(DT.float, [TILE_M // 2, 64], Position.UB)

    stage1_cnt = Var(0)
    stage2_cnt = Var(0)
    half_rows = Var(TILE_M // 2)

    tile_m = CeilDiv(M, TILE_M)
    tile_m_per_core = CeilDiv(tile_m, GetCubeNum())
    tile_m_begin = Var(tile_m_per_core * GetCubeIdx())
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    with auto_sync():
        for mt in range(tile_m_begin, tile_m_end):
            m0 = Var(mt * TILE_M)
            valid_m = Min(TILE_M, M - m0)

            row_start = Var(GetSubBlockIdx() * half_rows)
            if row_start < valid_m:
                row_end = Min(row_start + half_rows, valid_m)
                rows_this = Var(row_end - row_start)
                init_running_state_vf(state_ub[0:rows_this, :], rows_this)

            for t0 in range(0, T + 1):
                if t0 < T:
                    vc_stage1.lock()
                    if row_start < valid_m:
                        src_ub[stage1_cnt][0:rows_this, 0:K1] <<= src_stream[t0, m0 + row_start:m0 + row_end, 0:K1]
                        preprocess_stream_input_vf(src_ub[stage1_cnt][0:rows_this, 0:K1], src_proc_ub[stage1_cnt][0:rows_this, 0:K1], rows_this)
                        l1_stage1_in[stage1_cnt][row_start:row_end, 0:K1] <<= src_proc_ub[stage1_cnt][0:rows_this, 0:K1]
                    vc_stage1.ready()

                    vc_stage1.wait()
                    l1_rhs1[stage1_cnt][0:S1, 0:K1] <<= rhs1[0:S1, 0:K1]
                    matmul(l0c_stage1[stage1_cnt][:valid_m, :S1], l1_stage1_in[stage1_cnt], l1_rhs1[stage1_cnt], m=valid_m, n=S1, k=K1)
                    vc_stage1.free()

                    cv_stage1.lock()
                    cv_stage1.ready()

                    cv_stage1.wait()
                    if row_start < valid_m:
                        stage1_ub[stage1_cnt][0:rows_this, 0:S1] <<= l0c_stage1[stage1_cnt][row_start:row_end, 0:S1]
                        update_state_and_emit_stage2_input_vf(stage1_ub[stage1_cnt][0:rows_this, 0:S1], state_ub[0:rows_this, :], stage2_in_ub[stage1_cnt][0:rows_this, 0:S1], rows_this)
                    cv_stage1.free()

                    vc_stage2.lock()
                    if row_start < valid_m:
                        l1_stage2_in[stage1_cnt][row_start:row_end, 0:S1] <<= stage2_in_ub[stage1_cnt][0:rows_this, 0:S1]
                    vc_stage2.ready()

                    stage1_cnt += 1

                if t0 > 0:
                    prev_t0 = Var(t0 - 1)

                    vc_stage2.wait()
                    l1_rhs2[stage2_cnt][0:N2, 0:S1] <<= rhs2[0:N2, 0:S1]
                    matmul(l0c_stage2[stage2_cnt][:valid_m, :N2], l1_stage2_in[stage2_cnt], l1_rhs2[stage2_cnt], m=valid_m, n=N2, k=S1)
                    vc_stage2.free()

                    out[prev_t0, m0:m0 + valid_m, 0:N2] <<= l0c_stage2[stage2_cnt][:valid_m, :N2]
                    stage2_cnt += 1

    return out{main_block}'''


def render_skeleton(
    name: str, topology: str, formula: str, layout: str, grid_mode: str,
    m_split: Optional[int], n_split: Optional[int],
    profile: str, k_loop_mode: str, with_main: bool,
) -> str:
    if topology == "cube-only":
        return render_cube_only(
            name, formula, layout, grid_mode, m_split, n_split, profile, k_loop_mode, with_main,
        )
    if topology == "cube->vec":
        return render_cube_vec(
            name, formula, layout, grid_mode, m_split, n_split, profile, k_loop_mode, with_main,
        )
    if topology == "vec->cube":
        return render_vec_cube(name, formula, layout, grid_mode, m_split, n_split, profile, k_loop_mode, with_main)
    if topology == "vec->cube->vec":
        return render_vec_cube_vec(
            name, formula, layout, grid_mode, m_split, n_split, profile, k_loop_mode, with_main,
        )
    if topology == "cube->vec->cube->vec":
        return render_cube_vec_cube_vec(
            name, formula, layout, grid_mode, m_split, n_split, profile, k_loop_mode, with_main,
        )
    if topology == "vec->cube->vec->cube":
        return render_vec_cube_vec_cube(
            name, formula, layout, grid_mode, m_split, n_split, profile, k_loop_mode, with_main,
        )
    raise ValueError(
        "v1 currently supports only --topology"
        " cube-only|cube->vec|vec->cube|vec->cube->vec|cube->vec->cube->vec|vec->cube->vec->cube"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate repository-style kernel skeletons.")
    parser.add_argument("--name", required=True, help="Kernel basename or filename under kernels/.")
    parser.add_argument(
        "--topology", required=True,
        choices=["cube-only", "cube->vec", "vec->cube", "vec->cube->vec",
                 "cube->vec->cube->vec", "vec->cube->vec->cube"],
        help="Supported topology class.",
    )
    parser.add_argument(
        "--grid-mode", default="tile-m", choices=["tile-m", "tile-n", "mix"],
        help="Tile-grid ownership mode for the generated skeleton.",
    )
    parser.add_argument("--m-split", type=int, help="Required with --grid-mode mix.")
    parser.add_argument("--n-split", type=int, help="Required with --grid-mode mix.")
    parser.add_argument("--formula", default="", help="Optional formula text to place in the file header.")
    parser.add_argument(
        "--layout", default="mknk", choices=["mknk", "kmkn", "custom"],
        help="Layout hint for comments and __main__ reference stub.",
    )
    parser.add_argument("--profile", default="", help="Optional topology-specific profile name.")
    parser.add_argument(
        "--k-loop-mode", default="auto", choices=["auto", "always", "never"],
        help="Whether to scaffold a tiled K loop or a single-shot cube call.",
    )
    parser.add_argument(
        "--print", action="store_true", dest="print_only",
        help="Print the generated skeleton instead of writing a file.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the target file if it already exists.")
    parser.add_argument("--no-main", action="store_true", help="Skip generating the __main__ runnable stub.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        name = _normalize_name(args.name)
        content = render_skeleton(
            name=name,
            topology=args.topology,
            formula=args.formula,
            layout=args.layout,
            grid_mode=args.grid_mode,
            m_split=args.m_split,
            n_split=args.n_split,
            profile=args.profile,
            k_loop_mode=args.k_loop_mode,
            with_main=not args.no_main,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.print_only:
        print(content)
        return 0

    target = KERNEL_DIR / f"{name}.py"
    if target.exists() and not args.force:
        print(f"Target already exists: {target}. Use --force to overwrite.", file=sys.stderr)
        return 1
    target.write_text(content + "\n", encoding="utf-8")
    print(str(target.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
