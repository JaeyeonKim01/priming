"""
This file was used for developing and debugging jvp sdpa.
Please don't use any of this code in production.

Keeping this around for reference - if we plan to extend to improve our jvp sdpa.
"""

import math
import time
from typing import Any

import torch
from jvp_utils.ryu_triton import (
    flash_attention_jvp_multihead_triton_kernel_wrapper,
)
from torch.autograd.forward_ad import (
    _DecoratorContextManager,
    enter_dual_level,
    exit_dual_level,
)

# ruff: noqa: F841


class exit_dual_level_wrapper(_DecoratorContextManager):
    r"""Context-manager for forward AD, where all forward AD computation must occur within the ``dual_level`` context.

    .. Note::

        The ``dual_level`` context appropriately enters and exit the dual level to
        controls the current forward AD level, which is used by default by the other
        functions in this API.

        We currently don't plan to support nested ``dual_level`` contexts, however, so
        only a single forward AD level is supported. To compute higher-order
        forward grads, one can use :func:`torch.func.jvp`.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> x = torch.tensor([1])
        >>> x_t = torch.tensor([1])
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     # Do computations with inp
        ...     out = your_fn(inp)
        ...     _, grad = unpack_dual(out)
        >>> grad is None
        False
        >>> # After exiting the level, the grad is deleted
        >>> _, grad_after = unpack_dual(out)
        >>> grad is None
        True

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """

    def __enter__(self):
        print("enter exit_dual_level_wrapper")
        return exit_dual_level()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        print("exit exit_dual_level_wrapper")
        enter_dual_level()


class VanillaAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        attn = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        attn = torch.softmax(attn, dim=-1)
        x = attn @ v
        return x


class TorchSDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class TritonJVPSDPAInnerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        t_q: torch.Tensor,
        t_k: torch.Tensor,
        t_v: torch.Tensor,
        o: torch.Tensor,
        M: torch.Tensor,
        MU: torch.Tensor,
        LI: torch.Tensor,
        DEBUG_OUT: torch.Tensor,
    ):
        with torch.no_grad():
            o_new, t_o, M, MU, LI = flash_attention_jvp_multihead_triton_kernel_wrapper(
                Q=q,
                K=k,
                V=v,
                t_Q=t_q,
                t_K=t_k,
                t_V=t_v,
                y=o,
                M=M,
                MU=MU,
                LI=LI,
                return_M=True,
                # DEBUG_OUT=DEBUG_OUT,
            )

        return t_o, M, MU, LI

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass
        # ctx.save_for_forward(*inputs, output)
        # ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pass


class TritonJVPSDPAFunction(torch.autograd.Function):
    @staticmethod
    def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, H, S_q, D_q = q.shape
        B, H, S_v, D_v = v.shape

        out_shape = (B, H, S_q, D_v)
        # out = torch.empty(out_shape, device=q.device, dtype=q.dtype)
        out = torch.zeros(out_shape, device=q.device, dtype=q.dtype)
        print("normal forward")
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_forward(*inputs, output)
        # ctx.save_for_backward(*inputs)

    @staticmethod
    def jvp(ctx, t_q: torch.Tensor, t_k: torch.Tensor, t_v: torch.Tensor):
        q, k, v, o = ctx.saved_for_forward
        print("jvp forward")
        t_o = TritonJVPSDPAInnerFunction.apply(q, k, v, t_q, t_k, t_v, o)
        return t_o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pass


class TritonJVPAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return TritonJVPSDPAFunction.apply(q, k, v)


class TritonJVPAttention2(torch.nn.Module):
    class SDPAJVPForwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(_q, _k, _v, t_q, t_k, t_v, y, M, MU, LI, DEBUG_OUT):
            # with torch.inference_mode():
            #     # print(f"jvp {M.data_ptr()=}")
            t_y, M, MU, LI = TritonJVPSDPAInnerFunction.apply(
                _q, _k, _v, t_q, t_k, t_v, y, M, MU, LI, DEBUG_OUT
            )
            return t_y, M, MU, LI

        # @staticmethod
        def setup_context(ctx, inputs: tuple[Any, ...], output: Any) -> Any:
            # ctx.save_for_backward(*inputs, *output[1:])
            # ctx.save_for_backward(*output[1:])
            # t_y, M = output
            # ctx.save_for_backward(*inputs[:-2], inputs[-1])
            ctx.save_for_backward(*inputs, *output[:1])

        @staticmethod
        def backward(ctx, d_t_y, *args):
            # print(args)
            # a_1, b, g, k, f_t, m_t, t_v3, o_t3, _v3, t_q3, _k4, _q3, t_k4 = ctx.saved_tensors
            # grads = SDPAJVPBackwardFunction.apply(a_1, b, g, k, f_t, m_t, t_v3, o_t3, _v3, t_q3, _k4, _q3, t_k4, d_t_y)
            q, k, v, t_q, t_k, t_v, y, M, MU, LI, DEBUG_OUT, y = ctx.saved_tensors
            grads = TritonJVPAttention2.SDPAJVPBackwardFunction.apply(
                # grads = TritonJVPAttention2.SDPAJVPFlashBackwardFunction.apply(
                # *ctx.saved_tensors[:-1], d_t_y
                q,
                k,
                v,
                t_q,
                t_k,
                t_v,
                d_t_y,
                DEBUG_OUT,
                M,
                MU,
                LI,
                y,
            )
            return *grads, None, None, None, None, None

    class SDPAJVPBackwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            _q: torch.Tensor,
            _k: torch.Tensor,
            _v: torch.Tensor,
            t_q: torch.Tensor,
            t_k: torch.Tensor,
            t_v: torch.Tensor,
            tangents_t_o: torch.Tensor,
            DEBUG_OUT: torch.Tensor,
            lse_orig: torch.Tensor | None = None,
            mu_orig: torch.Tensor | None = None,
            li_orig: torch.Tensor | None = None,
            o_orig: torch.Tensor | None = None,
        ):
            q_orig = _q.clone()
            k_orig = _k.clone()
            v_orig = _v.clone()
            t_q_orig = t_q.clone()
            t_k_orig = t_k.clone()
            t_v_orig = t_v.clone()
            tangents_t_o_orig = tangents_t_o.clone()

            B, H, S, D = q_orig.shape

            torch.save(
                {
                    "q_orig": q_orig,
                    "k_orig": k_orig,
                    "v_orig": v_orig,
                    "t_q_orig": t_q_orig,
                    "t_k_orig": t_k_orig,
                    "t_v_orig": t_v_orig,
                    "tangents_o_orig": tangents_t_o_orig,
                    "DEBUG_OUT": DEBUG_OUT,
                    "lse_orig": lse_orig,
                    "mu_orig": mu_orig,
                    "li_orig": li_orig,
                    "o_orig": o_orig,
                },
                f"tmp/debug_data_bhsd_{B}_{H}_{S}_{D}.pt",
            )

            import math

            orig_shape = tangents_t_o.shape
            d_scale = math.sqrt(orig_shape[-1])

            _q_orig_shape = _q.shape
            _q_3d = _q.view(-1, *_q.shape[-2:])
            _q = None
            _k_t = _k.transpose(-1, -2)
            _k = None
            _k_t_3d = _k_t.view(-1, *_k_t.shape[-2:])
            _k_t = None
            a_1 = torch.bmm(_q_3d, _k_t_3d)
            a_2 = a_1.view(*_q_orig_shape[:-2], *a_1.shape[-2:])
            a_test = torch.bmm(_q_3d[:, :16, :128], _k_t_3d[:, :128, :16])
            # print(f"{a_1.shape=} {_q_3d.shape=} {_k_t_3d.shape=}")
            # print(f"{_q_3d[:,:16,:128]=}")
            # print(f"{_k_t_3d[:,:128,:16]=}")
            # print(f"{a_test=}")
            # print(f"{a_1[:,:16,:16]=}")
            a_1 = None
            a = a_2 / d_scale

            # softmax
            amax_sm1 = a_2.amax(dim=-1, keepdim=True)
            sub_sm1 = a_2 - amax_sm1
            a_2 = amax_sm1 = None
            div_sm1 = sub_sm1 / d_scale
            sub_sm1 = None
            exp_sm1 = torch.exp(div_sm1)
            div_sm1 = None
            sum_sm1 = exp_sm1.sum(dim=-1, keepdim=True)
            # print(f"{sum_sm1[:,:16,:16]=}")
            b = exp_sm1 / sum_sm1
            exp_sm1 = sum_sm1 = None
            # end of softmax

            _v_3d = _v.view(-1, *_v.shape[-2:])
            _v = None

            g = a.amax(dim=-1, keepdim=True)
            h = a - g
            # a = None
            i = torch.exp(h)
            h = None
            l_i = i.sum(dim=-1, keepdim=True)
            t_k_t_3d = t_k.transpose(2, 3)
            t_k = None
            t_k_t_3d = t_k_t_3d.reshape(-1, *t_k_t_3d.shape[-2:])
            q_tk = torch.bmm(_q_3d, t_k_t_3d)
            q_tk1 = q_tk.view(*_q_orig_shape[:-2], *q_tk.shape[-2:])
            q_tk = None
            t_q_3d = t_q.view(-1, *t_q.shape[-2:])
            t_q = None
            k_tq = torch.bmm(t_q_3d, _k_t_3d)
            k_tq1 = k_tq.view(*_q_orig_shape[:-2], *k_tq.shape[-2:])
            k_tq = None
            c_t = q_tk1 + k_tq1
            q_tk1 = k_tq1 = None
            e_t = c_t
            c_t = None
            f_t = e_t / d_scale
            e_t = None
            j_t = i * f_t
            # i = None
            l_t = j_t.sum(dim=-1, keepdim=True)
            # j_t = None
            m_t = l_t / l_i
            # print(f"{m_t=}")
            # print(f"{f_t=}")
            # print(f"{l_t=}")
            # print(f"{k=}")
            l_t = None
            n = f_t - m_t
            # f_t = None
            o_t = n * b
            # n_t = None
            o_t_3d = o_t.view(-1, *o_t.shape[-2:])
            o_t = None

            o_t3 = o_t_3d.transpose(-1, -2)
            o_t_3d = None
            _v3 = _v_3d.transpose(-1, -2)
            _v_3d = None
            t_q3 = t_q_3d.transpose(-1, -2)
            t_q_3d = None
            _k4 = _k_t_3d.transpose(-1, -2)
            _k_t_3d = None
            _q3 = _q_3d.transpose(-1, -2)
            _q_3d = None
            t_k4 = t_k_t_3d.transpose(-1, -2)
            t_k_t_3d = None

            # end of forward pass ------------------------------------------------------------

            tangents_t_o_3d = tangents_t_o.reshape(-1, *tangents_t_o.shape[-2:])
            tangents_t_o = None

            #  -------------------- grad_t_v --------------------
            b_3d = b.view(-1, *b.shape[-2:])
            b_3d_t = b_3d.transpose(-1, -2)

            # trying to get grad_t_v using flash backward

            q_flash = q_orig.transpose(-2, -3).contiguous()
            k_flash = k_orig.transpose(-2, -3).contiguous()
            v_flash = v_orig.transpose(-2, -3).contiguous()
            o_flash = o_orig.transpose(-2, -3).contiguous()
            tangents_o_flash = tangents_t_o_orig.transpose(-2, -3).contiguous()
            t_q_flash = t_q_orig.transpose(-2, -3).contiguous()
            t_k_flash = t_k_orig.transpose(-2, -3).contiguous()
            t_v_flash = t_v_orig.transpose(-2, -3).contiguous()

            d_q_flash = torch.zeros_like(q_flash)
            d_k_flash = torch.zeros_like(k_flash)
            d_v_flash = torch.zeros_like(v_flash)

            d_t_q_flash = torch.zeros_like(t_q_flash)
            d_t_k_flash = torch.zeros_like(t_k_flash)
            d_t_v_flash = torch.zeros_like(t_v_flash)

            # TODO @Mathias: this should be allocated with the right shape before
            lse_flash = lse_orig.clone()
            S_lse = lse_flash.shape[-1]
            if S_lse % 128 != 0:
                lse_flash = torch.cat(
                    [
                        lse_flash,
                        torch.zeros(
                            (*lse_flash.shape[:-1], 128 - S_lse % 128),
                            device=lse_flash.device,
                            dtype=lse_flash.dtype,
                        ),
                    ],
                    dim=-1,
                )

            # os.environ["TRITON_INTERPRET"] = "1"

            # importlib.reload(triton)
            from jvp_utils.flash_jvp_backward import (
                _flash_attn_backward as flash_jvp_backward,
            )

            # dto, q, k, v, o, tq, tk, tv, lse, dq, dk, dv, dtq, dtk, dtv, softmax_scale=None
            flash_jvp_backward(
                tangents_o_flash,
                q_flash,
                k_flash,
                v_flash,
                o_flash,
                t_q_flash,
                t_k_flash,
                t_v_flash,
                lse_flash,
                mu_orig,
                li_orig,
                d_q_flash,
                d_k_flash,
                d_v_flash,
                d_t_q_flash,
                d_t_k_flash,
                d_t_v_flash,
            )
            d_v_flash = d_v_flash.transpose(-2, -3).contiguous()
            d_k_flash = d_k_flash.transpose(-2, -3).contiguous()
            d_q_flash = d_q_flash.transpose(-2, -3).contiguous()
            d_t_v_flash = d_t_v_flash.transpose(-2, -3).contiguous()
            d_t_k_flash = d_t_k_flash.transpose(-2, -3).contiguous()
            d_t_q_flash = d_t_q_flash.transpose(-2, -3).contiguous()
            # return (
            #     d_q_flash,
            #     d_k_flash,
            #     d_v_flash,
            #     d_t_q_flash,
            #     d_t_k_flash,
            #     d_t_v_flash,
            # )

            # print(f"{a=}")
            # print(f"{b_3d[...,:5,:5]=}")
            # print(f"{i=}")
            # print(f"{m_t=}")
            # print(f"{n[...,:5,:5]=}")
            # print(f"{f_t.T[...,:5,:5]=}")
            # print(f"{j_t[...,:5,:5]=}")
            # print(f"{i[...,:5,:5]=}")
            # print(f"{o_t3.transpose(-1, -2)[...,:5,:5]=}")
            # print(f"{DEBUG_OUT[...,:5,:5]=}")
            # breakpoint()
            b_3d = None
            grad_t_v_1 = torch.bmm(b_3d_t, tangents_t_o_3d)

            grad_t_v = grad_t_v_1.view(*orig_shape[:-2], *grad_t_v_1.shape[-2:])
            grad_t_v_1 = None

            print(f"{grad_t_v=}")
            print(f"{d_t_v_flash=}")

            #  -------------------- grad_v --------------------
            grad_v = torch.bmm(o_t3, tangents_t_o_3d)
            # print(f"{o_t3=}")
            # print(f"{o_t3.shape=}")
            o_t3 = None
            grad_v = grad_v.view(*orig_shape[:-2], *grad_v.shape[-2:])

            print(f"{grad_v=}")
            print(f"{d_v_flash=}")

            # ----------------- other grads -----------------

            # t_v_3d = t_v.view(-1, *t_v.shape[-2:])
            # t_v = None
            # t_v3 = t_v_3d.transpose(-1, -2)
            # t_v_3d = None
            tan_to_mm_tv = torch.bmm(
                tangents_t_o_3d, t_v.view(-1, *t_v.shape[-2:]).transpose(-1, -2)
            )  # q back
            tan_to_mm_v = torch.bmm(tangents_t_o_3d, _v3)  # p back

            # print(f"{tan_to_mm_tv[...,:5,:5]=}")
            # print(f"{tan_to_mm_v[...,:5,:5]=}")

            tangents_t_o_3d = None
            t_v3 = None
            tan_o_mm_tv1 = tan_to_mm_tv.view(
                *orig_shape[:-2], *tan_to_mm_tv.shape[-2:]
            )  # q back
            tan_to_mm_tv = None

            tan_o_mm_v1 = tan_to_mm_v.view(
                *orig_shape[:-2], *tan_to_mm_v.shape[-2:]
            )  # p back
            tan_to_mm_v = None

            # sub_3 = f_t - m_t # this is n
            tan_o_mm_v_m_n = tan_o_mm_v1 * n
            n = None
            tan_o_mm_v_m_b = tan_o_mm_v1 * b
            tan_o_mm_v1 = None
            add_2 = tan_o_mm_tv1 + tan_o_mm_v_m_n
            # print(f"{add_2[...,:5,:5]=}")
            # print(f"{tan_o_mm_tv1[...,:5,:5]=}")
            # print(f"{tan_o_mm_v_m_n[...,:5,:5]=}")

            tan_o_mm_tv1 = tan_o_mm_v_m_n = None
            sum_4 = tan_o_mm_v_m_b.sum(dim=-1, keepdim=True)
            # print(f"{sum_4[0,0,:,0]=}")
            # print(f"{m_t[...,:5,:5]=}")
            # print(f"{l_i[...,:5,:5]=}")
            m_t_div_k = m_t / l_i
            m_t = None
            mul_5 = sum_4 * m_t_div_k
            m_t_div_k = None
            div_6 = -sum_4 / l_i

            # test = i / l_i
            # print(f"{test[...,:5,:5]=}")
            # print(f"{b[...,:5,:5]=}")
            # print(f"{i[...,:5,:5]=}")
            # print(f"{l_i[...,:5,:5]=}")
            # print(f"{f_t[...,:5,:5]=}")
            # print(f"{(f_t / l_i)[...,:5,:5]=}")

            # TODO @Mathias: uncomment below
            # sum_4 = l_i = None

            div_6_1 = div_6.view(*orig_shape[:-2], *div_6.shape[-2:])
            div_6 = None

            mul_6 = div_6_1 * i
            # breakpoint()

            mul_7 = div_6_1 * f_t
            # print(f"{mul_7[...,:5,:5]=}")
            # print(f"{div_6_1[...,:5,:5]=}")
            # print(f"{f_t[...,:5,:5]=}")
            div_6_1 = f_t = None
            add_3 = tan_o_mm_v_m_b + mul_6
            tan_o_mm_v_m_b = mul_6 = None

            div_7 = add_3 / d_scale
            add_3 = None

            view_24 = div_7.view(-1, *div_7.shape[-2:])
            div_7 = None
            ktq_grad_k = torch.bmm(t_q3, view_24)  #  ktq -> k grad backward
            t_q3 = None
            grad_t_q_1 = torch.bmm(view_24, _k4)

            #  -------------------- grad_t_q --------------------
            grad_t_q = grad_t_q_1.view(*orig_shape[:-2], *grad_t_q_1.shape[-2:])
            grad_t_q_1 = None

            print(f"{grad_t_q[...,:5,:5]=}")
            print(f"{d_t_q_flash[...,:5,:5]=}")

            bmm_12 = torch.bmm(_q3, view_24)

            view_28 = bmm_12.view(*orig_shape[:-2], *bmm_12.shape[-2:])
            bmm_12 = None

            #  -------------------- grad_t_k --------------------
            grad_t_k = view_28.transpose(-1, -2)
            view_28 = None

            print(f"{grad_t_k[...,:5,:5]=}")
            print(f"{d_t_k_flash[...,:5,:5]=}")

            ktq_grad_k_1 = ktq_grad_k.view(*orig_shape[:-2], *ktq_grad_k.shape[-2:])
            ktq_grad_k = None
            ktq_grad_k_t = ktq_grad_k_1.transpose(-1, -2)
            # print(f"{ktq_grad_k_t[...,:5,:5]=}")
            ktq_grad_k_1 = None

            qtk_grad_q = torch.bmm(view_24, t_k4)
            view_24 = t_k4 = None
            qkt_grad_q_1 = qtk_grad_q.view(*orig_shape[:-2], *qtk_grad_q.shape[-2:])
            # print(f"{qkt_grad_q_1[...,:5,:5]=}")
            qtk_grad_q = None

            add_4 = mul_7 + mul_5
            # print(f"{add_4[...,:5,:5]=}")
            # print(f"{mul_7[...,:5,:5]=}")
            # print(f"{mul_5[...,:5,:5]=}")
            mul_7 = mul_5 = None
            mul_8 = add_4 * i
            # print(f"{i[...,:5,:5]=}")
            # print(f"{mul_8[...,:5,:5]=}")
            add_4 = i = None

            # amax backward
            sum_5 = -mul_8.sum(dim=-1, keepdim=True)
            eq = g == a
            a = None
            sum_6 = eq.sum(dim=-1, keepdim=True)
            div_8 = sum_5 / sum_6
            sum_5 = sum_6 = None
            mul_9 = div_8 * eq
            div_8 = eq = None

            add_5 = mul_8 + mul_9
            mul_8 = mul_9 = None

            mul_11 = add_2 * b
            add_8 = None
            sum_7 = mul_11.sum(dim=-1, keepdim=True)
            # print(f"{sum_7[0,0,:,0]=}")
            # breakpoint()

            neg_4 = -b
            fma = torch.addcmul(mul_11, neg_4, sum_7)
            # print(f"{fma[...,:5,:5]=}")

            neg_4 = sum_7 = mul_11 = None
            add_9 = add_5 + fma
            # print(f"{add_9[...,:5,:5]=}")
            add_5 = fma = None

            div_9 = add_9 / d_scale
            # print(f"{div_9[...,:5,:5]=}")
            add_9 = None
            view_33 = div_9.view(
                -1, *div_9.shape[-2:]
            )  # this is basically the equivalent to "ds" in tri dao's triton bw
            div_9 = None

            a1_back_k_grad = torch.bmm(_q3, view_33)  #  a1 back -> k gradient
            # print(f"{a1_back_k_grad[...,:5,:5]=}")
            _q3 = None
            a1_back_q_grad = torch.bmm(view_33, _k4)  #  a1 back -> q gradient
            # print(f"{a1_back_q_grad[...,:5,:5]=}")
            view_33 = _k4 = None

            a1_back_k_grad_1 = a1_back_k_grad.view(
                *orig_shape[:-2], *a1_back_k_grad.shape[-2:]
            )
            a1_back_k_grad = None
            a1_back_q_grad_1 = a1_back_q_grad.view(
                *orig_shape[:-2], *a1_back_q_grad.shape[-2:]
            )
            a1_back_q_grad = None

            grad_q = qkt_grad_q_1 + a1_back_q_grad_1
            print(f"{grad_q[...,:5,:5]=}")
            print(f"{d_q_flash[...,:5,:5]=}")
            qkt_grad_q_1 = a1_back_q_grad_1 = None

            a1_back_k_grad_t = a1_back_k_grad_1.transpose(-1, -2)
            a1_back_k_grad_1 = None

            grad_k = ktq_grad_k_t + a1_back_k_grad_t
            print(f"{grad_k[...,:5,:5]=}")
            print(f"{d_k_flash[...,:5,:5]=}")
            ktq_grad_k_t = a1_back_k_grad_t = None
            return (grad_q, grad_k, grad_v, grad_t_q, grad_t_k, grad_t_v)

        @staticmethod
        def backward(ctx, d_q, d_k, d_v, d_t_q, d_t_k, d_t_v):
            raise NotImplementedError("Not implemented")

    class SDPAJVPFlashBackwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            _q: torch.Tensor,
            _k: torch.Tensor,
            _v: torch.Tensor,
            t_q: torch.Tensor,
            t_k: torch.Tensor,
            t_v: torch.Tensor,
            tangents_t_o: torch.Tensor,
            DEBUG_OUT: torch.Tensor,
            lse_orig: torch.Tensor | None = None,
            mu_orig: torch.Tensor | None = None,
            li_orig: torch.Tensor | None = None,
            o_orig: torch.Tensor | None = None,
        ):
            q_orig = _q  # _q.clone()
            k_orig = _k  # _k.clone()
            v_orig = _v  # _v.clone()
            t_q_orig = t_q  # t_q.clone()
            t_k_orig = t_k  # t_k.clone()
            t_v_orig = t_v  # t_v.clone()
            tangents_t_o_orig = tangents_t_o  # tangents_t_o.clone()

            B, H, S, D = q_orig.shape

            # torch.save(
            #     {
            #         "q_orig": q_orig,
            #         "k_orig": k_orig,
            #         "v_orig": v_orig,
            #         "t_q_orig": t_q_orig,
            #         "t_k_orig": t_k_orig,
            #         "t_v_orig": t_v_orig,
            #         "tangents_o_orig": tangents_t_o_orig,
            #         "DEBUG_OUT": DEBUG_OUT,
            #         "lse_orig": lse_orig,
            #         "mu_orig": mu_orig,
            #         "li_orig": li_orig,
            #         "o_orig": o_orig,
            #     },
            #     f"tmp/debug_data_bhsd_{B}_{H}_{S}_{D}.pt",
            # )

            import math

            orig_shape = tangents_t_o.shape
            d_scale = math.sqrt(orig_shape[-1])

            q_flash = q_orig.transpose(-2, -3).contiguous()
            k_flash = k_orig.transpose(-2, -3).contiguous()
            v_flash = v_orig.transpose(-2, -3).contiguous()
            o_flash = o_orig.transpose(-2, -3).contiguous()
            tangents_o_flash = tangents_t_o_orig.transpose(-2, -3).contiguous()
            t_q_flash = t_q_orig.transpose(-2, -3).contiguous()
            t_k_flash = t_k_orig.transpose(-2, -3).contiguous()
            t_v_flash = t_v_orig.transpose(-2, -3).contiguous()

            d_q_flash = torch.zeros_like(q_flash)
            d_k_flash = torch.zeros_like(k_flash)
            d_v_flash = torch.zeros_like(v_flash)

            d_t_q_flash = torch.zeros_like(t_q_flash)
            d_t_k_flash = torch.zeros_like(t_k_flash)
            d_t_v_flash = torch.zeros_like(t_v_flash)

            # TODO @Mathias: this should be allocated with the right shape before
            lse_flash = lse_orig.clone()
            S_lse = lse_flash.shape[-1]
            if S_lse % 128 != 0:
                lse_flash = torch.cat(
                    [
                        lse_flash,
                        torch.zeros(
                            (*lse_flash.shape[:-1], 128 - S_lse % 128),
                            device=lse_flash.device,
                            dtype=lse_flash.dtype,
                        ),
                    ],
                    dim=-1,
                )

            from kuma.projects.accel.sdpa_jvp.flash_jvp_backward import (
                _flash_attn_backward as flash_jvp_backward,
            )

            # dto, q, k, v, o, tq, tk, tv, lse, dq, dk, dv, dtq, dtk, dtv, softmax_scale=None
            flash_jvp_backward(
                tangents_o_flash,
                q_flash,
                k_flash,
                v_flash,
                o_flash,
                t_q_flash,
                t_k_flash,
                t_v_flash,
                lse_flash,
                mu_orig,
                li_orig,
                d_q_flash,
                d_k_flash,
                d_v_flash,
                d_t_q_flash,
                d_t_k_flash,
                d_t_v_flash,
            )
            d_v_flash = d_v_flash.transpose(-2, -3).contiguous()
            d_k_flash = d_k_flash.transpose(-2, -3).contiguous()
            d_q_flash = d_q_flash.transpose(-2, -3).contiguous()
            d_t_v_flash = d_t_v_flash.transpose(-2, -3).contiguous()
            d_t_k_flash = d_t_k_flash.transpose(-2, -3).contiguous()
            d_t_q_flash = d_t_q_flash.transpose(-2, -3).contiguous()
            return (
                d_q_flash,
                d_k_flash,
                d_v_flash,
                d_t_q_flash,
                d_t_k_flash,
                d_t_v_flash,
            )

        @staticmethod
        def backward(ctx, d_q, d_k, d_v, d_t_q, d_t_k, d_t_v):
            raise NotImplementedError("Not implemented")

    class SDPABackwardFunction(torch.autograd.Function):
        # @staticmethod
        # def forward(ctx, q, k, v, o, do, M):
        #     ctx.sm_scale = 1.0 / math.sqrt(q.shape[-1])
        #     ctx.HEAD_DIM = q.shape[-1]
        #     assert do.is_contiguous()
        #     # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        #     for i, s in enumerate(q.shape):
        #         if s > 1:
        #             assert q.stride(i) == k.stride(i) == v.stride(i) == o.stride(i) == do.stride(i)
        #     dq = torch.empty_like(q)
        #     dk = torch.empty_like(k)
        #     dv = torch.empty_like(v)
        #     BATCH, N_HEAD, N_CTX = q.shape[:3]
        #     NUM_WARPS, NUM_STAGES = 4, 5
        #     BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        #     # BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
        #     BLK_SLICE_FACTOR = 2
        #     RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        #     arg_k = k
        #     arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        #     PRE_BLOCK = 128
        #     assert N_CTX % PRE_BLOCK == 0
        #     pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        #     delta = torch.empty_like(M)
        #     from tmp.triton_fused_attn import _attn_bwd_preprocess, _attn_bwd
        #     _attn_bwd_preprocess[pre_grid](
        #         o, do,  #
        #         delta,  #
        #         BATCH, N_HEAD, N_CTX,  #
        #         BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        #     )
        #     grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        #     _attn_bwd[grid](
        #         q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
        #         M, delta,  #
        #         q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        #         N_HEAD, N_CTX,  #
        #         BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
        #         BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
        #         BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
        #         HEAD_DIM=ctx.HEAD_DIM,  #
        #         num_warps=NUM_WARPS,  #
        #         num_stages=NUM_STAGES  #
        #     )
        #     return dq, dk, dv

        @staticmethod
        def forward(ctx, q, k, v, o, do, M):
            # assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
            # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
            # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
            # print(f"backward {M.data_ptr()=}")
            with torch.inference_mode():
                q = q.transpose(-2, -3).contiguous()
                k = k.transpose(-2, -3).contiguous()
                v = v.transpose(-2, -3).contiguous()
                o = o.transpose(-2, -3).contiguous()
                do = do.transpose(-2, -3).contiguous()
                lse = M
                dq = torch.empty_like(q)
                dk = torch.empty_like(k)
                dv = torch.empty_like(v)
                from jvp_utils.flash_jvp_backward import (
                    _flash_attn_backward,
                )

                # TODO @Mathias: this should be allocated with the right shape before
                S_lse = lse.shape[-1]
                if S_lse % 128 != 0:
                    lse = torch.cat(
                        [
                            lse,
                            torch.zeros(
                                (*lse.shape[:-1], 128 - S_lse % 128),
                                device=lse.device,
                                dtype=lse.dtype,
                            ),
                        ],
                        dim=-1,
                    )

                _flash_attn_backward(
                    do,
                    q,
                    k,
                    v,
                    o,
                    lse,
                    dq,
                    dk,
                    dv,
                    bias=None,
                    causal=False,
                    softmax_scale=None,
                    # softmax_scale=1.0 / math.sqrt(q.shape[-1]),
                )
                dq = dq.transpose(-2, -3).contiguous()
                dk = dk.transpose(-2, -3).contiguous()
                dv = dv.transpose(-2, -3).contiguous()
            return dq, dk, dv

        @staticmethod
        def backward(ctx, d_q, d_k, d_v, d_t_q, d_t_k, d_t_v):
            raise NotImplementedError("Not implemented")

    class SDPAFunction(torch.autograd.Function):
        @staticmethod
        def forward(q, k, v):
            # attn = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            # attn = torch.softmax(attn, dim=-1)
            # x = attn @ v
            # return x
            B, H, S, D = q.shape
            D_v = v.shape[-1]
            y = torch.empty((B, H, S, D_v), device=q.device, dtype=q.dtype)
            # lse = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
            S_div_up = ((S + 127) // 128) * 128
            M = torch.empty((B, H, S_div_up), device=q.device, dtype=torch.float32)
            MU = torch.empty((B, H, S_div_up), device=q.device, dtype=torch.float32)
            LI = torch.empty((B, H, S_div_up), device=q.device, dtype=torch.float32)
            # q, k, v = [x.transpose(-2, -3).contiguous() for x in [q, k, v]]
            # q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
            # from kuma.projects.accel.sdpa_jvp.flash_attn_triton import (
            #     _flash_attn_forward,
            # )

            # o, lse, *_ = _flash_attn_forward(
            #     q, k, v, bias=None, causal=False, softmax_scale=None
            # )
            # o = o.transpose(-2, -3).contiguous()
            # y = o
            # ctx.save_for_backward(q, k, v, o, lse, bias)
            # ctx.causal = causal
            return y, M, MU, LI

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            q = inputs[0]
            B, H, S = q.shape[:3]  # pre-allocated softmax stats
            # M = torch.zeros((B, H, S), device=q.device, dtype=torch.float32)
            y, M, MU, LI = output
            # M = lse
            # print(f"{y.shape=} {lse.shape=} {M.shape=}")
            ctx.save_for_forward(*inputs, y, M, MU, LI)
            ctx.save_for_backward(*inputs, y, M, MU, LI)
            ctx.set_materialize_grads(False)

        @staticmethod
        def jvp(ctx, t_q, t_k, t_v):
            q, k, v, y, M, MU, LI = ctx.saved_for_forward
            DEBUG_OUT = torch.zeros_like(y)
            t_y, _, _, _ = TritonJVPAttention2.SDPAJVPForwardFunction.apply(
                q, k, v, t_q, t_k, t_v, y, M, MU, LI, DEBUG_OUT
            )
            # print(f"{M.shape=} {t_y.shape=} {M.view(-1)[:10]=} {t_y.view(-1)[:10]=}")
            return t_y, M, MU, LI

        @staticmethod
        def backward(ctx, d_y, d_M, d_MU, d_LI):
            with torch.inference_mode():
                if d_y is None:
                    return None, None, None
                q, k, v, y, M, MU, LI = ctx.saved_tensors
                # M_err = (lse-M)
                # print(f"{(lse-M).abs().max()=} {M_err.view(-1)[:10]=} {M.view(-1)[:10]=} {lse.view(-1)[:10]=}")
                d_y = d_y.contiguous()
                dq, dk, dv = TritonJVPAttention2.SDPABackwardFunction.apply(
                    q, k, v, y, d_y, M
                )
            return dq, dk, dv

            # print("Backwards Not implemented, but grad is None anyways")
            # return None, None, None

    def forward(self, q, k, v):
        y, M, MU, LI = TritonJVPAttention2.SDPAFunction.apply(q, k, v)
        return y


class Block(torch.nn.Module):
    def __init__(
        self, d: int, n_heads: int, attn_backend: str, skip_o_proj: bool = False
    ):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d * n_heads)
        self.k_proj = torch.nn.Linear(d, d * n_heads)
        self.v_proj = torch.nn.Linear(d, d * n_heads)
        self.o_proj = torch.nn.Linear(d * n_heads, d)
        self.act = torch.nn.Identity()
        self.q_norm = torch.nn.RMSNorm(d)
        self.k_norm = torch.nn.RMSNorm(d)
        if attn_backend == "vanilla":
            self.attn = VanillaAttention()
        elif attn_backend == "triton":
            self.attn = TritonJVPAttention()
        elif attn_backend == "triton2":
            self.attn = TritonJVPAttention2()
        elif attn_backend == "torch":
            self.attn = TorchSDPA()
        else:
            raise ValueError(f"Invalid attention backend: {attn_backend}")
        self.n_heads = n_heads
        self.d = d
        self.skip_o_proj = skip_o_proj

    def forward(self, x):
        q = self.q_proj(x)
        q_orig_shape = q.shape
        k = self.k_proj(x)
        v = self.v_proj(x)
        v = self.act(v)

        q = q.view(*q.shape[:-1], self.n_heads, self.d).transpose(-2, -3).contiguous()
        k = k.view(*k.shape[:-1], self.n_heads, self.d).transpose(-2, -3).contiguous()
        v = v.view(*v.shape[:-1], self.n_heads, self.d).transpose(-2, -3).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        o = self.attn(q, k, v)

        o = o.transpose(-2, -3).contiguous().view(*q_orig_shape).contiguous()
        if not self.skip_o_proj:
            o = self.o_proj(o)
            o = self.act(o)
        return o


class Model(torch.nn.Module):
    def __init__(self, d: int, n_blocks: int, n_heads: int, attn_backend: str):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                Block(d, n_heads, attn_backend, skip_o_proj=i == n_blocks - 1)
                for i in range(n_blocks)
            ]
        )

    def forward(self, x):
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
        return x


def test_jvp_fw():
    b = 1
    s = 128
    h = 16
    d = 128
    n_blocks = 2
    device = torch.device("cuda")

    model_vanilla = Model(d=d, n_blocks=n_blocks, n_heads=h, attn_backend="torch").to(
        device
    )
    model = Model(d=d, n_blocks=n_blocks, n_heads=h, attn_backend="triton").to(device)
    model.load_state_dict(model_vanilla.state_dict())

    x = torch.randn((b, s, d), device=device)
    t_x = torch.randn_like(x)
    print(torch.cuda.memory_summary(device=device))
    torch.cuda.synchronize()
    y_ref = model_vanilla(x)
    y, t_y = torch.func.jvp(model, (x,), (t_x,))
    torch.cuda.synchronize()
    print(torch.cuda.memory_summary(device=device))

    y_err = y - y_ref
    print(y_err.abs().max())
    print("")


def print_jvp_sdpa_backward():
    torch.manual_seed(123)
    b = 1
    s = 128
    h = 16
    d = 128
    device = torch.device("cuda")

    q, k, v = torch.randn((3, b, h, s, d), device=device)
    q.view(-1)[0] = -1
    k.view(-1)[0] = -2
    v.view(-1)[0] = -3
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    t_q = torch.randn_like(q)
    t_k = torch.randn_like(k)
    t_v = torch.randn_like(v)
    t_q.view(-1)[0] = 1
    t_k.view(-1)[0] = 2
    t_v.view(-1)[0] = 3
    t_q.requires_grad_(True)
    t_k.requires_grad_(True)
    t_v.requires_grad_(True)

    # def model(q, k, v):
    #     # return TritonJVPSDPAFunction.apply(q, k, v)
    #     return TritonJVPSDPAFunction.apply(q, k, v)

    model = VanillaAttention()

    y, t_y = torch.func.jvp(model, (q, k, v), (t_q, t_k, t_v))
    # grad = torch.autograd.grad(t_y.mean(), (q, k, v, t_q, t_k, t_v), create_graph=True, retain_graph=True)
    grad = torch.autograd.grad(
        t_y.mean(), (q, k, v), create_graph=True, retain_graph=True
    )
    # print(q.grad)
    # print(k.grad)
    # print(v.grad)

    print("")

    from torchviz import make_dot

    make_dot(t_y.mean(), params={}, show_attrs=True).render(
        "0_jvp_sdpa_backward", format="png"
    )
    make_dot(y.mean(), params={}, show_attrs=True).render(
        "0_sdpa_backward", format="png"
    )
    make_dot(grad[0].mean(), params={}, show_attrs=True).render(
        "0_grad_q", format="png"
    )
    make_dot(grad[1].mean(), params={}, show_attrs=True).render(
        "0_grad_k", format="png"
    )
    make_dot(grad[2].mean(), params={}, show_attrs=True).render(
        "0_grad_v", format="png"
    )
    # make_dot(grad[3].mean(), params={}).render("0_grad_t_q", format="png")
    # make_dot(grad[4].mean(), params={}).render("0_grad_t_k", format="png")
    # make_dot(grad[5].mean(), params={}).render("0_grad_t_v", format="png")

    # model_compiled = torch.compile(model, backend="aot_eager")
    # print("")


def test_functionalize():
    from torch.func import functionalize
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(x):
        y = x.clone()
        y.add_(1)
        return y

    functionalized_f = functionalize(f)
    x = torch.ones(4)
    print(torch.allclose(functionalized_f(x), f(x)))

    fx_g = make_fx(functionalized_f)(x)
    print(fx_g.code)


def test_linearize():
    import torch
    from torch.func import linearize
    from torch.fx.experimental.proxy_tensor import make_fx

    def fn(x):
        y = x.sin()
        return y

    output, jvp_fn = linearize(fn, torch.zeros(3, 3, requires_grad=True))
    fx_jvp = make_fx(jvp_fn)(torch.ones(3, 3))
    print(fx_jvp.code)


def jvp_sdpa_fused_vanilla(_q, _k, _v, t_q, t_k, t_v):
    # standard SDPA
    a = _q @ _k.transpose(-2, -1) / math.sqrt(_q.shape[-1])
    b = torch.softmax(a, dim=-1)
    y = b @ _v

    # all these values are independent of the tangents -> just subparts of the softmax
    d = a * 0  # TODO: check if correct

    g = torch.amax(a, dim=-1, keepdim=True)
    h = a - g
    i = torch.exp(h)
    k = i.sum(dim=-1, keepdim=True)

    # all variables with _t or _t(qkv) suffix depend on the tangents
    q_tk = _q @ t_k.transpose(-2, -1)
    k_tq = t_q @ _k.transpose(-2, -1)

    c_t = q_tk + k_tq

    e_t = c_t - d
    f_t = e_t / math.sqrt(_q.shape[-1])
    j_t = i * f_t

    l_t = j_t.sum(dim=-1, keepdim=True)
    m_t = l_t / k

    n_t = f_t - m_t

    o_t = n_t * b

    p_t = o_t @ _v

    q_t = b @ t_v

    t_y = p_t + q_t
    return y, t_y


def jvp_sdpa_fused_vanilla_2(_q, _k, _v, t_q, t_k, t_v):
    # standard SDPA
    a = _q @ _k.transpose(-2, -1) / math.sqrt(_q.shape[-1])
    # b = torch.softmax(a, dim=-1)
    g = torch.amax(a, dim=-1, keepdim=True)
    h = a - g
    i = torch.exp(h)
    k = i.sum(dim=-1, keepdim=True)
    b = i / k

    # y = b @ _v

    # all these values are independent of the tangents -> just subparts of the softmax
    # d = a * 0  # TODO: check if correct

    # g = torch.amax(a, dim=-1, keepdim=True)
    # h = a - g
    # i = torch.exp(h)
    # k = i.sum(dim=-1, keepdim=True)

    # all variables with _t or _t(qkv) suffix depend on the tangents
    q_tk = _q @ t_k.transpose(-2, -1)
    k_tq = t_q @ _k.transpose(-2, -1)

    c_t = q_tk + k_tq

    e_t = c_t
    f_t = e_t / math.sqrt(_q.shape[-1])
    j_t = i * f_t

    l_t = j_t.sum(dim=-1, keepdim=True)
    m_t = l_t / k

    n_t = f_t - m_t

    o_t = n_t * b

    p_t = o_t @ _v

    q_t = b @ t_v

    t_y = p_t + q_t
    # return y, t_y
    return t_y


def print_jvp_sdpa_fused_vanilla():
    torch.manual_seed(123)
    b = 1
    s = 128
    h = 16
    d = 128
    device = torch.device("cuda")

    q, k, v = torch.randn((3, b, h, s, d), device=device)
    q.view(-1)[0] = -1
    k.view(-1)[0] = -2
    v.view(-1)[0] = -3
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    t_q = torch.randn_like(q)
    t_k = torch.randn_like(k)
    t_v = torch.randn_like(v)
    t_q.view(-1)[0] = 1
    t_k.view(-1)[0] = 2
    t_v.view(-1)[0] = 3
    t_q.requires_grad_(True)
    t_k.requires_grad_(True)
    t_v.requires_grad_(True)

    y, t_y = jvp_sdpa_fused_vanilla(q, k, v, t_q, t_k, t_v)
    from torchviz import make_dot

    make_dot(t_y.mean(), params={}, show_attrs=True).render(
        "0_jvp_sdpa_fused_vanilla", format="png"
    )


def test_compile_jvp_sdpa_fused_vanilla_graph():
    def model(q, k, v, t_q, t_k, t_v):
        return jvp_sdpa_fused_vanilla(q, k, v, t_q, t_k, t_v)

    model_compiled = torch.compile(model)
    b, h, s, d = 1, 16, 128, 128
    q, k, v = torch.randn((3, b, h, s, d), device=torch.device("cuda"))
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    t_q, t_k, t_v = (
        torch.randn_like(q, requires_grad=True),
        torch.randn_like(k, requires_grad=True),
        torch.randn_like(v, requires_grad=True),
    )
    y, t_y = model_compiled(q, k, v, t_q, t_k, t_v)

    t_y.sum().backward()
    print(y)


def test_compile_jvp_sdpa_fused_vanilla_2_graph():
    def model(q, k, v, t_q, t_k, t_v):
        return jvp_sdpa_fused_vanilla_2(q, k, v, t_q, t_k, t_v)

    model_compiled = torch.compile(model)
    b, h, s, d = 1, 16, 128, 128
    q, k, v = torch.randn((3, b, h, s, d), device=torch.device("cuda"))
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    t_q, t_k, t_v = (
        torch.randn_like(q, requires_grad=True),
        torch.randn_like(k, requires_grad=True),
        torch.randn_like(v, requires_grad=True),
    )
    # y, t_y = model_compiled(q, k, v, t_q, t_k, t_v)
    t_y = model_compiled(q, k, v, t_q, t_k, t_v)

    t_y.sum().backward()
    # print(y)


def test_compiled_graph():
    model = VanillaAttention()

    model_compiled = torch.compile(model)

    q, k, v = torch.randn((3, 1, 16, 128), device=torch.device("cuda"))
    t_q, t_k, t_v = torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)
    y, t_y = torch.func.jvp(model_compiled, (q, k, v), (t_q, t_k, t_v))
    # y, t_y = model_compiled(q, k, v, t_q, t_k, t_v)
    # print(y)


def test_flattened_autograd():
    class SDPAJVPForwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, t_q, t_k, t_v):
            k_1 = torch.ops.aten.permute.default(k, [0, 2, 1])
            k = None
            q_1 = torch.ops.aten.expand.default(q, [1, 16, 128])
            q = None
            k_2 = torch.ops.aten.expand.default(k_1, [1, 128, 16])
            k_1 = None
            a_1 = torch.ops.aten.bmm.default(q_1, k_2)
            a = torch.ops.aten.div.Tensor(a_1, 11.313708498984761)
            a_2 = torch.ops.aten.mul.Tensor(a_1, 1)
            g_1 = torch.ops.aten.amax.default(a_2, [-1], True)
            h_1 = torch.ops.aten.sub.Tensor(a_2, g_1)
            a_2 = g_1 = None
            h_2 = torch.ops.aten.div.Tensor(h_1, 11.313708498984761)
            h_1 = None
            i_1 = torch.ops.aten.exp.default(h_2)
            h_2 = None
            k_1 = torch.ops.aten.sum.dim_IntList(i_1, [-1], True)
            b = torch.ops.aten.div.Tensor(i_1, k_1)
            i_1 = k_1 = None
            b_1 = torch.ops.aten.expand.default(b, [1, 16, 16])
            v_1 = torch.ops.aten.expand.default(v, [1, 16, 128])
            v = None
            y = torch.ops.aten.bmm.default(b_1, v_1)  # y
            d = torch.ops.aten.mul.Tensor(a, 0)
            g = torch.ops.aten.amax.default(a, [-1], True)
            h = torch.ops.aten.sub.Tensor(a, g)
            a = None
            i = torch.ops.aten.exp.default(h)
            h = None
            k_2 = torch.ops.aten.sum.dim_IntList(i, [-1], True)  # k
            t_k1 = torch.ops.aten.permute.default(t_k, [0, 2, 1])
            t_k = None
            t_k2 = torch.ops.aten.expand.default(t_k1, [1, 128, 16])
            t_k1 = None
            q_tk = torch.ops.aten.bmm.default(q_1, t_k2)
            t_q2 = torch.ops.aten.expand.default(t_q, [1, 16, 128])
            t_q = None
            k_tq = torch.ops.aten.bmm.default(t_q2, k_2)
            c_t = torch.ops.aten.add.Tensor(q_tk, k_tq)
            q_tk = k_tq = None
            e_t = torch.ops.aten.sub.Tensor(c_t, d)
            c_t = d = None
            f_t = torch.ops.aten.div.Tensor(e_t, 11.313708498984761)
            e_t = None
            j_t = torch.ops.aten.mul.Tensor(i, f_t)
            i = None
            l_t = torch.ops.aten.sum.dim_IntList(j_t, [-1], True)
            j_t = None
            m_t = torch.ops.aten.div.Tensor(l_t, k_2)
            l_t = None
            n_t = torch.ops.aten.sub.Tensor(f_t, m_t)
            o_t = torch.ops.aten.mul.Tensor(n_t, b)
            n_t = None
            o_t1 = torch.ops.aten.expand.default(o_t, [1, 16, 16])
            o_t = None
            p_t = torch.ops.aten.bmm.default(o_t1, v_1)
            t_v1 = torch.ops.aten.expand.default(t_v, [1, 16, 128])
            t_v = None
            q_t = torch.ops.aten.bmm.default(b_1, t_v1)
            b_1 = None
            t_y = torch.ops.aten.add.Tensor(p_t, q_t)
            p_t = q_t = None  # t_y
            t_v2 = torch.ops.aten.permute.default(t_v1, [0, 2, 1])
            t_v1 = None
            o_t2 = torch.ops.aten.permute.default(o_t1, [0, 2, 1])
            o_t1 = None
            v_2 = torch.ops.aten.permute.default(v_1, [0, 2, 1])
            v_1 = None
            t_q3 = torch.ops.aten.permute.default(t_q2, [0, 2, 1])
            t_q2 = None
            k_3 = torch.ops.aten.permute.default(k_2, [0, 2, 1])
            k_2 = None
            q_2 = torch.ops.aten.permute.default(q_1, [0, 2, 1])
            q_1 = None
            t_k3 = torch.ops.aten.permute.default(t_k2, [0, 2, 1])
            t_k2 = None
            # return (y, t_y, a_1, b, g, k_2, f_t, m_t, t_v2, o_t2, v_2, t_q3, k_3, q_2, t_k3)
            ctx.save_for_backward(
                a_1, b, g, k_2, f_t, m_t, t_v2, o_t2, v_2, t_q3, k_3, q_2, t_k3
            )
            return y, t_y

        @staticmethod
        def backward(ctx, d_y, d_t_y):
            (
                a_1,
                b,
                g,
                k_2,
                f_t,
                m_t,
                t_v2,
                o_t2,
                v_2,
                t_q3,
                k_3,
                q_2,
                t_k3,
            ) = ctx.saved_for_backward
            grads = SDPAJVPBackwardFunction.apply(
                # grads = SDPAJVPFlashBackwardFunction.apply(
                a_1,
                b,
                g,
                k_2,
                f_t,
                m_t,
                t_v2,
                o_t2,
                v_2,
                t_q3,
                k_3,
                q_2,
                t_k3,
                d_y,
                d_t_y,
            )
            return grads

    class SDPAJVPBackwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            a_1,
            b,
            g,
            k_2,
            f_t,
            m_t,
            t_v2,
            o_t2,
            v_2,
            t_q3,
            k_3,
            q_2,
            t_k3,
            d_y,
            d_t_y,
        ):
            expand_2 = torch.ops.aten.expand.default(b, [1, 16, 16])
            permute_3 = torch.ops.aten.permute.default(expand_2, [0, 2, 1])
            expand_2 = None
            grad_tv = torch.ops.aten.bmm.default(permute_3, d_t_y)
            bmm_7 = torch.ops.aten.bmm.default(d_t_y, t_v2)
            t_v2 = None
            bmm_8 = torch.ops.aten.bmm.default(o_t2, d_t_y)
            o_t2 = None
            bmm_9 = torch.ops.aten.bmm.default(d_t_y, v_2)
            d_t_y = None
            sub_3 = torch.ops.aten.sub.Tensor(f_t, m_t)
            mul_3 = torch.ops.aten.mul.Tensor(bmm_9, sub_3)
            sub_3 = None
            mul_4 = torch.ops.aten.mul.Tensor(bmm_9, b)
            bmm_9 = None
            add_2 = torch.ops.aten.add.Tensor(bmm_7, mul_3)
            bmm_7 = mul_3 = None
            neg = torch.ops.aten.neg.default(mul_4)
            sum_4 = torch.ops.aten.sum.dim_IntList(neg, [2], True)
            neg = None
            div_5 = torch.ops.aten.div.Tensor(m_t, k_2)
            m_t = None
            neg_1 = torch.ops.aten.neg.default(sum_4)
            mul_5 = torch.ops.aten.mul.Tensor(neg_1, div_5)
            neg_1 = div_5 = None
            div_6 = torch.ops.aten.div.Tensor(sum_4, k_2)
            sum_4 = k_2 = None
            expand_12 = torch.ops.aten.expand.default(div_6, [1, 16, 16])
            div_6 = None
            div = torch.ops.aten.div.Tensor(a_1, 11.313708498984761)
            a_1 = None
            sub_1 = torch.ops.aten.sub.Tensor(div, g)
            exp_1 = torch.ops.aten.exp.default(sub_1)
            sub_1 = None
            mul_6 = torch.ops.aten.mul.Tensor(expand_12, exp_1)
            mul_7 = torch.ops.aten.mul.Tensor(expand_12, f_t)
            expand_12 = f_t = None
            add_3 = torch.ops.aten.add.Tensor(mul_4, mul_6)
            mul_4 = mul_6 = None
            div_7 = torch.ops.aten.div.Tensor(add_3, 11.313708498984761)
            add_3 = None
            neg_2 = torch.ops.aten.neg.default(div_7)
            bmm_10 = torch.ops.aten.bmm.default(t_q3, div_7)
            t_q3 = None
            grad_tq = torch.ops.aten.bmm.default(div_7, k_3)
            permute_9 = torch.ops.aten.permute.default(bmm_10, [0, 2, 1])
            bmm_10 = None
            bmm_12 = torch.ops.aten.bmm.default(q_2, div_7)
            bmm_13 = torch.ops.aten.bmm.default(div_7, t_k3)
            div_7 = t_k3 = None
            grad_tk = torch.ops.aten.permute.default(bmm_12, [0, 2, 1])
            bmm_12 = None
            expand_13 = torch.ops.aten.expand.default(mul_5, [1, 16, 16])
            mul_5 = None
            add_4 = torch.ops.aten.add.Tensor(mul_7, expand_13)
            mul_7 = expand_13 = None
            mul_8 = torch.ops.aten.mul.Tensor(add_4, exp_1)
            add_4 = exp_1 = None
            neg_3 = torch.ops.aten.neg.default(mul_8)
            sum_5 = torch.ops.aten.sum.dim_IntList(neg_3, [2], True)
            neg_3 = None
            eq = torch.ops.aten.eq.Tensor(g, div)
            g = div = None
            sum_6 = torch.ops.aten.sum.dim_IntList(eq, [-1], True)
            div_8 = torch.ops.aten.div.Tensor(sum_5, sum_6)
            sum_5 = sum_6 = None
            mul_9 = torch.ops.aten.mul.Tensor(div_8, eq)
            div_8 = eq = None
            add_5 = torch.ops.aten.add.Tensor(mul_8, mul_9)
            mul_8 = mul_9 = None
            mul_10 = torch.ops.aten.mul.Tensor(neg_2, 0)
            neg_2 = None
            add_6 = torch.ops.aten.add.Tensor(add_5, mul_10)
            add_5 = mul_10 = None
            bmm_14 = torch.ops.aten.bmm.default(permute_3, d_y)
            permute_3 = None
            bmm_15 = torch.ops.aten.bmm.default(d_y, v_2)
            d_y = v_2 = None
            grad_v = torch.ops.aten.add.Tensor(bmm_8, bmm_14)
            bmm_8 = bmm_14 = None
            add_8 = torch.ops.aten.add.Tensor(add_2, bmm_15)
            add_2 = bmm_15 = None
            mul_11 = torch.ops.aten.mul.Tensor(add_8, b)
            add_8 = None
            sum_7 = torch.ops.aten.sum.dim_IntList(mul_11, [-1], True)
            neg_4 = torch.ops.aten.neg.default(b)
            b = None
            fma = torch.ops.prims.fma.default(neg_4, sum_7, mul_11)
            neg_4 = sum_7 = mul_11 = None
            add_9 = torch.ops.aten.add.Tensor(add_6, fma)
            add_6 = fma = None
            div_9 = torch.ops.aten.div.Tensor(add_9, 11.313708498984761)
            add_9 = None
            bmm_16 = torch.ops.aten.bmm.default(q_2, div_9)
            q_2 = None
            bmm_17 = torch.ops.aten.bmm.default(div_9, k_3)
            div_9 = k_3 = None
            grad_q = torch.ops.aten.add.Tensor(bmm_13, bmm_17)
            bmm_13 = bmm_17 = None
            permute_17 = torch.ops.aten.permute.default(bmm_16, [0, 2, 1])
            bmm_16 = None
            grad_k = torch.ops.aten.add.Tensor(permute_9, permute_17)
            permute_9 = permute_17 = None
            return (grad_q, grad_k, grad_v, grad_tk, grad_tq, grad_tv)

        @staticmethod
        def backward(ctx, d_q, d_k, d_v, d_t_q, d_t_k, d_t_v):
            raise NotImplementedError("Not implemented")

    torch.manual_seed(123)
    device = torch.device("cuda")
    # b = 1
    # s = 128
    # h = 16
    # d = 128
    # q, k, v = torch.randn((3, b, h, s, d), device=device)
    q, k, v = torch.randn((3, 1, 16, 128), device=device)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    t_q, t_k, t_v = (
        torch.randn_like(q, requires_grad=True),
        torch.randn_like(k, requires_grad=True),
        torch.randn_like(v, requires_grad=True),
    )
    y, t_y = SDPAJVPForwardFunction.apply(q, k, v, t_q, t_k, t_v)
    t_y.sum().backward()
    print(y)
    print(q.grad)
    print(k.grad)


def test_flattened_autograd2():
    class SDPAJVPForwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, _q, _k, _v, t_q, t_k, t_v):
            import math

            d_scale = math.sqrt(_q.shape[-1])
            # _k1 = torch.ops.aten.permute.default(_k, [0, 1, 3, 2]);  _k = None
            # _q1 = torch.ops.aten.expand.default(_q, [1, 16, 128, 128]);  _q = None
            # _q2 = torch.ops.aten.view.default(_q1, [16, 128, 128]);  _q1 = None
            # _k2 = torch.ops.aten.expand.default(_k1, [1, 16, 128, 128]);  _k1 = None
            # _k3 = torch.ops.aten.view.default(_k2, [16, 128, 128]);  _k2 = None
            # a_1 = torch.ops.aten.bmm.default(_q2, _k3)
            _q_orig_shape = _q.shape
            _q_3d = _q.view(-1, *_q.shape[-2:])
            _q = None
            _k_t = _k.transpose(2, 3)
            _k = None
            _k_t_3d = _k_t.view(-1, *_k_t.shape[-2:])
            _k_t = None
            a_1 = torch.bmm(_q_3d, _k_t_3d)
            a_2 = a_1.view(*_q_orig_shape[:-2], *a_1.shape[-2:])
            # a_2 = torch.ops.aten.view.default(a_1, [1, 16, 128, 128])
            a = torch.ops.aten.div.Tensor(a_2, d_scale)
            a_3 = torch.ops.aten.mul.Tensor(a_2, 1)
            a_2 = None
            amax_sm1 = torch.ops.aten.amax.default(a_3, [-1], True)
            sub_sm1 = torch.ops.aten.sub.Tensor(a_3, amax_sm1)
            a_3 = amax_sm1 = None
            div_sm1 = torch.ops.aten.div.Tensor(sub_sm1, d_scale)
            sub_sm1 = None
            exp_sm1 = torch.ops.aten.exp.default(div_sm1)
            div_sm1 = None
            sum_sm1 = torch.ops.aten.sum.dim_IntList(exp_sm1, [-1], True)
            b = torch.ops.aten.div.Tensor(exp_sm1, sum_sm1)
            exp_sm1 = sum_sm1 = None
            # b1 = torch.ops.aten.expand.default(b, [1, 16, 128, 128])
            # b2 = torch.ops.aten.view.default(b1, [16, 128, 128]);  b1 = None
            b_3d = b.view(-1, *b.shape[-2:])
            # _v1 = torch.ops.aten.expand.default(_v, [1, 16, 128, 128]);  _v = None
            # _v2 = torch.ops.aten.view.default(_v1, [16, 128, 128]);  _v1 = None
            _v_3d = _v.view(-1, *_v.shape[-2:])
            _v = None
            y = torch.bmm(b_3d, _v_3d)
            y1 = y.view(*_q_orig_shape[:-2], *y.shape[-2:])
            _y = None
            # y1 = torch.ops.aten.view.default(y, [1, 16, 128, 128]);  y = None
            d = torch.ops.aten.mul.Tensor(a, 0)
            g = torch.ops.aten.amax.default(a, [-1], True)
            h = torch.ops.aten.sub.Tensor(a, g)
            a = None
            i = torch.ops.aten.exp.default(h)
            h = None
            k = torch.ops.aten.sum.dim_IntList(i, [-1], True)
            # t_k1 = torch.ops.aten.permute.default(t_k, [0, 1, 3, 2]);  t_k = None
            # t_k2 = torch.ops.aten.expand.default(t_k1, [1, 16, 128, 128]);  t_k1 = None
            # t_k3 = torch.ops.aten.view.default(t_k2, [16, 128, 128]);  t_k2 = None
            t_k_t_3d = t_k.transpose(2, 3)
            t_k = None
            t_k_t_3d = t_k_t_3d.reshape(-1, *t_k_t_3d.shape[-2:])
            q_tk = torch.ops.aten.bmm.default(_q_3d, t_k_t_3d)
            q_tk1 = q_tk.view(*_q_orig_shape[:-2], *q_tk.shape[-2:])
            q_tk = None
            # q_tk1 = torch.ops.aten.view.default(q_tk, [1, 16, 128, 128]);  q_tk = None
            # t_q1 = torch.ops.aten.expand.default(t_q, [1, 16, 128, 128]);  t_q = None
            # t_q2 = torch.ops.aten.view.default(t_q1, [16, 128, 128]);  t_q1 = None
            t_q_3d = t_q.view(-1, *t_q.shape[-2:])
            t_q = None
            k_tq = torch.ops.aten.bmm.default(t_q_3d, _k_t_3d)
            k_tq1 = k_tq.view(*_q_orig_shape[:-2], *k_tq.shape[-2:])
            # k_tq1 = torch.ops.aten.view.default(k_tq, [1, 16, 128, 128]);  k_tq = None
            c_t = torch.ops.aten.add.Tensor(q_tk1, k_tq1)
            q_tk1 = k_tq1 = None
            e_t = torch.ops.aten.sub.Tensor(c_t, d)
            c_t = d = None
            f_t = torch.ops.aten.div.Tensor(e_t, d_scale)
            e_t = None
            j_t = torch.ops.aten.mul.Tensor(i, f_t)
            i = None
            l_t = torch.ops.aten.sum.dim_IntList(j_t, [-1], True)
            j_t = None
            m_t = torch.ops.aten.div.Tensor(l_t, k)
            l_t = None
            n_t = torch.ops.aten.sub.Tensor(f_t, m_t)
            o_t = torch.ops.aten.mul.Tensor(n_t, b)
            n_t = None
            o_t_3d = o_t.view(-1, *o_t.shape[-2:])
            o_t = None
            # o_t1 = torch.ops.aten.expand.default(o_t, [1, 16, 128, 128]);  o_t = None
            # o_t2 = torch.ops.aten.view.default(o_t1, [16, 128, 128]);  o_t1 = None
            p_t = torch.ops.aten.bmm.default(o_t_3d, _v_3d)
            p_t1 = p_t.view(*_q_orig_shape[:-2], *p_t.shape[-2:])
            p_t = None
            # p_t1 = torch.ops.aten.view.default(p_t, [1, 16, 128, 128]);  p_t = None
            # t_v1 = torch.ops.aten.expand.default(t_v, [1, 16, 128, 128]);  t_v = None
            # t_v2 = torch.ops.aten.view.default(t_v1, [16, 128, 128]);  t_v1 = None
            t_v_3d = t_v.view(-1, *t_v.shape[-2:])
            t_v = None
            q_t = torch.ops.aten.bmm.default(b_3d, t_v_3d)
            b_3d = None
            # q_T1 = torch.ops.aten.view.default(q_t, [1, 16, 128, 128]);  q_t = None
            q_T1 = q_t.view(*_q_orig_shape[:-2], *q_t.shape[-2:])
            q_t = None
            t_y = torch.ops.aten.add.Tensor(p_t1, q_T1)
            p_t1 = q_T1 = None
            # t_v3 = torch.ops.aten.permute.default(t_v_3d, [0, 2, 1]);  t_v_3d = None
            # o_t3 = torch.ops.aten.permute.default(o_t_3d, [0, 2, 1]);  o_t_3d = None
            # _v3 = torch.ops.aten.permute.default(_v_3d, [0, 2, 1]);  _v2 = None
            # t_q3 = torch.ops.aten.permute.default(t_q_3d, [0, 2, 1]);  t_q_3d = None
            # _k4 = torch.ops.aten.permute.default(_k_t_3d, [0, 2, 1]);  _k3 = None
            # _q3 = torch.ops.aten.permute.default(_q_3d, [0, 2, 1]);  _q2 = None
            # t_k4 = torch.ops.aten.permute.default(t_k_t_3d, [0, 2, 1]);  t_k_t_3d = None
            t_v3 = t_v_3d.transpose(-1, -2)
            t_v_3d = None
            o_t3 = o_t_3d.transpose(-1, -2)
            o_t_3d = None
            _v3 = _v_3d.transpose(-1, -2)
            _v_3d = None
            t_q3 = t_q_3d.transpose(-1, -2)
            t_q_3d = None
            _k4 = _k_t_3d.transpose(-1, -2)
            _k_t_3d = None
            _q3 = _q_3d.transpose(-1, -2)
            _q_3d = None
            t_k4 = t_k_t_3d.transpose(-1, -2)
            t_k_t_3d = None
            ctx.save_for_backward(
                a_1, b, g, k, f_t, m_t, t_v3, o_t3, _v3, t_q3, _k4, _q3, t_k4
            )
            return y1, t_y

        @staticmethod
        def backward(ctx, d_y, d_t_y):
            (
                a_1,
                b,
                g,
                k,
                f_t,
                m_t,
                t_v3,
                o_t3,
                _v3,
                t_q3,
                _k4,
                _q3,
                t_k4,
            ) = ctx.saved_tensors
            grads = SDPAJVPBackwardFunction.apply(
                a_1,
                b,
                g,
                k,
                f_t,
                m_t,
                t_v3,
                o_t3,
                _v3,
                t_q3,
                _k4,
                _q3,
                t_k4,
                d_y,
                d_t_y,
            )
            return grads

    class SDPAJVPBackwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            self,
            bmm,
            div_1,
            amax_1,
            sum_2,
            div_2,
            div_3,
            permute_4,
            permute_5,
            permute_6,
            permute_7,
            permute_8,
            permute_10,
            permute_11,
            tangents_1,
            tangents_2,
        ):
            # CONTINUE HERE:
            # CONTINUE HERE:
            # CONTINUE HERE:
            # CONTINUE HERE:

            # TODO: continue here -> make all operations dynamic instead of with hardcoded shapes
            # TODO: continue here -> make all operations dynamic instead of with hardcoded shapes
            # TODO: continue here -> make all operations dynamic instead of with hardcoded shapes
            # TODO: continue here -> make all operations dynamic instead of with hardcoded shapes
            # TODO: continue here -> make all operations dynamic instead of with hardcoded shapes

            # print("bmm", bmm.shape)
            # print("div_1", div_1.shape)
            # print("amax_1", amax_1.shape)
            # print("sum_2", sum_2.shape)
            # print("div_2", div_2.shape)
            # print("div_3", div_3.shape)
            # print("permute_4", permute_4.shape)
            # print("permute_5", permute_5.shape)
            # print("permute_6", permute_6.shape)
            # print("permute_7", permute_7.shape)
            # print("permute_8", permute_8.shape)
            # print("permute_10", permute_10.shape)
            # print("permute_11", permute_11.shape)
            # print("tangents_1", tangents_1.shape)
            # print("tangents_2", tangents_2.shape)

            print(f"tangents_1: {tangents_1}")
            print(f"tangents_2: {tangents_2}")

            orig_shape = tangents_2.shape
            d_scale = math.sqrt(orig_shape[-1])

            # view_18 = torch.ops.aten.view.default(tangents_2, [16, 128, 128]);  tangents_2 = None
            view_18 = tangents_2.view(-1, *tangents_2.shape[-2:])
            tangents_2 = None
            # expand_2 = torch.ops.aten.expand.default(div_1, [1, 16, 128, 128])
            # view_3 = torch.ops.aten.view.default(expand_2, [16, 128, 128]);  expand_2 = None
            view_3 = div_1.view(-1, *div_1.shape[-2:])  # ; div_1 = None

            # permute_3 = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
            # bmm_6 = torch.ops.aten.bmm.default(permute_3, view_18)
            # bmm_7 = torch.ops.aten.bmm.default(view_18, permute_4);  permute_4 = None

            permute_3 = view_3.transpose(-1, -2)
            view_3 = None
            bmm_6 = torch.bmm(permute_3, view_18)
            bmm_7 = torch.bmm(view_18, permute_4)
            permute_4 = None

            # grad_t_v = torch.ops.aten.view.default(bmm_6, [1, 16, 128, 128]);  bmm_6 = None
            # view_20 = torch.ops.aten.view.default(bmm_7, [1, 16, 128, 128]);  bmm_7 = None
            # bmm_8 = torch.ops.aten.bmm.default(permute_5, view_18);  permute_5 = None
            # bmm_9 = torch.ops.aten.bmm.default(view_18, permute_6);  view_18 = None
            # view_22 = torch.ops.aten.view.default(bmm_8, [1, 16, 128, 128]);  bmm_8 = None
            # view_23 = torch.ops.aten.view.default(bmm_9, [1, 16, 128, 128]);  bmm_9 = None

            grad_t_v = bmm_6.view(*orig_shape[:-2], *bmm_6.shape[-2:])
            bmm_6 = None
            view_20 = bmm_7.view(*orig_shape[:-2], *bmm_7.shape[-2:])
            bmm_7 = None
            bmm_8 = torch.bmm(permute_5, view_18)
            permute_5 = None
            bmm_9 = torch.bmm(view_18, permute_6)
            view_18 = None
            view_22 = bmm_8.view(*orig_shape[:-2], *bmm_8.shape[-2:])
            bmm_8 = None
            view_23 = bmm_9.view(*orig_shape[:-2], *bmm_9.shape[-2:])
            bmm_9 = None

            sub_3 = torch.ops.aten.sub.Tensor(div_2, div_3)
            mul_3 = torch.ops.aten.mul.Tensor(view_23, sub_3)
            sub_3 = None
            mul_4 = torch.ops.aten.mul.Tensor(view_23, div_1)
            view_23 = None
            add_2 = torch.ops.aten.add.Tensor(view_20, mul_3)
            view_20 = mul_3 = None
            neg = torch.ops.aten.neg.default(mul_4)
            sum_4 = torch.ops.aten.sum.dim_IntList(neg, [3], True)
            neg = None
            div_5 = torch.ops.aten.div.Tensor(div_3, sum_2)
            div_3 = None
            neg_1 = torch.ops.aten.neg.default(sum_4)
            mul_5 = torch.ops.aten.mul.Tensor(neg_1, div_5)
            neg_1 = div_5 = None
            div_6 = torch.ops.aten.div.Tensor(sum_4, sum_2)
            sum_4 = sum_2 = None

            # expand_12 = torch.ops.aten.expand.default(div_6, [1, 16, 128, 128]);  div_6 = None
            # view_2 = torch.ops.aten.view.default(bmm, [1, 16, 128, 128]);  bmm = None
            # div = torch.ops.aten.div.Tensor(view_2, 11.313708498984761);  view_2 = None

            expand_12 = div_6.view(*orig_shape[:-2], *div_6.shape[-2:])
            div_6 = None
            view_2 = bmm.view(*orig_shape[:-2], *bmm.shape[-2:])
            bmm = None
            div = view_2 / d_scale
            view_2 = None

            sub_1 = torch.ops.aten.sub.Tensor(div, amax_1)
            exp_1 = torch.ops.aten.exp.default(sub_1)
            sub_1 = None
            mul_6 = torch.ops.aten.mul.Tensor(expand_12, exp_1)
            mul_7 = torch.ops.aten.mul.Tensor(expand_12, div_2)
            expand_12 = div_2 = None
            add_3 = torch.ops.aten.add.Tensor(mul_4, mul_6)
            mul_4 = mul_6 = None

            # div_7 = torch.ops.aten.div.Tensor(add_3, 11.313708498984761);  add_3 = None
            div_7 = torch.ops.aten.div.Tensor(add_3, d_scale)
            add_3 = None

            neg_2 = torch.ops.aten.neg.default(div_7)
            # view_24 = torch.ops.aten.view.default(div_7, [16, 128, 128]);  div_7 = None
            view_24 = div_7.view(-1, *div_7.shape[-2:])
            div_7 = None
            bmm_10 = torch.ops.aten.bmm.default(permute_7, view_24)
            permute_7 = None
            bmm_11 = torch.ops.aten.bmm.default(view_24, permute_8)

            # view_25 = torch.ops.aten.view.default(bmm_10, [1, 16, 128, 128]);  bmm_10 = None
            # grad_t_q = torch.ops.aten.view.default(bmm_11, [1, 16, 128, 128]);  bmm_11 = None
            # permute_9 = torch.ops.aten.permute.default(view_25, [0, 1, 3, 2]);  view_25 = None
            view_25 = bmm_10.view(*orig_shape[:-2], *bmm_10.shape[-2:])
            bmm_10 = None
            grad_t_q = bmm_11.view(*orig_shape[:-2], *bmm_11.shape[-2:])
            bmm_11 = None
            permute_9 = view_25.transpose(-1, -2)
            view_25 = None

            bmm_12 = torch.ops.aten.bmm.default(permute_10, view_24)
            bmm_13 = torch.ops.aten.bmm.default(view_24, permute_11)
            view_24 = permute_11 = None

            # view_28 = torch.ops.aten.view.default(bmm_12, [1, 16, 128, 128]);  bmm_12 = None
            # view_29 = torch.ops.aten.view.default(bmm_13, [1, 16, 128, 128]);  bmm_13 = None
            # grad_t_k = torch.ops.aten.permute.default(view_28, [0, 1, 3, 2]);  view_28 = None
            # expand_13 = torch.ops.aten.expand.default(mul_5, [1, 16, 128, 128]);  mul_5 = None

            view_28 = bmm_12.view(*orig_shape[:-2], *bmm_12.shape[-2:])
            bmm_12 = None
            view_29 = bmm_13.view(*orig_shape[:-2], *bmm_13.shape[-2:])
            bmm_13 = None
            grad_t_k = view_28.transpose(-1, -2)
            view_28 = None

            # add_4 = torch.ops.aten.add.Tensor(mul_7, expand_13);  mul_7 = expand_13 = None
            add_4 = torch.ops.aten.add.Tensor(mul_7, mul_5)
            mul_7 = mul_5 = None
            mul_8 = torch.ops.aten.mul.Tensor(add_4, exp_1)
            add_4 = exp_1 = None
            neg_3 = torch.ops.aten.neg.default(mul_8)
            sum_5 = torch.ops.aten.sum.dim_IntList(neg_3, [3], True)
            neg_3 = None
            eq = torch.ops.aten.eq.Tensor(amax_1, div)  # ;  amax_1 = div = None
            sum_6 = torch.ops.aten.sum.dim_IntList(eq, [-1], True)
            div_8 = torch.ops.aten.div.Tensor(sum_5, sum_6)
            sum_5 = sum_6 = None
            mul_9 = torch.ops.aten.mul.Tensor(div_8, eq)
            div_8 = eq = None
            add_5 = torch.ops.aten.add.Tensor(mul_8, mul_9)
            mul_8 = mul_9 = None
            mul_10 = torch.ops.aten.mul.Tensor(neg_2, 0)
            neg_2 = None
            add_6 = torch.ops.aten.add.Tensor(add_5, mul_10)
            add_5 = mul_10 = None
            # view_30 = torch.ops.aten.view.default(tangents_1, [16, 128, 128]);  tangents_1 = None
            view_30 = tangents_1.view(-1, *tangents_1.shape[-2:])
            tangents_1 = None

            bmm_14 = torch.ops.aten.bmm.default(permute_3, view_30)
            permute_3 = None
            bmm_15 = torch.ops.aten.bmm.default(view_30, permute_6)
            view_30 = permute_6 = None

            # view_31 = torch.ops.aten.view.default(bmm_14, [1, 16, 128, 128]);  bmm_14 = None
            view_31 = bmm_14.view(*orig_shape[:-2], *bmm_14.shape[-2:])
            bmm_14 = None

            grad_v = torch.ops.aten.add.Tensor(view_22, view_31)
            view_22 = view_31 = None

            # view_32 = torch.ops.aten.view.default(bmm_15, [1, 16, 128, 128]);  bmm_15 = None
            view_32 = bmm_15.view(*orig_shape[:-2], *bmm_15.shape[-2:])
            bmm_15 = None

            add_8 = torch.ops.aten.add.Tensor(add_2, view_32)
            add_2 = view_32 = None
            mul_11 = torch.ops.aten.mul.Tensor(add_8, div_1)
            add_8 = None
            sum_7 = torch.ops.aten.sum.dim_IntList(mul_11, [-1], True)
            neg_4 = torch.ops.aten.neg.default(div_1)
            div_1 = None
            # fma = torch.ops.prims.fma.default(neg_4, sum_7, mul_11);  neg_4 = sum_7 = mul_11 = None
            fma = torch.addcmul(mul_11, neg_4, sum_7)
            neg_4 = sum_7 = mul_11 = None
            add_9 = torch.ops.aten.add.Tensor(add_6, fma)
            add_6 = fma = None

            # div_9 = torch.ops.aten.div.Tensor(add_9, 11.313708498984761);  add_9 = None
            # view_33 = torch.ops.aten.view.default(div_9, [16, 128, 128]);  div_9 = None
            div_9 = add_9 / d_scale
            add_9 = None
            view_33 = div_9.view(-1, *div_9.shape[-2:])
            div_9 = None

            bmm_16 = torch.ops.aten.bmm.default(permute_10, view_33)
            permute_10 = None
            bmm_17 = torch.ops.aten.bmm.default(view_33, permute_8)
            view_33 = permute_8 = None

            # view_34 = torch.ops.aten.view.default(bmm_16, [1, 16, 128, 128]);  bmm_16 = None
            # view_35 = torch.ops.aten.view.default(bmm_17, [1, 16, 128, 128]);  bmm_17 = None
            view_34 = bmm_16.view(*orig_shape[:-2], *bmm_16.shape[-2:])
            bmm_16 = None
            view_35 = bmm_17.view(*orig_shape[:-2], *bmm_17.shape[-2:])
            bmm_17 = None

            grad_q = torch.ops.aten.add.Tensor(view_29, view_35)
            view_29 = view_35 = None

            # permute_17 = torch.ops.aten.permute.default(view_34, [0, 1, 3, 2]);  view_34 = None
            permute_17 = view_34.transpose(-1, -2)
            view_34 = None

            grad_k = torch.ops.aten.add.Tensor(permute_9, permute_17)
            permute_9 = permute_17 = None
            return (grad_q, grad_k, grad_v, grad_t_q, grad_t_k, grad_t_v)

        @staticmethod
        def backward(ctx, d_q, d_k, d_v, d_t_q, d_t_k, d_t_v):
            raise NotImplementedError("Not implemented")

    torch.manual_seed(123)
    b = 1 + 1
    s = 128 * 2
    h = 16 + 8
    d = 128 + 16
    device = torch.device("cuda")
    q, k, v = torch.randn((3, b, h, s, d), device=device)
    # q, k, v = torch.randn((3, 1, 16, 128), device=device)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    # print(torch.cuda.memory_summary(device=device, abbreviated=True))
    t_q, t_k, t_v = (
        torch.randn_like(q, requires_grad=True),
        torch.randn_like(k, requires_grad=True),
        torch.randn_like(v, requires_grad=True),
    )
    y, t_y = SDPAJVPForwardFunction.apply(q, k, v, t_q, t_k, t_v)
    t_y.sum().backward()
    # print(torch.cuda.memory_summary(device=device, abbreviated=True))
    # print(y)
    # print(q.grad)
    # print(k.grad)

    torch.manual_seed(123)
    q_ref, k_ref, v_ref = torch.randn((3, b, h, s, d), device=device)
    q_ref.requires_grad_(True)
    k_ref.requires_grad_(True)
    v_ref.requires_grad_(True)
    t_q_ref, t_k_ref, t_v_ref = (
        torch.randn_like(q_ref, requires_grad=True),
        torch.randn_like(k_ref, requires_grad=True),
        torch.randn_like(v_ref, requires_grad=True),
    )
    model = VanillaAttention()
    y_ref, t_y_ref = torch.func.jvp(
        model, (q_ref, k_ref, v_ref), (t_q_ref, t_k_ref, t_v_ref)
    )
    t_y_ref.sum().backward()
    # print(y_ref)
    # print(q_ref.grad)
    # print(k_ref.grad)
    # print(v_ref.grad)

    y_err = (y - y_ref).abs().mean()
    t_y_err = (t_y - t_y_ref).abs().mean()
    print(f"y_err: {y_err}, t_y_err: {t_y_err}")

    q_err = (q_ref.grad - q.grad).abs().mean()
    k_err = (k_ref.grad - k.grad).abs().mean()
    v_err = (v_ref.grad - v.grad).abs().mean()
    t_q_err = (t_q_ref.grad - t_q.grad).abs().mean()
    t_k_err = (t_k_ref.grad - t_k.grad).abs().mean()
    t_v_err = (t_v_ref.grad - t_v.grad).abs().max()
    print(f"q_err: {q_err}, k_err: {k_err}, v_err: {v_err}")
    print(f"t_q_err: {t_q_err}, t_k_err: {t_k_err}, t_v_err: {t_v_err}")
    print("")


def test_flattened_autograd3():
    class SDPAJVPForwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(_q, _k, _v, t_q, t_k, t_v):
            import math

            d_scale = math.sqrt(_q.shape[-1])
            _q_orig_shape = _q.shape
            _q_3d = _q.view(-1, *_q.shape[-2:])
            _q = None
            _k_t = _k.transpose(2, 3)
            _k = None
            _k_t_3d = _k_t.view(-1, *_k_t.shape[-2:])
            _k_t = None
            a_1 = torch.bmm(_q_3d, _k_t_3d)
            a_2 = a_1.view(*_q_orig_shape[:-2], *a_1.shape[-2:])
            a = torch.ops.aten.div.Tensor(a_2, d_scale)
            a_3 = torch.ops.aten.mul.Tensor(a_2, 1)
            a_2 = None
            amax_sm1 = torch.ops.aten.amax.default(a_3, [-1], True)
            sub_sm1 = torch.ops.aten.sub.Tensor(a_3, amax_sm1)
            a_3 = amax_sm1 = None
            div_sm1 = torch.ops.aten.div.Tensor(sub_sm1, d_scale)
            sub_sm1 = None
            exp_sm1 = torch.ops.aten.exp.default(div_sm1)
            div_sm1 = None
            sum_sm1 = torch.ops.aten.sum.dim_IntList(exp_sm1, [-1], True)
            b = torch.ops.aten.div.Tensor(exp_sm1, sum_sm1)
            exp_sm1 = sum_sm1 = None

            b_3d = b.view(-1, *b.shape[-2:])
            _v_3d = _v.view(-1, *_v.shape[-2:])
            _v = None

            d = torch.ops.aten.mul.Tensor(a, 0)
            g = torch.ops.aten.amax.default(a, [-1], True)
            h = torch.ops.aten.sub.Tensor(a, g)
            a = None
            i = torch.ops.aten.exp.default(h)
            h = None
            k = torch.ops.aten.sum.dim_IntList(i, [-1], True)
            t_k_t_3d = t_k.transpose(2, 3)
            t_k = None
            t_k_t_3d = t_k_t_3d.reshape(-1, *t_k_t_3d.shape[-2:])
            q_tk = torch.ops.aten.bmm.default(_q_3d, t_k_t_3d)
            q_tk1 = q_tk.view(*_q_orig_shape[:-2], *q_tk.shape[-2:])
            q_tk = None
            t_q_3d = t_q.view(-1, *t_q.shape[-2:])
            t_q = None
            k_tq = torch.ops.aten.bmm.default(t_q_3d, _k_t_3d)
            k_tq1 = k_tq.view(*_q_orig_shape[:-2], *k_tq.shape[-2:])
            c_t = torch.ops.aten.add.Tensor(q_tk1, k_tq1)
            q_tk1 = k_tq1 = None
            e_t = torch.ops.aten.sub.Tensor(c_t, d)
            c_t = d = None
            f_t = torch.ops.aten.div.Tensor(e_t, d_scale)
            e_t = None
            j_t = torch.ops.aten.mul.Tensor(i, f_t)
            i = None
            l_t = torch.ops.aten.sum.dim_IntList(j_t, [-1], True)
            j_t = None
            m_t = torch.ops.aten.div.Tensor(l_t, k)
            l_t = None
            n_t = torch.ops.aten.sub.Tensor(f_t, m_t)
            o_t = torch.ops.aten.mul.Tensor(n_t, b)
            n_t = None
            o_t_3d = o_t.view(-1, *o_t.shape[-2:])
            o_t = None
            p_t = torch.ops.aten.bmm.default(o_t_3d, _v_3d)

            p_t1 = p_t.view(*_q_orig_shape[:-2], *p_t.shape[-2:])
            p_t = None
            t_v_3d = t_v.view(-1, *t_v.shape[-2:])
            t_v = None
            q_t = torch.ops.aten.bmm.default(b_3d, t_v_3d)
            b_3d = None
            q_T1 = q_t.view(*_q_orig_shape[:-2], *q_t.shape[-2:])
            q_t = None
            t_y = torch.ops.aten.add.Tensor(p_t1, q_T1)
            p_t1 = q_T1 = None
            t_v3 = t_v_3d.transpose(-1, -2)
            t_v_3d = None
            o_t3 = o_t_3d.transpose(-1, -2)
            o_t_3d = None
            _v3 = _v_3d.transpose(-1, -2)
            _v_3d = None
            t_q3 = t_q_3d.transpose(-1, -2)
            t_q_3d = None
            _k4 = _k_t_3d.transpose(-1, -2)
            _k_t_3d = None
            _q3 = _q_3d.transpose(-1, -2)
            _q_3d = None
            t_k4 = t_k_t_3d.transpose(-1, -2)
            t_k_t_3d = None
            # ctx.save_for_backward(a_1, b, g, k, f_t, m_t, t_v3, o_t3, _v3, t_q3, _k4, _q3, t_k4)
            # return t_y
            return t_y, a_1, b, g, k, f_t, m_t, t_v3, o_t3, _v3, t_q3, _k4, _q3, t_k4

        # @staticmethod
        def setup_context(ctx, inputs: tuple[Any, ...], output: Any) -> Any:
            # ctx.save_for_backward(*inputs, *output[1:])
            ctx.save_for_backward(*output[1:])

        @staticmethod
        def backward(ctx, d_t_y, *args):
            # print(args)
            (
                a_1,
                b,
                g,
                k,
                f_t,
                m_t,
                t_v3,
                o_t3,
                _v3,
                t_q3,
                _k4,
                _q3,
                t_k4,
            ) = ctx.saved_tensors
            grads = SDPAJVPBackwardFunction.apply(
                a_1, b, g, k, f_t, m_t, t_v3, o_t3, _v3, t_q3, _k4, _q3, t_k4, d_t_y
            )
            return grads

    class SDPAJVPBackwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            bmm,
            div_1,
            amax_1,
            sum_2,
            div_2,
            div_3,
            permute_4,
            permute_5,
            permute_6,
            permute_7,
            permute_8,
            permute_10,
            permute_11,
            tangents_2,
        ):
            orig_shape = tangents_2.shape
            d_scale = math.sqrt(orig_shape[-1])

            view_18 = tangents_2.view(-1, *tangents_2.shape[-2:])
            tangents_2 = None
            view_3 = div_1.view(-1, *div_1.shape[-2:])  # ; div_1 = None

            permute_3 = view_3.transpose(-1, -2)
            view_3 = None
            bmm_6 = torch.bmm(permute_3, view_18)
            bmm_7 = torch.bmm(view_18, permute_4)
            permute_4 = None

            grad_t_v = bmm_6.view(*orig_shape[:-2], *bmm_6.shape[-2:])
            bmm_6 = None
            view_20 = bmm_7.view(*orig_shape[:-2], *bmm_7.shape[-2:])
            bmm_7 = None
            bmm_8 = torch.bmm(permute_5, view_18)
            permute_5 = None
            bmm_9 = torch.bmm(view_18, permute_6)
            view_18 = None
            view_22 = bmm_8.view(*orig_shape[:-2], *bmm_8.shape[-2:])
            bmm_8 = None
            view_23 = bmm_9.view(*orig_shape[:-2], *bmm_9.shape[-2:])
            bmm_9 = None

            sub_3 = torch.ops.aten.sub.Tensor(div_2, div_3)
            mul_3 = torch.ops.aten.mul.Tensor(view_23, sub_3)
            sub_3 = None
            mul_4 = torch.ops.aten.mul.Tensor(view_23, div_1)
            view_23 = None
            add_2 = torch.ops.aten.add.Tensor(view_20, mul_3)
            view_20 = mul_3 = None
            neg = torch.ops.aten.neg.default(mul_4)
            sum_4 = torch.ops.aten.sum.dim_IntList(neg, [3], True)
            neg = None
            div_5 = torch.ops.aten.div.Tensor(div_3, sum_2)
            div_3 = None
            neg_1 = torch.ops.aten.neg.default(sum_4)
            mul_5 = torch.ops.aten.mul.Tensor(neg_1, div_5)
            neg_1 = div_5 = None
            div_6 = torch.ops.aten.div.Tensor(sum_4, sum_2)
            sum_4 = sum_2 = None

            expand_12 = div_6.view(*orig_shape[:-2], *div_6.shape[-2:])
            div_6 = None
            view_2 = bmm.view(*orig_shape[:-2], *bmm.shape[-2:])
            bmm = None
            div = view_2 / d_scale
            view_2 = None

            sub_1 = torch.ops.aten.sub.Tensor(div, amax_1)
            exp_1 = torch.ops.aten.exp.default(sub_1)
            sub_1 = None
            mul_6 = torch.ops.aten.mul.Tensor(expand_12, exp_1)
            mul_7 = torch.ops.aten.mul.Tensor(expand_12, div_2)
            expand_12 = div_2 = None
            add_3 = torch.ops.aten.add.Tensor(mul_4, mul_6)
            mul_4 = mul_6 = None

            div_7 = torch.ops.aten.div.Tensor(add_3, d_scale)
            add_3 = None

            neg_2 = torch.ops.aten.neg.default(div_7)
            view_24 = div_7.view(-1, *div_7.shape[-2:])
            div_7 = None
            bmm_10 = torch.ops.aten.bmm.default(permute_7, view_24)
            permute_7 = None
            bmm_11 = torch.ops.aten.bmm.default(view_24, permute_8)

            view_25 = bmm_10.view(*orig_shape[:-2], *bmm_10.shape[-2:])
            bmm_10 = None
            grad_t_q = bmm_11.view(*orig_shape[:-2], *bmm_11.shape[-2:])
            bmm_11 = None
            permute_9 = view_25.transpose(-1, -2)
            view_25 = None

            bmm_12 = torch.ops.aten.bmm.default(permute_10, view_24)
            bmm_13 = torch.ops.aten.bmm.default(view_24, permute_11)
            view_24 = permute_11 = None

            view_28 = bmm_12.view(*orig_shape[:-2], *bmm_12.shape[-2:])
            bmm_12 = None
            view_29 = bmm_13.view(*orig_shape[:-2], *bmm_13.shape[-2:])
            bmm_13 = None
            grad_t_k = view_28.transpose(-1, -2)
            view_28 = None

            add_4 = torch.ops.aten.add.Tensor(mul_7, mul_5)
            mul_7 = mul_5 = None
            mul_8 = torch.ops.aten.mul.Tensor(add_4, exp_1)
            add_4 = exp_1 = None
            neg_3 = torch.ops.aten.neg.default(mul_8)
            sum_5 = torch.ops.aten.sum.dim_IntList(neg_3, [3], True)
            neg_3 = None
            eq = torch.ops.aten.eq.Tensor(amax_1, div)  # ;  amax_1 = div = None
            sum_6 = torch.ops.aten.sum.dim_IntList(eq, [-1], True)
            div_8 = torch.ops.aten.div.Tensor(sum_5, sum_6)
            sum_5 = sum_6 = None
            mul_9 = torch.ops.aten.mul.Tensor(div_8, eq)
            div_8 = eq = None
            add_5 = torch.ops.aten.add.Tensor(mul_8, mul_9)
            mul_8 = mul_9 = None
            mul_10 = torch.ops.aten.mul.Tensor(neg_2, 0)
            neg_2 = None
            add_6 = torch.ops.aten.add.Tensor(add_5, mul_10)
            add_5 = mul_10 = None
            grad_v = view_22
            view_22 = None

            add_8 = add_2
            add_2 = None
            mul_11 = torch.ops.aten.mul.Tensor(add_8, div_1)
            add_8 = None
            sum_7 = torch.ops.aten.sum.dim_IntList(mul_11, [-1], True)
            neg_4 = torch.ops.aten.neg.default(div_1)
            div_1 = None
            fma = torch.addcmul(mul_11, neg_4, sum_7)
            neg_4 = sum_7 = mul_11 = None
            add_9 = torch.ops.aten.add.Tensor(add_6, fma)
            add_6 = fma = None

            div_9 = add_9 / d_scale
            add_9 = None
            view_33 = div_9.view(-1, *div_9.shape[-2:])
            div_9 = None

            bmm_16 = torch.ops.aten.bmm.default(permute_10, view_33)
            permute_10 = None
            bmm_17 = torch.ops.aten.bmm.default(view_33, permute_8)
            view_33 = permute_8 = None

            view_34 = bmm_16.view(*orig_shape[:-2], *bmm_16.shape[-2:])
            bmm_16 = None
            view_35 = bmm_17.view(*orig_shape[:-2], *bmm_17.shape[-2:])
            bmm_17 = None

            grad_q = torch.ops.aten.add.Tensor(view_29, view_35)
            view_29 = view_35 = None

            permute_17 = view_34.transpose(-1, -2)
            view_34 = None

            grad_k = torch.ops.aten.add.Tensor(permute_9, permute_17)
            permute_9 = permute_17 = None
            return (grad_q, grad_k, grad_v, grad_t_q, grad_t_k, grad_t_v)

        @staticmethod
        def backward(ctx, d_q, d_k, d_v, d_t_q, d_t_k, d_t_v):
            raise NotImplementedError("Not implemented")

    class SDPAFunction(torch.autograd.Function):
        @staticmethod
        def forward(q, k, v):
            attn = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
            x = attn @ v
            return x

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            ctx.save_for_forward(*inputs)
            ctx.save_for_backward(*inputs, output)
            ctx.set_materialize_grads(False)

        @staticmethod
        def jvp(ctx, t_q, t_k, t_v):
            q, k, v = ctx.saved_for_forward
            t_y, *_ = SDPAJVPForwardFunction.apply(q, k, v, t_q, t_k, t_v)
            return t_y

        @staticmethod
        def backward(ctx, d_x):
            if d_x is not None:
                raise NotImplementedError("Not implemented")
            print("Backwards Not implemented, but grad is None anyways")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q, k, v):
            return SDPAFunction.apply(q, k, v)

    torch.manual_seed(123)
    b = 1 + 1
    s = 128 * 2
    h = 16 + 8
    d = 128 + 16
    device = torch.device("cuda")
    q, k, v = torch.randn((3, b, h, s, d), device=device)
    # q, k, v = torch.randn((3, 1, 16, 128), device=device)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    model = Model()

    # print(torch.cuda.memory_summary(device=device, abbreviated=True))
    t_q, t_k, t_v = (
        torch.randn_like(q, requires_grad=True),
        torch.randn_like(k, requires_grad=True),
        torch.randn_like(v, requires_grad=True),
    )
    # y, t_y = SDPAJVPForwardFunction.apply(q, k, v, t_q, t_k, t_v)
    y, t_y = torch.func.jvp(model, (q, k, v), (t_q, t_k, t_v))
    t_y.sum().backward()
    # print(torch.cuda.memory_summary(device=device, abbreviated=True))
    # print(y)
    # print(q.grad)
    # print(k.grad)

    torch.manual_seed(123)
    q_ref, k_ref, v_ref = torch.randn((3, b, h, s, d), device=device)
    q_ref.requires_grad_(True)
    k_ref.requires_grad_(True)
    v_ref.requires_grad_(True)
    t_q_ref, t_k_ref, t_v_ref = (
        torch.randn_like(q_ref, requires_grad=True),
        torch.randn_like(k_ref, requires_grad=True),
        torch.randn_like(v_ref, requires_grad=True),
    )
    model_vanilla = VanillaAttention()
    y_ref, t_y_ref = torch.func.jvp(
        model_vanilla, (q_ref, k_ref, v_ref), (t_q_ref, t_k_ref, t_v_ref)
    )
    t_y_ref.sum().backward()
    # print(y_ref)
    # print(q_ref.grad)
    # print(k_ref.grad)
    # print(v_ref.grad)

    y_err = (y - y_ref).abs().mean()
    t_y_err = (t_y - t_y_ref).abs().mean()
    print(f"y_err: {y_err}, t_y_err: {t_y_err}")

    q_err = (q_ref.grad - q.grad).abs().mean()
    k_err = (k_ref.grad - k.grad).abs().mean()
    v_err = (v_ref.grad - v.grad).abs().mean()
    t_q_err = (t_q_ref.grad - t_q.grad).abs().mean()
    t_k_err = (t_k_ref.grad - t_k.grad).abs().mean()
    t_v_err = (t_v_ref.grad - t_v.grad).abs().max()
    print(f"q_err: {q_err}, k_err: {k_err}, v_err: {v_err}")
    print(f"t_q_err: {t_q_err}, t_k_err: {t_k_err}, t_v_err: {t_v_err}")
    print("")


def test_flattened_autograd4():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q, k, v):
            return TritonJVPAttention2.SDPAFunction.apply(q, k, v)

    torch.manual_seed(123)
    b = 1 + 1
    s = 128 * 2
    h = 16 + 8
    d = 128
    device = torch.device("cuda")
    q, k, v = torch.randn((3, b, h, s, d), device=device)
    # q, k, v = torch.randn((3, 1, 16, 128), device=device)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    model = Model()

    print(torch.cuda.memory_summary(device=device, abbreviated=True))
    t_q, t_k, t_v = (
        torch.randn_like(q, requires_grad=True),
        torch.randn_like(k, requires_grad=True),
        torch.randn_like(v, requires_grad=True),
    )
    # y, t_y = SDPAJVPForwardFunction.apply(q, k, v, t_q, t_k, t_v)
    y, t_y = torch.func.jvp(model, (q, k, v), (t_q, t_k, t_v))
    print(torch.cuda.memory_summary(device=device, abbreviated=True))
    t_y.sum().backward()
    print(torch.cuda.memory_summary(device=device, abbreviated=True))
    torch.cuda.synchronize()
    # print(y)
    # print(q.grad)
    # print(k.grad)

    torch.manual_seed(123)
    q_ref, k_ref, v_ref = torch.randn((3, b, h, s, d), device=device)
    q_ref.requires_grad_(True)
    k_ref.requires_grad_(True)
    v_ref.requires_grad_(True)
    t_q_ref, t_k_ref, t_v_ref = (
        torch.randn_like(q_ref, requires_grad=True),
        torch.randn_like(k_ref, requires_grad=True),
        torch.randn_like(v_ref, requires_grad=True),
    )
    model_vanilla = VanillaAttention()
    y_ref, t_y_ref = torch.func.jvp(
        model_vanilla, (q_ref, k_ref, v_ref), (t_q_ref, t_k_ref, t_v_ref)
    )
    t_y_ref.sum().backward()
    # print(y_ref)
    # print(q_ref.grad)
    # print(k_ref.grad)
    # print(v_ref.grad)

    y_err = (y - y_ref).abs().mean()
    t_y_err = (t_y - t_y_ref).abs().mean()
    print(f"y_err: {y_err}, t_y_err: {t_y_err}")

    q_err = (q_ref.grad - q.grad).abs().mean()
    k_err = (k_ref.grad - k.grad).abs().mean()
    v_err = (v_ref.grad - v.grad).abs().mean()
    t_q_err = (t_q_ref.grad - t_q.grad).abs().mean()
    t_k_err = (t_k_ref.grad - t_k.grad).abs().mean()
    t_v_err = (t_v_ref.grad - t_v.grad).abs().max()
    print(f"q_err: {q_err}, k_err: {k_err}, v_err: {v_err}")
    print(f"t_q_err: {t_q_err}, t_k_err: {t_k_err}, t_v_err: {t_v_err}")
    print("")


def test_flattened_autograd5(b=1, h=24, s=3072, d=128, n_blocks=20, mode="debug"):
    dtype = torch.bfloat16

    device = torch.device("cuda")
    torch.manual_seed(123)
    model = (
        Model(d=d, n_blocks=n_blocks, n_heads=h, attn_backend="triton2")
        .to(device)
        # .to(dtype)
    )

    torch.manual_seed(123)
    x = torch.randn((b, s, d), device=device, requires_grad=True )
    t_x = torch.randn_like(x, requires_grad=True )

    n_warmup = 5
    n_bench = 15

    target_y = None
    target_t_y = None

    if mode == "bench" or mode == "profile":
        for _ in range(n_warmup):
            y, t_y = torch.func.jvp(model, (x,), (t_x,))
            if target_y is None:
                target_y = torch.randn_like(y)
            if target_t_y is None:
                target_t_y = torch.randn_like(t_y)
            ((y - target_y) + (t_y - target_t_y)).sum().backward()
            # (y + t_y).sum().backward()
            del y, t_y
        torch.cuda.synchronize()

    if mode == "profile":
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # record_shapes=True,
            # profile_memory=True,
        ) as prof:
            y, t_y = torch.func.jvp(model, (x,), (t_x,))
            if target_y is None:
                target_y = torch.randn_like(y)
            if target_t_y is None:
                target_t_y = torch.randn_like(t_y)
            ((y - target_y) + (t_y - target_t_y)).sum().backward()
            # (y + t_y).sum().backward()
            del y, t_y
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace("jvp_trace.json")
        return

    if mode == "bench":
        t0 = time.time()
        for _ in range(n_bench):
            y, t_y = torch.func.jvp(model, (x,), (t_x,))
            if target_y is None:
                target_y = torch.randn_like(y)
            if target_t_y is None:
                target_t_y = torch.randn_like(t_y)
            ((y - target_y) + (t_y - target_t_y)).sum().backward()
            # (y + t_y).sum().backward()
            del y, t_y
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"time flash: {(t1-t0)/n_bench}")

    if mode == "debug":
        print(torch.cuda.memory_summary(device=device, abbreviated=True))
        with torch.amp.autocast('cuda',   dtype=dtype):
            y, t_y = torch.func.jvp(model, (x,), (t_x,))
        # y2 = model(x)
        print(torch.cuda.memory_summary(device=device, abbreviated=True))
        if target_y is None:
            target_y = torch.randn_like(y)
        if target_t_y is None:
            target_t_y = torch.randn_like(t_y)
        ((y - target_y) + (t_y - target_t_y)).sum().backward()
        breakpoint()
        # ((y - target_y)).sum().backward()
        # (y + t_y).sum().backward()
        print(torch.cuda.memory_summary(device=device, abbreviated=True))
        torch.cuda.synchronize()
        # print(y)

    torch.manual_seed(123)
    x_ref = torch.randn((b, s, d), device=device, requires_grad=True, dtype=dtype)
    t_x_ref = torch.randn_like(x_ref, requires_grad=True, dtype=dtype)
    torch.manual_seed(123)
    model_vanilla = (
        Model(d=d, n_blocks=n_blocks, n_heads=h, attn_backend="vanilla")
        .to(device)
        .to(dtype)
    )
    if mode == "bench":
        for _ in range(n_warmup):
            y_ref, t_y_ref = torch.func.jvp(model_vanilla, (x_ref,), (t_x_ref,))
            if target_y is None:
                target_y = torch.randn_like(y)
            if target_t_y is None:
                target_t_y = torch.randn_like(t_y)
            ((y_ref - target_y) + (t_y_ref - target_t_y)).sum().backward()
            # (y_ref + t_y_ref).sum().backward()
            del y_ref, t_y_ref
        torch.cuda.synchronize()
    if mode == "bench":
        t0 = time.time()
        for _ in range(n_bench):
            y_ref, t_y_ref = torch.func.jvp(model_vanilla, (x_ref,), (t_x_ref,))
            if target_y is None:
                target_y = torch.randn_like(y)
            if target_t_y is None:
                target_t_y = torch.randn_like(t_y)
            ((y_ref - target_y) + (t_y_ref - target_t_y)).sum().backward()
            # (y_ref + t_y_ref).sum().backward()
            del y_ref, t_y_ref
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"time vanilla: {(t1-t0)/n_bench}")
        return

    if mode == "debug":
        y_ref, t_y_ref = torch.func.jvp(model_vanilla, (x_ref,), (t_x_ref,))
        # y_ref2 = model_vanilla(x_ref)
        if target_y is None:
            target_y = torch.randn_like(y)
        if target_t_y is None:
            target_t_y = torch.randn_like(t_y)
        ((y_ref - target_y) + (t_y_ref - target_t_y)).sum().backward()
        # ((y_ref - target_y)).sum().backward()
        # (y_ref + t_y_ref).sum().backward()
        print(torch.cuda.memory_summary(device=device, abbreviated=True))
        # print(y_ref)

        y_err = (y - y_ref).abs().mean()
        t_y_err = (t_y - t_y_ref).abs().mean()
        print(f"y_err: {y_err}, t_y_err: {t_y_err}")

        x_err = (x_ref.grad - x.grad).abs().mean()
        print(f"x_err: {x_err}")
        print("")
        t_x_err = (t_x_ref.grad - t_x.grad).abs().mean()
        print(f"t_x_err: {t_x_err}")


def test_jvp_backward_with_dump(path, bench=False):
    data = torch.load(path)
    q_orig = data["q_orig"]
    k_orig = data["k_orig"]
    v_orig = data["v_orig"]
    t_q_orig = data["t_q_orig"]
    t_k_orig = data["t_k_orig"]
    t_v_orig = data["t_v_orig"]
    tangents_o_orig = data["tangents_o_orig"]
    DEBUG_OUT = data["DEBUG_OUT"]
    lse_orig = data["lse_orig"]
    mu_orig = data["mu_orig"]
    li_orig = data["li_orig"]
    o_orig = data["o_orig"]

    n_warmup = 5 if bench else 0
    n_bench = 15 if bench else 1
    for _ in range(n_warmup):
        (
            dq,
            dk,
            dv,
            dtq,
            dtk,
            dtv,
        ) = TritonJVPAttention2.SDPAJVPFlashBackwardFunction.apply(
            q_orig,
            k_orig,
            v_orig,
            t_q_orig,
            t_k_orig,
            t_v_orig,
            tangents_o_orig,
            DEBUG_OUT,
            lse_orig,
            mu_orig,
            li_orig,
            o_orig,
        )
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_bench):
        (
            dq,
            dk,
            dv,
            dtq,
            dtk,
            dtv,
        ) = TritonJVPAttention2.SDPAJVPFlashBackwardFunction.apply(
            q_orig,
            k_orig,
            v_orig,
            t_q_orig,
            t_k_orig,
            t_v_orig,
            tangents_o_orig,
            DEBUG_OUT,
            lse_orig,
            mu_orig,
            li_orig,
            o_orig,
        )
    torch.cuda.synchronize()
    t1 = time.time()

    print(f"time: {1000*(t1-t0)/n_bench} ms")


if __name__ == "__main__":
    # print_jvp_sdpa_backward()
    # print_jvp_sdpa_fused_vanilla()
    # test_functionalize()
    # test_linearize()
    # test_compiled_graph()
    # test_compile_jvp_sdpa_fused_vanilla_graph()
    # test_flattened_autograd()
    # test_flattened_autograd2()
    # test_flattened_autograd3()
    # test_flattened_autograd4()
    # test_flattened_autograd5()
    # test_flattened_autograd5(b=1, h=24, s=3072, d=128, n_blocks=3, mode="debug")
    test_flattened_autograd5(b=1, h=5, s=165, d=128, n_blocks=1, mode="bench")
    # test_flattened_autograd5(b=1, h=24, s=3072, d=128, n_blocks=3, mode="profile")
    # test_flattened_autograd5(b=1, h=24, s=1024*16, d=128, n_blocks=3, mode="profile")
    # test_flattened_autograd5(b=1, h=24, s=1024*8, d=128, n_blocks=20, mode="bench")
    # test_flattened_autograd5(b=1, h=24, s=3072, d=128, n_blocks=20, mode="bench")
    # test_flattened_autograd5(b=1, h=24, s=4096, d=128, n_blocks=10, mode="bench")
    # test_flattened_autograd5(b=1, h=2, s=256, d=128, n_blocks=1, mode="debug")
    # test_compile_jvp_sdpa_fused_vanilla_2_graph()

    # test_jvp_backward_with_dump("tmp/debug_data_bhsd_1_24_3072_128.pt", bench=True)
    # test_jvp_backward_with_dump("tmp/debug_data_bhsd_1_2_256_128.pt")
    # test_jvp_backward_with_dump("tmp/debug_data_bhsd_1_2_16_128.pt")