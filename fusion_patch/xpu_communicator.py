# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Modified 2026 by tsumugi-xe for Intel Arc Pro B70 (BMG) host-staged TP=2
# (VLLM_XPU_HOSTSTAGED_TP=1). Changes are marked inline with "[B70 TP=2 fusion]".
# Derived from vLLM; remains licensed under Apache-2.0 (see ../NOTICE).


import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)

# [B70 TP=2 fusion] When VLLM_XPU_HOSTSTAGED_TP=1, oneCCL/xccl is never created
# (parallel_state builds the device group on gloo). gloo cannot move GPU tensors,
# so every collective is host-staged: GPU -> CPU -> gloo collective -> GPU. This
# mirrors llama.cpp ggml-sycl dev2dev_memcpy's host-staged fallback (probe_v2 proved
# the host path clean) and sidesteps the same-root-complex P2P frontier (#41663/#8022)
# that hangs the XCCL OFI/ATL handshake. Payload per decode all_reduce is the hidden
# vector (~10KB), so this is latency-bound, not bandwidth-bound.
_HOSTSTAGED = os.environ.get("VLLM_XPU_HOSTSTAGED_TP", "0") == "1"


class XpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        self._hoststaged = _HOSTSTAGED
        if self._hoststaged:
            logger.info(
                "[B70 TP=2 fusion] XpuCommunicator host-staged collectives ACTIVE "
                "(gloo cpu_group, xccl bypassed)."
            )
        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

            else:  # type: ignore[has-type]
                logger.warning(
                    "`%s` all2all manager is not supported on XPU. "
                    "Falling back to AgRs manager for XPU, "
                    "which is the Default backend",
                    self.all2all_backend,  # type: ignore[has-type]
                )
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self._hoststaged:
            # gloo cannot reduce float16/bfloat16 ("Gloo does not support half"),
            # so upcast to fp32 on the host for the reduction, then cast back.
            need = input_.dtype in (torch.float16, torch.bfloat16)
            cpu = input_.to("cpu", dtype=torch.float32 if need else input_.dtype)
            dist.all_reduce(cpu, group=self.cpu_group)
            return cpu.to(input_.device, dtype=input_.dtype)
        output = input_.clone()
        dist.all_reduce(output, group=self.device_group)
        return output

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Base class all_gather does a raw dist.all_gather_into_tensor on device_group
        # (gloo here) with a GPU tensor -> "No backend for xpu". Host-stage it. all_gather
        # is a pure data move (no reduction) so no fp32 upcast needed; gloo handles fp16.
        if not self._hoststaged:
            return super().all_gather(input_, dim)
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        work = input_.to("cpu")
        output_tensor = torch.empty(output_size, dtype=work.dtype, device="cpu")
        dist.all_gather_into_tensor(output_tensor, work, group=self.cpu_group)
        output_tensor = output_tensor.to(input_.device)
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        if self._hoststaged:
            need = input_tensor.dtype in (torch.float16, torch.bfloat16)
            rdt = torch.float32 if need else input_tensor.dtype
            cpu_in = input_tensor.to("cpu", dtype=rdt)
            cpu_out = torch.empty(output_shape, dtype=rdt, device="cpu")
            dist.reduce_scatter_tensor(cpu_out, cpu_in, group=self.cpu_group)
            output = cpu_out.to(input_tensor.device, dtype=input_tensor.dtype)
            return output.movedim(0, dim).contiguous()

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        group = self.cpu_group if self._hoststaged else self.device_group
        if self._hoststaged:
            need = input_tensor.dtype in (torch.float16, torch.bfloat16)
            rdt = torch.float32 if need else input_tensor.dtype
            src = input_tensor.to("cpu", dtype=rdt)
        else:
            src = input_tensor
        output = torch.empty(
            output_shape, dtype=src.dtype, device=src.device
        )
        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
            # if inputs shape in different ranks is not the same using reduce_scatter
            input_splits = list(src.split(sizes, dim=0))
            dist.reduce_scatter(output, input_splits, group=group)
        else:
            dist.reduce_scatter_tensor(output, src, group=group)
        if self._hoststaged:
            output = output.to(input_tensor.device, dtype=input_tensor.dtype)
        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        group = self.cpu_group if self._hoststaged else self.device_group

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            dev = input_.device
            work = input_.to("cpu") if self._hoststaged else input_
            input_size = work.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert work.shape[dim] == sizes[self.rank_in_group], (
                    f"{work.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=work.dtype, device=work.device
            )

            if sizes is not None:
                all_gather_list = []
                for size in sizes:
                    all_gather_list.append(
                        torch.empty(
                            (size,) + work.shape[1:],
                            dtype=work.dtype,
                            device=work.device,
                        )
                    )
                dist.all_gather(all_gather_list, work, group=group)
                output_tensor = torch.cat(all_gather_list, dim=0)
            else:
                dist.all_gather([output_tensor], work, group=group)
            if self._hoststaged:
                output_tensor = output_tensor.to(dev)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        return output_list

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        group = self.cpu_group if self._hoststaged else self.device_group
        work = input_.to("cpu") if self._hoststaged else input_
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=work.dtype, device=work.device
        )
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, work, group=group)
        if self._hoststaged:
            output_tensor = output_tensor.to(input_.device)
        if self.rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> None:
        if self._hoststaged:
            cpu = input_.to("cpu")
            dist.broadcast(cpu, src=src, group=self.cpu_group)
            input_.copy_(cpu.to(input_.device))
            return
        dist.broadcast(input_, src=src, group=self.device_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )
