# [BMG / Arc Pro B70] Is cross-root-port P2P expected to work? L0 direct device-to-device copy returns `0x70000003`; kernel refuses at the P2PDMA layer (host bridge not whitelisted) — and can the capability be made queryable?

> Note: English isn't my first language, so I drafted this writeup with AI help — but every result below was measured by me on the hardware described, and the reproducer is a single self-contained file you can build and run.

> **Confound ruled out (control done).** The test host normally runs `pcie_acs_override=downstream,multifunction` (added for an unrelated VFIO passthrough), which can make the kernel refuse peer P2P via `pci_p2pdma_distance()` and would mimic this signature. I re-ran the probe on a control boot with that override **removed**: the peer copy still returns `0x70000003`, host-staged still passes — identical to the override-on baseline (see `acs_probe_acsOFF_result.log`). So the override is **not** the cause. On that same control boot the kernel logged the reason directly: `xe 0000:07:00.0: cannot be used for peer-to-peer DMA as the client and provider (0000:03:00.0) do not share an upstream bridge or whitelisted host bridge`.

## Summary

On a workstation with **two Intel Arc Pro B70 (BMG)** GPUs, a Level-Zero **direct device-to-device** copy between the two cards — issued from a single process using per-device L0 contexts — returns **`0x70000003` (`ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`)** instead of performing the peer copy. It is a clean error return (not a hang). A **host-staged** copy of the same buffers (dev0 → host → dev1) succeeds, so the allocations and queues themselves are fine — only the cross-device peer path fails.

This appears to be the low-level reason multi-GPU collective stacks can't establish GPU P2P on this configuration (see "Downstream impact" below).

**This may well be expected behavior** — I ruled out my own host config (see the control below), and the kernel then refuses the peer path because the two cards are on separate root ports with no shared PCIe switch and the host bridge isn't on `pci_p2pdma_whitelist`. So I'm not claiming a clear bug. I'm asking two things: (1) **should** BMG + a consumer Intel host bridge be able to route P2P through the root complex (i.e. is this a whitelist gap to fix, or a topology limit to document), and (2) regardless, could the unsupported case be **reported as a queryable capability** instead of surfacing as `OUT_OF_DEVICE_MEMORY` on a valid allocation, so frameworks stop probing-by-failure.

## Environment

- GPUs: 2× Intel Arc Pro B70 (BMG), separate PCIe root ports
- Kernel: Linux 7.x with the **xe** driver — reproduced on two builds (stock Ubuntu HWE 7.0.0-22 and a self-built 7.1-rc), same result
- compute-runtime / Level-Zero: 1.15.x (NEO 26.x, IGC 2.34.x)
- Single process, two GPUs exposed via `ONEAPI_DEVICE_SELECTOR=level_zero:*`, `ZES_ENABLE_SYSMAN=1`
- IOMMU/VT-d enabled; both GPUs in separate IOMMU groups

## Reproducer

Single file, ~80 lines (attached as `b70_p2p_copy_probe.cpp`). It enumerates the two discrete L0 GPUs, builds **per-device contexts/queues**, allocates 16 MB on each, then attempts the L0 immediate-command-list copy dev0→dev1, and finally a host-staged copy for comparison.

```
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -O2 b70_p2p_copy_probe.cpp -lze_loader -o b70_p2p_copy_probe
ONEAPI_DEVICE_SELECTOR=level_zero:* ZES_ENABLE_SYSMAN=1 ./b70_p2p_copy_probe
```

### Observed output

```
Discrete Level-Zero GPUs: 2
  dev[0] = Intel(R) Arc(TM) Pro B70 Graphics
  dev[1] = Intel(R) Arc(TM) Pro B70 Graphics
--- TEST 1: L0 direct copy dev0 -> dev1 ---
  zeCommandListCreateImmediate  -> 0x0
  zeCommandListAppendMemoryCopy -> 0x70000003 (FAILED)
--- TEST 2: host-staged copy dev0 -> host -> dev1 ---
  checksum: PASS
RESULT: l0_p2p=BROKEN(ze-error)  host_staged=WORKS
```

## Expected

The direct dev0→dev1 `zeCommandListAppendMemoryCopy` should either perform the peer copy (if P2P is supported) or, if peer access is genuinely unsupported on this platform, surface a more specific/queryable capability (e.g. via `zeDeviceCanAccessPeer`-style reporting) rather than an `OUT_OF_DEVICE_MEMORY` on a valid 16 MB allocation.

## Downstream impact (why this matters)

- **oneCCL / PyTorch XCCL**: a 2-rank (2-process) tensor-parallel job using the `xccl` backend hangs during the OFI/ATL transport handshake on this hardware. An exhaustive environment sweep (provider, process launcher, `CCL_LOCAL_RANK`/`SIZE` injection, `CCL_ENABLE_SYCL_KERNELS=0`) does not change the outcome — consistent with the underlying device-P2P path, not an env/config issue.
- **Single-process multi-GPU (llama.cpp ggml-sycl)** runs fine: its `dev2dev_memcpy` tries the same L0 direct copy, gets the error, and **falls back to a host-staged copy** — exactly TEST 2 above — so a model split across both B70 works (just without the fast P2P path).

## Negative controls (what is NOT the cause)

I swapped the host platform (different CPU/chipset: separate PCIe root ports, Gen4 x8/x8, separate IOMMU groups) — the L0 P2P copy fails identically and the oneCCL job hangs at the same point. So PCIe bandwidth, root-port separation, and IOMMU grouping are ruled out. The behavior reproduces across two kernel versions as well.

I also re-ran on a control boot with `pcie_acs_override` **removed** (full log: `acs_probe_acsOFF_result.log`): override confirmed absent, peer copy still `0x70000003`, host-staged still PASS — so the override is ruled out too. On that boot the kernel P2PDMA layer logged the cause explicitly:

```
xe 0000:07:00.0: cannot be used for peer-to-peer DMA as the client and provider
(0000:03:00.0) do not share an upstream bridge or whitelisted host bridge
```

The two cards sit on separate root ports with no shared PCIe switch, so a peer copy must traverse the root complex / host bridge, which the kernel only permits when that host bridge is on `pci_p2pdma_whitelist` — and this Intel client host bridge isn't. Neither GPU exposes a `p2pdma` sysfs node even with ACS off (xe not registered as a P2PDMA provider here).

## Ask

1. Given the kernel refuses the path because the host bridge isn't on `pci_p2pdma_whitelist`, is BMG + a consumer Intel host bridge **eligible** for that whitelist (i.e. does the silicon route P2P through the root complex correctly), or is cross-root-port P2P genuinely unsupported on this class of platform? If the former, can the host bridge be whitelisted; if the latter, can it be documented so users don't chase it?
2. Either way, could compute-runtime **report the capability explicitly** (e.g. a `zeDeviceCanAccessPeer`-style query) so frameworks choose the host-staged path deliberately instead of probing-by-failure?
3. Is `0x70000003` (`OUT_OF_DEVICE_MEMORY`) the intended L0 error code to surface for "peer access refused by the kernel P2PDMA layer", or does it mask the more specific underlying reason the kernel already names in dmesg?

Happy to run additional diagnostics or driver builds on this hardware — I have the two B70 set up and can iterate.
