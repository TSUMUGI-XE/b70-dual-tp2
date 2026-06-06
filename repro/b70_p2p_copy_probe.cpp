// b70_p2p_copy_probe.cpp
// Minimal reproducer: Level-Zero direct device-to-device memcpy between two discrete
// Intel GPUs in a SINGLE process, using per-device L0 contexts (the pattern a single-process
// multi-GPU framework such as llama.cpp's ggml-sycl dev2dev_memcpy uses).
//
// Observed on 2x Intel Arc Pro B70 (BMG) with the xe kernel driver + compute-runtime:
//   zeCommandListAppendMemoryCopy(dst_on_dev1, src_on_dev0) returns
//   0x70000003 (ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) — i.e. the destination engine cannot
//   reach the source device's allocation (peer access not established). It is a clean ERROR
//   return, NOT a hang. A host-staged copy (dev0 -> host -> dev1) of the same buffers succeeds.
//
// Build: source /opt/intel/oneapi/setvars.sh
//        icpx -fsycl -O2 b70_p2p_copy_probe.cpp -lze_loader -o b70_p2p_copy_probe
// Run:   ONEAPI_DEVICE_SELECTOR=level_zero:* ZES_ENABLE_SYSMAN=1 ./b70_p2p_copy_probe
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <vector>
#include <cstdlib>

static int l0_dev2dev(sycl::queue & q_dst, void * dst, const void * src, size_t bytes) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
    ze_command_queue_desc_t cq = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                  ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t cl;
    ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq, &cl);
    printf("  zeCommandListCreateImmediate  -> 0x%x\n", r);
    if (r != ZE_RESULT_SUCCESS) return -1;
    r = zeCommandListAppendMemoryCopy(cl, dst, src, bytes, nullptr, 0, nullptr);
    printf("  zeCommandListAppendMemoryCopy -> 0x%x %s\n", r,
           r == ZE_RESULT_SUCCESS ? "(copy done)" : "(FAILED)");
    zeCommandListDestroy(cl);
    return r == ZE_RESULT_SUCCESS ? 1 : -1;
}

int main() {
    std::vector<sycl::device> gpus;
    for (auto & p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (auto & d : p.get_devices(sycl::info::device_type::gpu))
            if (!d.get_info<sycl::info::device::host_unified_memory>()) gpus.push_back(d);  // discrete only
    }
    printf("Discrete Level-Zero GPUs: %zu\n", gpus.size());
    for (size_t i = 0; i < gpus.size(); i++)
        printf("  dev[%zu] = %s\n", i, gpus[i].get_info<sycl::info::device::name>().c_str());
    if (gpus.size() < 2) { printf("RESULT: NEED-2-GPUS\n"); return 2; }

    // per-device contexts (each GPU is its own L0 context/queue)
    sycl::context ctx0(gpus[0]), ctx1(gpus[1]);
    sycl::queue q0(ctx0, gpus[0]), q1(ctx1, gpus[1]);

    const size_t N = 4u * 1024 * 1024, bytes = N * sizeof(float);
    float * p0 = sycl::malloc_device<float>(N, q0);
    float * p1 = sycl::malloc_device<float>(N, q1);
    if (!p0 || !p1) { printf("RESULT: ALLOC-FAIL\n"); return 3; }
    q0.parallel_for(N, [=](sycl::id<1> i){ p0[i] = float(i % 1000); }).wait();
    q1.memset(p1, 0, bytes).wait();

    printf("--- TEST 1: L0 direct copy dev0 -> dev1 ---\n");
    int l0 = l0_dev2dev(q1, p1, p0, bytes);
    size_t errs = 0;
    if (l0 == 1) {
        std::vector<float> h(N); q1.memcpy(h.data(), p1, bytes).wait();
        for (size_t i = 0; i < N; i++) if (h[i] != float(i % 1000)) errs++;
        printf("  checksum: %s\n", errs == 0 ? "PASS" : "FAIL");
    }

    printf("--- TEST 2: host-staged copy dev0 -> host -> dev1 ---\n");
    q1.memset(p1, 0, bytes).wait();
    char * hbuf = (char *)malloc(bytes);
    q0.memcpy(hbuf, p0, bytes).wait();
    q1.memcpy(p1, hbuf, bytes).wait();
    free(hbuf);
    std::vector<float> h2(N); q1.memcpy(h2.data(), p1, bytes).wait();
    size_t errh = 0; for (size_t i = 0; i < N; i++) if (h2[i] != float(i % 1000)) errh++;
    printf("  checksum: %s\n", errh == 0 ? "PASS" : "FAIL");

    sycl::free(p0, q0); sycl::free(p1, q1);
    printf("\nRESULT: l0_p2p=%s  host_staged=%s\n",
           (l0 == 1 && errs == 0) ? "WORKS" : "BROKEN(ze-error)",
           errh == 0 ? "WORKS" : "BROKEN");
    return 0;
}
