// b70_p2p_canaccess_probe.cpp
// Companion to b70_p2p_copy_probe.cpp that answers issue #1's TWO explicit checks:
//   (1) zeDeviceCanAccessPeer(dev_a, dev_b) == 1   (peer access reported), AND
//   (2) the copied buffer matches the source (checksum PASS) — not just a SUCCESS code.
// Runs the direct L0 device-to-device copy in BOTH directions and verifies each.
//
// Build: source /opt/intel/oneapi/setvars.sh
//        icpx -fsycl -O2 b70_p2p_canaccess_probe.cpp -lze_loader -o b70_p2p_canaccess_probe
// Run:   ONEAPI_DEVICE_SELECTOR=level_zero:* ZES_ENABLE_SYSMAN=1 ./b70_p2p_canaccess_probe
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <vector>

static ze_device_handle_t zedev(const sycl::queue & q) {
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
}

// direct L0 immediate-command-list copy issued on q_dst's context/device
static int l0_copy(sycl::queue & q_dst, void * dst, const void * src, size_t bytes) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
    ze_command_queue_desc_t cq = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                  ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t cl;
    if (zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq, &cl) != ZE_RESULT_SUCCESS) return -1;
    ze_result_t r = zeCommandListAppendMemoryCopy(cl, dst, src, bytes, nullptr, 0, nullptr);
    zeCommandListDestroy(cl);
    return r == ZE_RESULT_SUCCESS ? 1 : -1;
}

// one direction: alloc src on q_src device, dst on q_dst device, fill+copy+verify
static bool one_dir(const char * tag, sycl::queue & q_src, sycl::queue & q_dst) {
    const size_t N = 4u*1024*1024, bytes = N*sizeof(float);
    float * src = sycl::malloc_device<float>(N, q_src);
    float * dst = sycl::malloc_device<float>(N, q_dst);
    if (!src || !dst) { printf("  %s: ALLOC-FAIL\n", tag); return false; }
    // distinctive pattern (not all-0, not all-1s) so a dropped-data RC is obvious
    q_src.parallel_for(N, [=](sycl::id<1> i){ src[i] = float((i*2654435761u) % 1000003u); }).wait();
    q_dst.memset(dst, 0xAB, bytes).wait();

    int rc = l0_copy(q_dst, dst, src, bytes);
    bool ok = false;
    if (rc == 1) {
        std::vector<float> h(N); q_dst.memcpy(h.data(), dst, bytes).wait();
        size_t errs = 0;
        for (size_t i = 0; i < N; i++)
            if (h[i] != float((i*2654435761u) % 1000003u)) errs++;
        ok = (errs == 0);
        printf("  %s: zeCommandListAppendMemoryCopy=SUCCESS  checksum=%s (%zu/%zu mismatched)\n",
               tag, ok ? "PASS" : "FAIL", errs, N);
    } else {
        printf("  %s: zeCommandListAppendMemoryCopy=FAILED (clean error, no copy)\n", tag);
    }
    sycl::free(src, q_src); sycl::free(dst, q_dst);
    return ok;
}

int main() {
    std::vector<sycl::device> gpus;
    for (auto & p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (auto & d : p.get_devices(sycl::info::device_type::gpu))
            if (!d.get_info<sycl::info::device::host_unified_memory>()) gpus.push_back(d);
    }
    printf("Discrete Level-Zero GPUs: %zu\n", gpus.size());
    if (gpus.size() < 2) { printf("RESULT: NEED-2-GPUS\n"); return 2; }

    sycl::context ctx0(gpus[0]), ctx1(gpus[1]);
    sycl::queue q0(ctx0, gpus[0]), q1(ctx1, gpus[1]);

    // CHECK 1: zeDeviceCanAccessPeer, both directions
    ze_bool_t ab = 0, ba = 0;
    ze_result_t r1 = zeDeviceCanAccessPeer(zedev(q0), zedev(q1), &ab);
    ze_result_t r2 = zeDeviceCanAccessPeer(zedev(q1), zedev(q0), &ba);
    printf("--- CHECK 1: zeDeviceCanAccessPeer ---\n");
    printf("  dev0->dev1 : ret=0x%x  canAccess=%d\n", r1, (int)ab);
    printf("  dev1->dev0 : ret=0x%x  canAccess=%d\n", r2, (int)ba);

    // CHECK 2: real copy + checksum, both directions
    printf("--- CHECK 2: direct L0 peer copy + verify ---\n");
    bool fwd = one_dir("dev0->dev1", q0, q1);
    bool rev = one_dir("dev1->dev0", q1, q0);

    bool peer_ok = ab && ba && fwd && rev;
    printf("\nRESULT: canAccessPeer=%s  copy_verify=%s  =>  P2P_%s\n",
           (ab && ba) ? "YES" : "NO",
           (fwd && rev) ? "PASS" : "FAIL",
           peer_ok ? "VERIFIED" : "BROKEN");
    return peer_ok ? 0 : 1;
}
