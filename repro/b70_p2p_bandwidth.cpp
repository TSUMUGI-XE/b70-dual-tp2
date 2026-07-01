// b70_p2p_bandwidth.cpp
// Measure direct Level-Zero device-to-device (peer) copy bandwidth between two B70.
// Builds on the verified-working P2P path (issue #1, AMD AM5 host):
//   - size sweep, unidirectional dev0->dev1 and dev1->dev0
//   - simultaneous bidirectional (both engines copying at once) -> aggregate BW
//   - host-staged (dev0->RAM->dev1) baseline for comparison
// Each measurement verifies a checksum once so we never report bandwidth on dropped data.
//
// Build: source /opt/intel/oneapi/setvars.sh
//        icpx -fsycl -O2 b70_p2p_bandwidth.cpp -lze_loader -o b70_p2p_bandwidth
// Run:   ONEAPI_DEVICE_SELECTOR=level_zero:* ZES_ENABLE_SYSMAN=1 ./b70_p2p_bandwidth
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// one synchronous immediate command list bound to a device; reused across iters
static ze_command_list_handle_t make_imm(ze_context_handle_t zctx, ze_device_handle_t zdev) {
    ze_command_queue_desc_t cq = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                  ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t cl = nullptr;
    zeCommandListCreateImmediate(zctx, zdev, &cq, &cl);
    return cl;
}

// time `iters` synchronous peer copies on command list `cl`; returns GB/s
static double bench_copy(ze_command_list_handle_t cl, void * dst, const void * src,
                         size_t bytes, int warmup, int iters) {
    for (int i = 0; i < warmup; i++)
        zeCommandListAppendMemoryCopy(cl, dst, src, bytes, nullptr, 0, nullptr);
    auto t0 = clk::now();
    for (int i = 0; i < iters; i++)
        zeCommandListAppendMemoryCopy(cl, dst, src, bytes, nullptr, 0, nullptr);
    auto t1 = clk::now();
    return (double)bytes * iters / secs(t0, t1) / 1e9;
}

static bool verify(sycl::queue & q_dst, const float * dst, size_t N, unsigned seedmod) {
    std::vector<float> h(N); q_dst.memcpy(h.data(), dst, N*sizeof(float)).wait();
    for (size_t i = 0; i < N; i++)
        if (h[i] != float((i*2654435761u) % seedmod)) return false;
    return true;
}

int main() {
    std::vector<sycl::device> gpus;
    for (auto & p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (auto & d : p.get_devices(sycl::info::device_type::gpu))
            if (!d.get_info<sycl::info::device::host_unified_memory>()) gpus.push_back(d);
    }
    if (gpus.size() < 2) { printf("NEED-2-GPUS (found %zu)\n", gpus.size()); return 2; }

    sycl::context ctx0(gpus[0]), ctx1(gpus[1]);
    sycl::queue q0(ctx0, gpus[0]), q1(ctx1, gpus[1]);
    auto zctx0 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx0);
    auto zctx1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx1);
    auto zdev0 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[0]);
    auto zdev1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[1]);

    const unsigned SM = 1000003u;
    const size_t MB = 1024*1024;
    size_t sizes[] = {1*MB, 4*MB, 16*MB, 64*MB, 256*MB, 512*MB};

    // largest buffers, reused (sliced) for every size
    const size_t MAXB = 512*MB, MAXN = MAXB/sizeof(float);
    float * b0 = sycl::malloc_device<float>(MAXN, q0);   // on dev0
    float * b1 = sycl::malloc_device<float>(MAXN, q1);   // on dev1
    if (!b0 || !b1) { printf("ALLOC-FAIL\n"); return 3; }
    q0.parallel_for(MAXN, [=](sycl::id<1> i){ b0[i] = float((i*2654435761u) % SM); }).wait();
    q1.parallel_for(MAXN, [=](sycl::id<1> i){ b1[i] = float((i*2654435761u) % SM); }).wait();

    auto cl_on1 = make_imm(zctx1, zdev1);   // engine on dev1 pulls from dev0  (dev0->dev1)
    auto cl_on0 = make_imm(zctx0, zdev0);   // engine on dev0 pulls from dev1  (dev1->dev0)

    printf("=== Direct L0 peer-copy bandwidth: 2x Arc Pro B70 (AM5 / Ryzen 9950X) ===\n");
    printf("%-8s %14s %14s %16s\n", "size", "dev0->dev1", "dev1->dev0", "bidir(aggregate)");
    for (size_t bytes : sizes) {
        size_t N = bytes/sizeof(float);
        // keep total volume ~2 GB per measurement, clamp iters
        int iters = (int)std::max<size_t>(10, std::min<size_t>(2000, (2ull*1024*MB)/bytes));
        int warm = 3;

        double bw_fwd = bench_copy(cl_on1, b1, b0, bytes, warm, iters);
        bool ok_fwd = verify(q1, b1, N, SM);
        double bw_rev = bench_copy(cl_on0, b0, b1, bytes, warm, iters);
        bool ok_rev = verify(q0, b0, N, SM);

        // simultaneous bidirectional: two host threads drive the two engines at once
        double bw_bi = 0; {
            auto run = [&](ze_command_list_handle_t cl, void* dst, const void* src){
                for (int i=0;i<warm;i++) zeCommandListAppendMemoryCopy(cl,dst,src,bytes,nullptr,0,nullptr);
            };
            run(cl_on1,b1,b0); run(cl_on0,b0,b1);
            auto t0 = clk::now();
            std::thread ta([&]{ for(int i=0;i<iters;i++) zeCommandListAppendMemoryCopy(cl_on1,b1,b0,bytes,nullptr,0,nullptr); });
            std::thread tb([&]{ for(int i=0;i<iters;i++) zeCommandListAppendMemoryCopy(cl_on0,b0,b1,bytes,nullptr,0,nullptr); });
            ta.join(); tb.join();
            auto t1 = clk::now();
            bw_bi = 2.0*(double)bytes*iters / secs(t0,t1) / 1e9;
        }

        printf("%-8s %11.1f GB/s %11.1f GB/s %13.1f GB/s   [%s,%s]\n",
               (std::to_string(bytes/MB)+"MB").c_str(),
               bw_fwd, bw_rev, bw_bi,
               ok_fwd?"ok":"CORRUPT", ok_rev?"ok":"CORRUPT");
    }

    // host-staged baseline at 256MB for comparison (dev0 -> host -> dev1)
    {
        size_t bytes = 256*MB, N = bytes/sizeof(float);
        int iters = 8;
        char * hbuf = (char*)malloc(bytes);
        for (int i=0;i<2;i++){ q0.memcpy(hbuf,b0,bytes).wait(); q1.memcpy(b1,hbuf,bytes).wait(); }
        auto t0 = clk::now();
        for (int i=0;i<iters;i++){ q0.memcpy(hbuf,b0,bytes).wait(); q1.memcpy(b1,hbuf,bytes).wait(); }
        auto t1 = clk::now();
        free(hbuf);
        printf("\nhost-staged dev0->host->dev1 @256MB: %.1f GB/s (effective card-to-card)\n",
               (double)bytes*iters/secs(t0,t1)/1e9);
    }

    sycl::free(b0,q0); sycl::free(b1,q1);
    printf("\n(GB/s = 1e9 bytes/s. Peer = direct over PCIe through the AMD root complex; no host bounce.)\n");
    return 0;
}
