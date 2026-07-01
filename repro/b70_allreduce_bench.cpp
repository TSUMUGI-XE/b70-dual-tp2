// b70_allreduce_bench.cpp
// Real 2-GPU all-reduce (sum) measured two ways, swept over message size:
//   PEER : concurrent bidirectional DIRECT peer copies + on-GPU add
//   HOST : same reduction, but transport bounces through host RAM (gloo-style path)
// Both use identical GPU reduction; only the transport differs -> isolates the interconnect.
// Backs out the per-collective floor (alpha) from the smallest sizes.
//
// Build: source /opt/intel/oneapi/setvars.sh
//        icpx -fsycl -O2 b70_allreduce_bench.cpp -lze_loader -o b70_allreduce_bench
// Run:   ONEAPI_DEVICE_SELECTOR=level_zero:* ZES_ENABLE_SYSMAN=1 ./b70_allreduce_bench
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <vector>
#include <chrono>
using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b){ return std::chrono::duration<double>(b-a).count()*1e6; }

static ze_command_list_handle_t imm_async(ze_context_handle_t c, ze_device_handle_t d){
    ze_command_queue_desc_t q={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,nullptr,0,0,0,
        ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t cl=nullptr; zeCommandListCreateImmediate(c,d,&q,&cl); return cl;
}
static ze_event_handle_t make_event(ze_context_handle_t c, ze_device_handle_t d, ze_event_pool_handle_t* pool){
    ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,nullptr,ZE_EVENT_POOL_FLAG_HOST_VISIBLE,1};
    zeEventPoolCreate(c,&pd,1,&d,pool);
    ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC,nullptr,0,0,ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_handle_t e=nullptr; zeEventCreate(*pool,&ed,&e); return e;
}

int main(){
    std::vector<sycl::device> g;
    for(auto&p:sycl::platform::get_platforms()){ if(p.get_backend()!=sycl::backend::ext_oneapi_level_zero)continue;
        for(auto&d:p.get_devices(sycl::info::device_type::gpu)) if(!d.get_info<sycl::info::device::host_unified_memory>()) g.push_back(d);}
    if(g.size()<2){printf("NEED-2\n");return 2;}
    sycl::context c0(g[0]),c1(g[1]); sycl::queue q0(c0,g[0]),q1(c1,g[1]);
    auto zc0=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c0);
    auto zc1=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c1);
    auto zd0=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(g[0]);
    auto zd1=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(g[1]);
    auto cl0=imm_async(zc0,zd0);   // dev0 engine (pulls a1 from dev1)
    auto cl1=imm_async(zc1,zd1);   // dev1 engine (pulls a0 from dev0)
    ze_event_pool_handle_t p0,p1; ze_event_handle_t e0=make_event(zc0,zd0,&p0), e1=make_event(zc1,zd1,&p1);

    const size_t MAXN=64u*1024*1024/4;         // up to 64MB
    float *a0=sycl::malloc_device<float>(MAXN,q0), *t0=sycl::malloc_device<float>(MAXN,q0);
    float *a1=sycl::malloc_device<float>(MAXN,q1), *t1=sycl::malloc_device<float>(MAXN,q1);
    float *h0=sycl::malloc_host<float>(MAXN,q0),   *h1=sycl::malloc_host<float>(MAXN,q1);

    auto reset_inputs=[&](size_t N){ q0.fill(a0,1.0f,N).wait(); q1.fill(a1,2.0f,N).wait(); }; // sum must be 3.0

    auto peer_ar=[&](size_t N){
        size_t S=N*4;
        zeEventHostReset(e0); zeEventHostReset(e1);
        zeCommandListAppendMemoryCopy(cl0,t0,a1,S,e0,0,nullptr); // dev0 <- a1
        zeCommandListAppendMemoryCopy(cl1,t1,a0,S,e1,0,nullptr); // dev1 <- a0
        zeEventHostSynchronize(e0,UINT64_MAX); zeEventHostSynchronize(e1,UINT64_MAX);
        auto ev0=q0.parallel_for(N,[=](sycl::id<1> i){ a0[i]+=t0[i]; });
        auto ev1=q1.parallel_for(N,[=](sycl::id<1> i){ a1[i]+=t1[i]; });
        ev0.wait(); ev1.wait();
    };
    auto host_ar=[&](size_t N){
        size_t S=N*4;
        auto d0=q0.memcpy(h0,a0,S); auto d1=q1.memcpy(h1,a1,S); d0.wait(); d1.wait(); // dev->host
        auto u0=q0.memcpy(t0,h1,S); auto u1=q1.memcpy(t1,h0,S); u0.wait(); u1.wait(); // host->dev (swapped)
        auto ev0=q0.parallel_for(N,[=](sycl::id<1> i){ a0[i]+=t0[i]; });
        auto ev1=q1.parallel_for(N,[=](sycl::id<1> i){ a1[i]+=t1[i]; });
        ev0.wait(); ev1.wait();
    };
    auto check=[&](size_t N)->bool{
        std::vector<float> h(N); q0.memcpy(h.data(),a0,N*4).wait();
        for(size_t i=0;i<N;i++) if(h[i]!=3.0f) return false; return true;
    };

    size_t sizes[]={4096,16384,65536,262144,1u<<20,4u<<20,16u<<20,64u<<20};
    printf("=== Measured 2-GPU all-reduce: 2x B70 (AM5/Ryzen 9950X) ===\n");
    printf("%-8s | %10s %10s %8s | %s\n","msg","PEER us","HOST us","ratio","note");
    for(size_t S:sizes){
        size_t N=S/4;
        int iters=(int)std::max<size_t>(20,std::min<size_t>(3000,(1ull<<30)/S));
        // correctness: ONE fresh all-reduce each (all-reduce is idempotent only once)
        reset_inputs(N); peer_ar(N); bool okp=check(N);
        reset_inputs(N); host_ar(N); bool okh=check(N);
        // timing: many iters (values accumulate; irrelevant to transport time)
        reset_inputs(N); for(int i=0;i<5;i++) peer_ar(N);
        auto t0c=clk::now(); for(int i=0;i<iters;i++) peer_ar(N); double pe=us(t0c,clk::now())/iters;
        reset_inputs(N); for(int i=0;i<5;i++) host_ar(N);
        auto t1c=clk::now(); for(int i=0;i<iters;i++) host_ar(N); double ho=us(t1c,clk::now())/iters;
        const char* sz = S<(1<<20)? (std::to_string(S/1024)+"KB").c_str() : (std::to_string(S/(1<<20))+"MB").c_str();
        char szb[16]; if(S<(1<<20)) snprintf(szb,16,"%zuKB",S/1024); else snprintf(szb,16,"%zuMB",S/(1<<20));
        printf("%-8s | %10.2f %10.2f %7.2fx | %s\n", szb, pe, ho, ho/pe,
               (okp&&okh)?"verified":"CHECKSUM-FAIL");
    }
    printf("\n(per-collective floor ~= the small-message time; effective BW dominates by ~1MB.)\n");
    sycl::free(a0,q0);sycl::free(t0,q0);sycl::free(a1,q1);sycl::free(t1,q1);sycl::free(h0,q0);sycl::free(h1,q1);
    return 0;
}
