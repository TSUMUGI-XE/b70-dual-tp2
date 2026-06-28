// b70_p2p_latency.cpp — small-message latency of peer copy vs host-staged round-trip.
// Feeds the alpha (latency) term of an alpha + beta*size comm model.
// Build: icpx -fsycl -O2 b70_p2p_latency.cpp -lze_loader -o b70_p2p_latency
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <chrono>
using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b){ return std::chrono::duration<double>(b-a).count()*1e6; }

int main(){
    std::vector<sycl::device> g;
    for (auto&p:sycl::platform::get_platforms()){ if(p.get_backend()!=sycl::backend::ext_oneapi_level_zero)continue;
        for(auto&d:p.get_devices(sycl::info::device_type::gpu)) if(!d.get_info<sycl::info::device::host_unified_memory>()) g.push_back(d);}
    if(g.size()<2){printf("NEED-2\n");return 2;}
    sycl::context c0(g[0]),c1(g[1]); sycl::queue q0(c0,g[0]),q1(c1,g[1]);
    auto zc1=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c1);
    auto zd1=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(g[1]);
    ze_command_queue_desc_t cq={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,nullptr,0,0,0,ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t cl; zeCommandListCreateImmediate(zc1,zd1,&cq,&cl);

    size_t szs[]={256, 4096, 65536, 262144};
    int iters=2000, warm=200;
    float* p0=sycl::malloc_device<float>(262144/4,q0);
    float* p1=sycl::malloc_device<float>(262144/4,q1);
    char*  hb=(char*)malloc(262144);
    printf("%-10s %14s %18s\n","size","peer-copy us","host-staged us");
    for(size_t b:szs){
        for(int i=0;i<warm;i++) zeCommandListAppendMemoryCopy(cl,p1,p0,b,nullptr,0,nullptr);
        auto t0=clk::now();
        for(int i=0;i<iters;i++) zeCommandListAppendMemoryCopy(cl,p1,p0,b,nullptr,0,nullptr);
        auto t1=clk::now();
        double peer=us(t0,t1)/iters;
        // host-staged round trip: dev0->host then host->dev1 (two synchronous SYCL copies)
        for(int i=0;i<warm;i++){ q0.memcpy(hb,p0,b).wait(); q1.memcpy(p1,hb,b).wait(); }
        auto h0=clk::now();
        for(int i=0;i<iters;i++){ q0.memcpy(hb,p0,b).wait(); q1.memcpy(p1,hb,b).wait(); }
        auto h1=clk::now();
        double host=us(h0,h1)/iters;
        printf("%-10zu %12.2f   %16.2f\n",b,peer,host);
    }
    free(hb); sycl::free(p0,q0); sycl::free(p1,q1);
    return 0;
}
