#include "ac_arena_storage.hpp"
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>

namespace {
struct Edge {
    int to=0, rev=0;
    double weight=0;
    int count=1;
    static inline std::size_t attempts=0, live=0;
    static inline std::size_t fail_at=std::numeric_limits<std::size_t>::max();
    static void* operator new[](std::size_t bytes) {
        if(attempts++==fail_at)throw std::bad_alloc();
        void* out=::operator new(bytes);++live;return out;
    }
    static void operator delete[](void* data) noexcept {
        if(data){--live;::operator delete(data);}
    }
};
using Store=hdfe::detail::AcArenaStorage<Edge,16>;
void check(bool yes,const char* label){if(!yes)throw std::runtime_error(label);}
void fill(Edge* data,int n,int label){for(int i=0;i<n;++i)data[i]={label,i,1.0+i*0.125,label+1};}
void values(const Edge* data,int n,int label){for(int i=0;i<n;++i)
    check(data[i].to==label && data[i].rev==i && data[i].weight==1.0+i*0.125 && data[i].count==label+1,"relocated edge changed");}

void lifetime() {
    Store store(5,true);
    Edge* a=store.initial(0,8);Edge* b=store.initial(1,8);
    Edge* c=store.initial(2,9);Edge* d=store.initial(3,17);Edge* e=store.initial(4,4);
    fill(a,8,0);fill(b,8,1);fill(c,9,2);fill(d,17,3);fill(e,4,4);
    store.finish_initial();
    check(b==a+8 && c!=a && e==c+9,"initial packing or dedicated tail changed");
    check(Edge::live==3 && store.owners_valid(),"initial ownership");
    auto fresh=Store::growth_buffer(16);std::copy_n(a,8,fresh.get());a=fresh.get();
    store.install_growth(0,std::move(fresh),8,16,8);
    values(a,8,0);values(b,8,1);
    check(Edge::live==4 && store.owns(0,a,16) && !store.owns(0,a,17),"growth ownership");
    check(store.eliminate(1,8),"initial list must detach");b=nullptr;
    check(Edge::live==3 && store.owners_valid(),"shared slab not reclaimed");
    check(store.eliminate(3,17),"dedicated slab detach");d=nullptr;
    check(Edge::live==2,"dedicated slab not reclaimed");
    fresh=Store::growth_buffer(16);std::copy_n(c,9,fresh.get());c=fresh.get();
    store.install_growth(2,std::move(fresh),9,16,9);
    values(c,9,2);values(e,4,4);
    check(Edge::live==3 && store.owners_valid(),"remaining initial list lost");
    check(store.eliminate(4,4),"last slab detach");e=nullptr;
    check(Edge::live==2 && store.statistics().initial_live_slots==0,"last slab pinned");
    check(!store.eliminate(0,16) && !store.eliminate(2,16),"small buffers changed policy");
    check(store.statistics().heap_live_slots==32 && store.statistics().peak_owned_slots==65,
          "storage peak or retained capacity accounting");
    check(store.owners_valid(),"final ownership");
}

void failures() {
    Store store(1,true);
    Edge::fail_at=Edge::attempts;
    bool failed=false;
    try{(void)store.initial(0,8);}catch(const std::bad_alloc&){failed=true;}
    check(failed && Edge::live==0 && store.owners_valid(),"initial failure left storage");
    Edge::fail_at=std::numeric_limits<std::size_t>::max();
    Edge* data=store.initial(0,8);fill(data,8,7);store.finish_initial();
    Edge::fail_at=Edge::attempts;failed=false;
    try{auto buffer=Store::growth_buffer(16);(void)buffer;}catch(const std::bad_alloc&){failed=true;}
    check(failed && Edge::live==1 && store.owners_valid() && store.owns(0,data,8),
          "growth failure revoked old list");
    values(data,8,7);Edge::fail_at=std::numeric_limits<std::size_t>::max();
    auto fresh=Store::growth_buffer(16);std::copy_n(data,8,fresh.get());data=fresh.get();
    store.install_growth(0,std::move(fresh),8,16,8);
    fresh=Store::growth_buffer(128);std::copy_n(data,8,fresh.get());data=fresh.get();
    store.install_growth(0,std::move(fresh),16,128,8);
    values(data,8,7);check(Edge::live==1 && store.owners_valid(),"old grown buffer survived");
    check(store.eliminate(0,128) && Edge::live==0,"large grown buffer not freed");data=nullptr;
    check(store.owns(0,data,0) && store.statistics().heap_live_slots==0,"freed owner still live");
}

void boundaries() {
    Store empty(0,true);empty.finish_initial();check(empty.owners_valid(),"empty storage");
    check(Store::growth_capacity(1)==1 && Store::growth_capacity(9)==16 &&
          Store::growth_capacity(64)==64 && Store::growth_capacity(65)==128,"growth capacities");
    bool failed=false;try{(void)Store::growth_capacity(std::numeric_limits<int>::max());}
    catch(const std::length_error&){failed=true;}check(failed,"capacity overflow accepted");
    failed=false;try{Store invalid(-1);}catch(const std::invalid_argument&){failed=true;}
    check(failed,"negative vertex count accepted");
}
}

int main(){
    try{
        lifetime();check(Edge::live==0,"lifetime leak");
        failures();check(Edge::live==0,"failure leak");
        boundaries();check(Edge::live==0,"boundary leak");
        std::puts("AC_ARENA_STORAGE_CONTRACT_PASS");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"AC_ARENA_STORAGE_CONTRACT_FAIL: %s\n",e.what());return 1;}
}
