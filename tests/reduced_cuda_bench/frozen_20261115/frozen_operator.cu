// FROZEN reference for the bench only: src/reduced_schur_pcg_cuda.cu as it stood before package 2
// (see frozen_operator.hpp); only the include and the namespace are renamed.
#include "frozen_operator.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <limits>
#include <utility>
#include "hdfe/ieee_bits.hpp"

namespace xhdfe_bench::frozen {
namespace {

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void check(cudaError_t status, const char* operation, bool allocating = false) {
    if (status == cudaSuccess) return;
    ErrorKind kind = ErrorKind::Fatal;
    if (allocating && status == cudaErrorMemoryAllocation) {
        kind = ErrorKind::Memory;
        (void)cudaGetLastError();
    } else if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        kind = ErrorKind::Unavailable;
    }
    throw Error(kind, std::string("CUDA reduced operator: ") + operation + ": " + cudaGetErrorString(status));
}

template<class T> struct Buffer {
    T* ptr = nullptr;
    ~Buffer() { if (ptr) (void)cudaFree(ptr); }
    void allocate(std::size_t size) {
        if(size>std::numeric_limits<std::size_t>::max()/sizeof(T))
            throw Error(ErrorKind::Memory,"CUDA reduced operator: allocation byte count overflow");
        if (size) check(cudaMalloc(reinterpret_cast<void**>(&ptr), size * sizeof(T)), "allocation", true);
    }
    void upload(const std::vector<T>& values, cudaStream_t stream, SetupSplit* split = nullptr) {
        const auto t0 = split ? Clock::now() : Clock::time_point{};
        allocate(values.size());
        const auto t1 = split ? Clock::now() : Clock::time_point{};
        if (!values.empty()) check(cudaMemcpyAsync(ptr, values.data(), values.size() * sizeof(T),
                                                   cudaMemcpyHostToDevice, stream), "layout upload");
        if (split) {
            split->alloc_seconds += std::chrono::duration<double>(t1 - t0).count();
            split->copy_seconds += seconds(t1);
        }
    }
};

struct Stream {
    cudaStream_t value = nullptr;
    ~Stream() { if (value) (void)cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value = nullptr;
    ~Event() { if (value) (void)cudaEventDestroy(value); }
};

__device__ double gather(const double* in, const int* offsets, const int* codes,
                         int dims, int cells, int cell, int lanes, int lane) {
    double value = 0.0;
    for (int j = 0; j < dims; ++j)
        value += in[static_cast<std::size_t>(offsets[j] + codes[static_cast<std::size_t>(j) * cells + cell]) * lanes + lane];
    return value;
}

// Fixed item sizes and a fixed binary tree: no atomic update participates
// in an operator value. Lanes never mix in the tree.
__global__ void segment_items(const double* in, const int* offsets, const int* codes,
    const int* mult, int dims, int cells, int lanes, const int* begin, const int* end,
    double* partial) {
    extern __shared__ double sum[];
    const int item = blockIdx.x, lane = threadIdx.x % lanes;
    double value = 0.0;
    for (std::size_t pos = static_cast<std::size_t>(begin[item]) + threadIdx.x / lanes;
         pos < static_cast<std::size_t>(end[item]); pos += static_cast<std::size_t>(blockDim.x) / lanes) {
        const int c=static_cast<int>(pos); // pos < end <= INT_MAX
        value += (mult ? static_cast<double>(mult[c]) : 1.0) *
                 gather(in, offsets, codes, dims, cells, c, lanes, lane);
    }
    sum[threadIdx.x] = value;
    __syncthreads();
    for (int step = blockDim.x / 2; step >= lanes; step /= 2) {
        if (threadIdx.x < step) sum[threadIdx.x] += sum[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x < lanes) partial[static_cast<std::size_t>(item) * lanes + lane] = sum[threadIdx.x];
}

__global__ void segment_finish(const double* partial, const int* ptr, const double* inverse,
                                int segments, int lanes, double* means) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(segments) * lanes) return;
    const int segment = index / lanes, lane = index % lanes;
    double value = 0.0;
    for (int item = ptr[segment]; item < ptr[segment + 1]; ++item)
        value += partial[static_cast<std::size_t>(item) * lanes + lane];
    means[index] = value * inverse[segment];
}

__global__ void level_items(const double* in, const int* offsets, const int* codes,
    const int* mult, const int* segment, const int* csr_cell, const double* means,
    int dims, int cells, int lanes, const int* level, const int* begin,
    const int* end, double* partial) {
    extern __shared__ double sum[];
    const int item = blockIdx.x, lane = threadIdx.x % lanes;
    const int lev = level[item];
    int dim = 0;
    while (dim + 1 < dims && lev >= offsets[dim + 1]) ++dim;
    const std::size_t base = static_cast<std::size_t>(dim) * cells;
    double value = 0.0;
    for (std::size_t pos = static_cast<std::size_t>(begin[item]) + threadIdx.x / lanes;
         pos < static_cast<std::size_t>(end[item]); pos += static_cast<std::size_t>(blockDim.x) / lanes) {
        const int c = csr_cell[base + pos];
        const double u = gather(in, offsets, codes, dims, cells, c, lanes, lane);
        value += (mult ? static_cast<double>(mult[c]) : 1.0) *
                 (u - means[static_cast<std::size_t>(segment[c]) * lanes + lane]);
    }
    sum[threadIdx.x] = value;
    __syncthreads();
    for (int step = blockDim.x / 2; step >= lanes; step /= 2) {
        if (threadIdx.x < step) sum[threadIdx.x] += sum[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x < lanes) partial[static_cast<std::size_t>(item) * lanes + lane] = sum[threadIdx.x];
}

__global__ void level_finish(const double* partial, const int* ptr, int levels,
                              int lanes, double* out) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(levels) * lanes) return;
    const int level = index / lanes, lane = index % lanes;
    double value = 0.0;
    for (int item = ptr[level]; item < ptr[level + 1]; ++item)
        value += partial[static_cast<std::size_t>(item) * lanes + lane];
    out[index] = value;
}

template<class T> std::size_t bytes(const std::vector<T>& v) { return v.size() * sizeof(T); }
template<class T> std::size_t capacity_bytes(const std::vector<T>& v) { return v.capacity() * sizeof(T); }
std::size_t add(std::initializer_list<std::size_t> sizes) {
    std::size_t total=0;
    for(auto size:sizes) {
        if(size>std::numeric_limits<std::size_t>::max()-total)
            throw Error(ErrorKind::Memory,"CUDA reduced operator: byte count overflow");
        total+=size;
    }
    return total;
}
void validate(const Layout& l) {
    const auto bad=[] { throw Error(ErrorKind::Fatal,"CUDA reduced operator: invalid layout shape"); };
    if(l.dims<1 || l.dims>4 || l.levels<0 || l.segments<0 || l.cells<0 ||
       (l.lanes!=1 && l.lanes!=2 && l.lanes!=4 && l.lanes!=8)) bad();
    const std::size_t b=l.dims,m=l.levels,L=l.segments,C=l.cells;
    if(l.offsets.size()!=b+1 || l.offsets.front()!=0 || l.offsets.back()!=l.levels ||
       l.segment_ptr.size()!=L+1 || l.segment_ptr.front()!=0 || l.segment_ptr.back()!=l.cells ||
       l.cell_segment.size()!=C || l.codes.size()!=b*C || (!l.mult.empty() && l.mult.size()!=C) ||
       l.segment_inverse.size()!=L || l.csr_offsets.size()!=b || l.csr_ptr.size()!=m+b ||
       l.csr_cell.size()!=b*C || l.a_item_ptr.size()!=L+1 || l.b_item_ptr.size()!=m+1 ||
       l.a_item_begin.size()!=l.a_item_end.size() || l.a_item_segment.size()!=l.a_item_begin.size() ||
       l.b_item_begin.size()!=l.b_item_end.size() || l.b_item_level.size()!=l.b_item_begin.size() ||
       l.a_item_begin.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       l.b_item_begin.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       l.a_item_ptr.front()!=0 || l.a_item_ptr.back()!=static_cast<int>(l.a_item_begin.size()) ||
       l.b_item_ptr.front()!=0 || l.b_item_ptr.back()!=static_cast<int>(l.b_item_begin.size())) bad();
    for(int j=0;j<l.dims;++j)
        if(l.offsets[j]<0 || l.offsets[j]>l.offsets[j+1] || l.csr_offsets[j]<0 ||
           static_cast<std::size_t>(l.csr_offsets[j])!=static_cast<std::size_t>(l.offsets[j])+j) bad();
    for(int s=0;s<l.segments;++s)
        if(l.segment_ptr[s]>l.segment_ptr[s+1] ||
           !hdfe::detail::ieee_finite(l.segment_inverse[s]) || !(l.segment_inverse[s]>0)) bad();
}

} // namespace

std::size_t Layout::host_bytes() const {
    return add({capacity_bytes(offsets),capacity_bytes(segment_ptr),capacity_bytes(cell_segment),capacity_bytes(codes),capacity_bytes(mult),
        capacity_bytes(segment_inverse),capacity_bytes(csr_offsets),capacity_bytes(csr_ptr),capacity_bytes(csr_cell),
        capacity_bytes(a_item_segment),capacity_bytes(a_item_begin),capacity_bytes(a_item_end),capacity_bytes(a_item_ptr),
        capacity_bytes(b_item_level),capacity_bytes(b_item_begin),capacity_bytes(b_item_end),capacity_bytes(b_item_ptr)});
}
std::size_t Layout::device_bytes() const {
    return add({bytes(offsets),bytes(cell_segment),bytes(codes),bytes(mult),bytes(segment_inverse),
        bytes(csr_cell),bytes(a_item_begin),bytes(a_item_end),bytes(a_item_ptr),
        bytes(b_item_level),bytes(b_item_begin),bytes(b_item_end),bytes(b_item_ptr),
        sizeof(double)*static_cast<std::size_t>(lanes)*(2U*static_cast<std::size_t>(levels)+
            static_cast<std::size_t>(segments)+a_item_begin.size()+b_item_begin.size())});
}

struct Operator::State {
    Stream stream;
    Event start, uploaded, computed, downloaded;
    Buffer<int> offsets, codes, mult, segment, a_begin, a_end, a_ptr, b_level, b_begin, b_end, b_ptr, csr_cell;
    Buffer<double> inverse, in, out, means, a_partial, b_partial;
    int dims=0, cells=0, levels=0, segments=0, lanes=0, a_items=0, b_items=0;
    bool profile=false;
    Statistics stats;
    ~State() { if(stream.value) (void)cudaStreamSynchronize(stream.value); }
    // split: trace only (SetupSplit); no clock is read for it when it is null.
    void init(const Layout& l, std::size_t budget, bool profiling, SetupSplit* split) {
        const auto t0 = Clock::now();
        validate(l);profile=profiling;stats.timed=profile;
        const auto t_device = split ? Clock::now() : Clock::time_point{};
        if (split) split->validate_seconds = std::chrono::duration<double>(t_device - t0).count();
        if (l.device_bytes() > budget) throw Error(ErrorKind::Memory, "CUDA reduced operator: device budget");
        int devices=0; check(cudaGetDeviceCount(&devices), "device count",true);
        if (!devices) throw Error(ErrorKind::Unavailable, "CUDA reduced operator: no device");
        check(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking), "stream creation",true);
        std::size_t free=0,total=0; check(cudaMemGetInfo(&free,&total), "available memory",true);
        if (l.device_bytes() > free) throw Error(ErrorKind::Memory, "CUDA reduced operator: free device memory");
        if(profile) for (Event* e : {&start,&uploaded,&computed,&downloaded})
            check(cudaEventCreate(&e->value), "event creation",true);
        if (split) split->device_seconds = seconds(t_device);
        dims=l.dims;cells=l.cells;levels=l.levels;segments=l.segments;lanes=l.lanes;
        a_items=static_cast<int>(l.a_item_begin.size());b_items=static_cast<int>(l.b_item_begin.size());
        offsets.upload(l.offsets,stream.value,split);codes.upload(l.codes,stream.value,split);mult.upload(l.mult,stream.value,split);
        segment.upload(l.cell_segment,stream.value,split);inverse.upload(l.segment_inverse,stream.value,split);
        a_begin.upload(l.a_item_begin,stream.value,split);a_end.upload(l.a_item_end,stream.value,split);a_ptr.upload(l.a_item_ptr,stream.value,split);
        b_level.upload(l.b_item_level,stream.value,split);b_begin.upload(l.b_item_begin,stream.value,split);b_end.upload(l.b_item_end,stream.value,split);b_ptr.upload(l.b_item_ptr,stream.value,split);csr_cell.upload(l.csr_cell,stream.value,split);
        const auto t_alloc = split ? Clock::now() : Clock::time_point{};
        in.allocate(static_cast<std::size_t>(levels)*lanes);out.allocate(static_cast<std::size_t>(levels)*lanes);
        means.allocate(static_cast<std::size_t>(segments)*lanes);
        a_partial.allocate(static_cast<std::size_t>(a_items)*lanes);b_partial.allocate(static_cast<std::size_t>(b_items)*lanes);
        if (split) split->alloc_seconds += seconds(t_alloc);
        const auto t_sync = split ? Clock::now() : Clock::time_point{};
        check(cudaStreamSynchronize(stream.value), "layout completion");
        if (split) split->sync_seconds = seconds(t_sync);
        stats.host_layout_bytes=l.host_bytes();stats.device_payload_bytes=l.device_bytes();stats.setup_seconds=seconds(t0);
    }
};

Operator::Operator(const Layout& layout, std::size_t budget, bool profile, SetupSplit* setup) {
    auto state=std::make_unique<State>();state->init(layout,budget,profile,setup);state_=std::move(state);
}
Operator::~Operator() = default;
const Statistics& Operator::statistics() const noexcept { return state_->stats; }
std::size_t Operator::host_state_bytes() noexcept { return sizeof(Operator) + sizeof(State); }

void Operator::apply(const double* input, double* output, KernelSplit* kernels) {
    State& s=*state_;const auto t0=s.profile?Clock::now():Clock::time_point{};
    // Trace only: three more events between the four kernels, for this call.
    KernelSplit* const split_to=s.profile?kernels:nullptr;
    Event split[3];
    if(split_to) for(Event& e:split) check(cudaEventCreate(&e.value),"split event creation",true);
    const std::size_t nbytes=sizeof(double)*static_cast<std::size_t>(s.levels)*s.lanes;
    if(s.profile) check(cudaEventRecord(s.start.value,s.stream.value),"start event");
    check(cudaMemcpyAsync(s.in.ptr,input,nbytes,cudaMemcpyHostToDevice,s.stream.value),"operand upload");
    if(s.profile) check(cudaEventRecord(s.uploaded.value,s.stream.value),"uploaded event");
    if(s.a_items) {
        segment_items<<<s.a_items,256,256*sizeof(double),s.stream.value>>>(s.in.ptr,s.offsets.ptr,s.codes.ptr,s.mult.ptr,s.dims,s.cells,s.lanes,s.a_begin.ptr,s.a_end.ptr,s.a_partial.ptr);
        check(cudaGetLastError(),"segment item launch");
    }
    if(split_to) check(cudaEventRecord(split[0].value,s.stream.value),"split event");
    if(s.segments) {
        segment_finish<<<(static_cast<std::size_t>(s.segments)*s.lanes+255)/256,256,0,s.stream.value>>>(s.a_partial.ptr,s.a_ptr.ptr,s.inverse.ptr,s.segments,s.lanes,s.means.ptr);
        check(cudaGetLastError(),"segment reduction launch");
    }
    if(split_to) check(cudaEventRecord(split[1].value,s.stream.value),"split event");
    if(s.b_items) {
        level_items<<<s.b_items,256,256*sizeof(double),s.stream.value>>>(s.in.ptr,s.offsets.ptr,s.codes.ptr,s.mult.ptr,s.segment.ptr,s.csr_cell.ptr,s.means.ptr,s.dims,s.cells,s.lanes,s.b_level.ptr,s.b_begin.ptr,s.b_end.ptr,s.b_partial.ptr);
        check(cudaGetLastError(),"level item launch");
    }
    if(split_to) check(cudaEventRecord(split[2].value,s.stream.value),"split event");
    if(s.levels) {
        level_finish<<<(static_cast<std::size_t>(s.levels)*s.lanes+255)/256,256,0,s.stream.value>>>(s.b_partial.ptr,s.b_ptr.ptr,s.levels,s.lanes,s.out.ptr);
        check(cudaGetLastError(),"level reduction launch");
    }
    if(s.profile) check(cudaEventRecord(s.computed.value,s.stream.value),"computed event");
    check(cudaMemcpyAsync(output,s.out.ptr,nbytes,cudaMemcpyDeviceToHost,s.stream.value),"result download");
    if(s.profile) check(cudaEventRecord(s.downloaded.value,s.stream.value),"downloaded event");
    check(cudaStreamSynchronize(s.stream.value),"operator completion");
    if(s.profile) {
        float ms=0.0f;check(cudaEventElapsedTime(&ms,s.start.value,s.uploaded.value),"upload timing");s.stats.h2d_ms+=ms;
        check(cudaEventElapsedTime(&ms,s.uploaded.value,s.computed.value),"kernel timing");s.stats.kernel_ms+=ms;
        check(cudaEventElapsedTime(&ms,s.computed.value,s.downloaded.value),"download timing");s.stats.d2h_ms+=ms;
    }
    if(split_to) {
        float ms=0.0f;
        check(cudaEventElapsedTime(&ms,s.uploaded.value,split[0].value),"split timing");split_to->segment_items_ms+=ms;
        check(cudaEventElapsedTime(&ms,split[0].value,split[1].value),"split timing");split_to->segment_finish_ms+=ms;
        check(cudaEventElapsedTime(&ms,split[1].value,split[2].value),"split timing");split_to->level_items_ms+=ms;
        check(cudaEventElapsedTime(&ms,split[2].value,s.computed.value),"split timing");split_to->level_finish_ms+=ms;
    }
    ++s.stats.calls;s.stats.h2d_bytes+=nbytes;s.stats.d2h_bytes+=nbytes;
    if(s.profile)s.stats.apply_seconds+=seconds(t0);
}

} // namespace xhdfe_bench::frozen
