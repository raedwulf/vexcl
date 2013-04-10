#include <iostream>
#include <random>
#include <iterator>
#include <cassert>
#include <limits>
#include <tuple>

//#define VEXCL_SHOW_KERNELS
#include <vexcl/vexcl.hpp>

#define TESTS_ON
#define RUNS 1000

#include "radix_sort.h"

using namespace vex;

static bool all_passed = true;

bool run_test(const std::string &name, std::function<std::tuple<bool,double,double,double>()> test) {
    char fc = std::cout.fill('.');
    std::cout << name << ": " << std::setw(62 - name.size()) << "." << std::flush;
    std::cout.fill(fc);

    bool rc;
    double avg_time, min_time, max_time;
    std::tie(rc, avg_time, min_time, max_time) = test();
    all_passed = all_passed && rc;
    std::cout << (rc ? " success." : " failed.") << " [" <<
        "avg:" << int(avg_time * 1000000) << "us," <<
        "min:" << int(min_time * 1000000) << "us," <<
        "max:" << int(max_time * 1000000) << "us]" << std::endl;
    return rc;
}

template<typename K, typename V>
struct RadixSort {
    const int sort_bits = 32;

    RadixSort(const std::vector<cl::CommandQueue>& queue,
              int workgroup_size = 128, int num_per_workitem = 4,
              int max_workgroups = 60)
        : queue(queue), workgroup_size(workgroup_size),
        num_per_workitem(num_per_workitem), max_workgroups(max_workgroups) {
        // Only works with single queues
        assert(queue.size() == 1);
        // Resize the work buffer
        work_buffer.resize(queue, max_workgroups * 16);
        // Build kernel for device in context
        cl::Program program = build_sources(qctx(queue[0]),
                                            radix_sort_source);
        count_kernel = cl::Kernel(program, "StreamCountKernel");
        scatter_kernel = cl::Kernel(program, "SortAndScatterKernel");
        scan_kernel = cl::Kernel(program, "PrefixScanKernel");
    }

    void execute(vector<K> &input, vector<K> &swap_buffer, vector<K> &output) {
        assert(output.size() == input.size() && input.size() == swap_buffer.size());
        assert(input.size() % (workgroup_size * num_per_workitem) == 0);

        int n = input.size();
        int n_blocks = n / (num_per_workitem * workgroup_size);
        int n_work_groups_to_execute = std::min(max_workgroups, n_blocks);
        int n_blocks_per_group =
            (n_blocks + n_work_groups_to_execute-1) / n_work_groups_to_execute;

        vector<K> *tmp, *src = &input, *dst = &swap_buffer;

        for (int j = 0, i = 0; j < sort_bits; j += 4, i++) {
            cl_int4 const_buffer = {i, n_blocks, n_work_groups_to_execute, n_blocks_per_group};
            count_kernel.setArg(0, (*src)(0));
            count_kernel.setArg(1, work_buffer(0));
            count_kernel.setArg(2, const_buffer);
            std::cout << workgroup_size * n_work_groups_to_execute << " " << workgroup_size << std::endl;
            queue[0].enqueueNDRangeKernel(count_kernel, cl::NullRange,
                                       workgroup_size * n_work_groups_to_execute,
                                       workgroup_size);
            std::cout << "WORKED!" << std::endl;
            scan_kernel.setArg(0, work_buffer(0));
            scan_kernel.setArg(1, const_buffer);
            queue[0].enqueueNDRangeKernel(scan_kernel, cl::NullRange,
                                       workgroup_size, workgroup_size);
            scatter_kernel.setArg(0, work_buffer(0));
            scatter_kernel.setArg(1, (*src)(0));
            scatter_kernel.setArg(2, (*dst)(0));
            scatter_kernel.setArg(3, const_buffer);
            queue[0].enqueueNDRangeKernel(scatter_kernel, cl::NullRange,
                                       workgroup_size * n_work_groups_to_execute,
                                       workgroup_size);
            tmp = dst; dst = src; src = dst;
        }
    }

    const std::vector<cl::CommandQueue>& queue;
    cl::Kernel count_kernel;
    cl::Kernel scatter_kernel;
    cl::Kernel scan_kernel;

    vex::vector<cl_uint> work_buffer;

    int workgroup_size, num_per_workitem, max_workgroups;
};

int main(int argc, char *argv[]) {
    try {
        vex::Context ctx(Filter::Env);
        std::cout << ctx << std::endl;

        if (ctx.empty()) {
            std::cerr << "No OpenCL devices found." << std::endl;
            return 1;
        }

        std::vector<cl::CommandQueue> single_queue(1, ctx.queue(0));

        uint seed = argc > 1 ? atoi(argv[1]) : static_cast<uint>(time(0));
        std::cout << "seed: " << seed << std::endl << std::endl;
        srand(seed);

        run_test("Sorting random numbers",
            [&]() -> std::tuple<bool, double, double, double> {
                const size_t N = 1 << 10;
                int rc = true;
                vex::vector<cl_uint> x(ctx, N);
                vex::vector<cl_uint> y(ctx, N);
                vex::vector<cl_uint> z(ctx, N);
                Random<cl_uint> rand0;
                x = rand0(element_index(), rand());
                profiler prof;
                double time = 0.0;
                double min_time = std::numeric_limits<double>::infinity();
                double max_time = -min_time;

                RadixSort<cl_uint,cl_uint> radix_sort(single_queue);
                for (auto i = 0; i < RUNS; ++i) {
                    prof.tic_cl("Run");
                    radix_sort.execute(x, y, z);
                    auto t = prof.toc("Run");
                    time += t;
                    if (min_time > t) min_time = t;
                    if (max_time < t) max_time = t;
                }
                return std::make_tuple(rc,time/double(RUNS),min_time,max_time);
            });

    } catch (const cl::Error &err) {
        std::cerr << "OpenCL error: " << err << std::endl;
        return 1;
    } catch (const std::exception &err) {
        std::cerr << "Error: " << err.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return 1;
    }

    return !all_passed;
}


