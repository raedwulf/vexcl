// Reference implementation from Slash Sandbox
// http://code.google.com/p/slash-sandbox

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

#define MORTON30 "#define MORTON30(x,y,z,out) " \
                 "x = (x | (x << 16)) & 0x030000FF;" \
                 "x = (x | (x <<  8)) & 0x0300F00F;" \
                 "x = (x | (x <<  4)) & 0x030C30C3;" \
                 "x = (x | (x <<  2)) & 0x09249249;" \
                 "y = (y | (y << 16)) & 0x030000FF;" \
                 "y = (y | (y <<  8)) & 0x0300F00F;" \
                 "y = (y | (y <<  4)) & 0x030C30C3;" \
                 "y = (y | (y <<  2)) & 0x09249249;" \
                 "z = (z | (z << 16)) & 0x030000FF;" \
                 "z = (z | (z <<  8)) & 0x0300F00F;" \
                 "z = (z | (z <<  4)) & 0x030C30C3;" \
                 "z = (z | (z <<  2)) & 0x09249249;" \
                 "out = x | (y << 1) | (z << 2);\n"

VEX_FUNCTION(morton_code30, uint(cl_uint4),
             MORTON30
             "uint out;"
             "MORTON30(prm1.x, prm1.y, prm1.z, out);"
             "return out;");

VEX_FUNCTION(morton_code60, cl_ulong(cl_uint4),
             MORTON30
             "uint lo_x = prm1.x & 1023u;"
             "uint lo_y = prm1.y & 1023u;"
             "uint lo_z = prm1.z & 1023u;"
             "uint hi_x = prm1.x >> 10u;"
             "uint hi_y = prm1.y >> 10u;"
             "uint hi_z = prm1.z >> 10u;"
             "uint lo, hi;"
             "MORTON30(lo_x, lo_y, lo_z, lo);"
             "MORTON30(hi_x, hi_y, hi_z, hi);"
             "return ((ulong)hi << 30) | (ulong)lo;");

VEX_FUNCTION(quantize, cl_uint4(cl_float4, cl_int4),
             "float4 nf = (float4)prm2;\n"
             "return (uint4)(max(min((int4)(prm1*nf),prm2-1),0));\n");

template <typename morton_type>
struct morton_code { };
template <>
struct morton_code<cl_uint> {
    template <typename T>
    static decltype(morton_code30(quantize(T(),cl_int4()))) calculate(T p) {
        return morton_code30(quantize(p, cl_int4({1024, 1024, 1024, 1024})));
    }
};
template <>
struct morton_code<cl_ulong> {
    template <typename T>
    static decltype(morton_code60(quantize(T(),cl_int4()))) calculate(T p) {
        return morton_code60(quantize(p, cl_int4({1 << 20, 1 << 20, 1 << 20, 1 << 20})));
    }
};

template <typename morton_type = cl_ulong>
struct lbvh_builder
{
    lbvh_builder(vex::multivector<cl_uint, 2>& nodes,
        vex::vector<cl_uint2>& leaves,
        vex::vector<cl_uint>& indices)
        : nodes(nodes), leaves(leaves), indices(indices) { }

    template <typename iterator>
    void build(const std::array<float, 3> bbox,
               const vex::vector<cl_float4> points,
               const uint point_count,
               const uint max_leaf_size) {

        //using bintree_gen_context::split_task;
        assert(points.size() >= point_count);

        _bbox = bbox;
        if (codes.size() < point_count) codes.resize(point_count);
        if (leaves.size() < point_count) leaves.resize(point_count);
        if (indices.size() < point_count) indices.resize(point_count);

        codes = morton_code<morton_type>::calculate(points);
    }

    vex::multivector<cl_uint, 2>& nodes;
    vex::vector<cl_uint2>& leaves;
    vex::vector<cl_uint>& indices;
    vex::vector<morton_type> codes;
    uint levels[64];
    std::array<float, 3> _bbox;
    uint node_count;
    uint leaf_count;
};

int main(int argc, char *argv[]) {
    try {
        vex::Context ctx(Filter::DoublePrecision && Filter::Env);
        std::cout << ctx << std::endl;

        if (ctx.empty()) {
            std::cerr << "No OpenCL devices found." << std::endl;
            return 1;
        }

        std::vector<cl::CommandQueue> single_queue(1, ctx.queue(0));

        uint seed = argc > 1 ? atoi(argv[1]) : static_cast<uint>(time(0));
        std::cout << "seed: " << seed << std::endl << std::endl;
        srand(seed);

        run_test("Generate 30-bit morton code",
            [&]() -> std::tuple<bool, double, double, double> {
                const size_t N = 1 << 28;
                int rc = true;
                vex::vector<cl_float3> x(ctx, N);
                vex::vector<uint> y(ctx, N);
                Random<cl_float3> rand0;
                x = rand0(element_index(), rand());
                profiler prof;
                double time = 0.0;
                double min_time = std::numeric_limits<double>::infinity();
                double max_time = -min_time;
                for (auto i = 0; i < RUNS; ++i) {
                    prof.tic_cl("Run");
                    y = morton_code<cl_uint>::calculate(x);
                    auto t = prof.toc("Run");
                    time += t;
                    if (min_time > t) min_time = t;
                    if (max_time < t) max_time = t;
                }
                return std::make_tuple(rc,time/double(RUNS),min_time,max_time);
            });

        run_test("Generate 60-bit morton code",
            [&]() -> std::tuple<bool, double, double, double> {
                const size_t N = 1 << 28;
                int rc = true;
                vex::vector<cl_float3> x(ctx, N);
                vex::vector<cl_ulong> y(ctx, N);
                Random<cl_float3> rand0;
                x = rand0(element_index(), rand());
                profiler prof;
                double time = 0.0;
                double min_time = std::numeric_limits<double>::infinity();
                double max_time = -min_time;
                for (auto i = 0; i < RUNS; ++i) {
                    prof.tic_cl("Run");
                    y = morton_code<cl_ulong>::calculate(x);
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


