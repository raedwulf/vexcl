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

#define MORTON30 "#define MORTON30(v,out) " \
                 "v = (v | (v << 16)) & 0x030000FF;" \
                 "v = (v | (v <<  8)) & 0x0300F00F;" \
                 "v = (v | (v <<  4)) & 0x030C30C3;" \
                 "v = (v | (v <<  2)) & 0x09249249;" \
                 "out = v.x | (v.y << 1) | (v.z << 2);\n"

VEX_FUNCTION(morton_code30, uint(cl_uint4),
             MORTON30
             "uint out = 0;\n"
             "MORTON30(prm1, out);\n"
             "return out;\n");

VEX_FUNCTION(morton_code60, cl_ulong(cl_uint4),
             MORTON30
             "uint4 lo = prm1 & 1023u;"
             "uint4 hi = prm1 >> 10u;"
             "uint out_lo, out_hi;"
             "MORTON30(lo, out_lo);"
             "MORTON30(hi, out_hi);"
             "return ((ulong)out_hi << 30) | (ulong)out_lo;");

VEX_FUNCTION(quantize, cl_uint4(cl_float4, cl_int),
             "int4 a = (int4)(prm1*(float)prm2);\n"
             "int4 b = prm2-1;\n"
             //"int4 mi4 = min(a,b);\n"
             //"uint4 ma4 = max(mi4,0);\n"
             "int4 mi4;\n"
             "uint4 ma4;\n"
             "mi4.x = min(a.x, b.x);\n"
             "mi4.y = min(a.y, b.y);\n"
             "mi4.z = min(a.z, b.z);\n"
             "ma4.x = max(mi4.x, 0);\n"
             "ma4.y = max(mi4.y, 0);\n"
             "ma4.z = max(mi4.z, 0);\n"
             "return ma4;\n");

template <typename morton_type>
struct morton_code { };
template <>
struct morton_code<cl_uint> {
    template <typename T>
    static inline decltype(morton_code30(quantize(T(),cl_int()))) calculate(const T& p) {
        return morton_code30(quantize(p, 1024));
    }
};
template <>
struct morton_code<cl_ulong> {
    template <typename T>
    static inline decltype(morton_code60(quantize(T(),cl_int()))) calculate(const T& p) {
        return morton_code60(quantize(p, 1 << 20));
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
        vex::Context ctx(Filter::Env);
        //vex::Context ctx(Filter::DoublePrecision && Filter::Env);
        std::cout << ctx << std::endl;

        if (ctx.empty()) {
            std::cerr << "No OpenCL devices found." << std::endl;
            return 1;
        }

        std::vector<cl::CommandQueue> single_queue(1, ctx.queue(0));

        uint seed = argc > 1 ? atoi(argv[1]) : static_cast<uint>(time(0));
        std::cout << "seed: " << seed << std::endl << std::endl;
        srand(seed);

        run_test("Quantize",
            [&]() -> std::tuple<bool, double, double, double> {
                const size_t N = 1 << 10;
                int rc = true;
                vex::vector<cl_float4> x(ctx, N);
                vex::vector<cl_uint4> y(ctx, N);
                Random<cl_float4> rand0;
                x = rand0(element_index(), rand());
                profiler prof;
                double time = 0.0;
                double min_time = std::numeric_limits<double>::infinity();
                double max_time = -min_time;
                for (auto i = 0; i < RUNS; ++i) {
                    prof.tic_cl("Run");
                    y = quantize(x, 1024);
                    auto t = prof.toc("Run");
                    time += t;
                    if (min_time > t) min_time = t;
                    if (max_time < t) max_time = t;
                }
                return std::make_tuple(rc,time/double(RUNS),min_time,max_time);
            });

        run_test("Generate 30-bit morton code",
            [&]() -> std::tuple<bool, double, double, double> {
                const size_t N = 1 << 10;
                int rc = true;
                vex::vector<cl_float4> x(ctx, N);
                vex::vector<uint> y(ctx, N);
                Random<cl_float4> rand0;
                x = rand0(element_index(), rand());
                profiler prof;
                double time = 0.0;
                double min_time = std::numeric_limits<double>::infinity();
                double max_time = -min_time;
                for (auto i = 0; i < RUNS; ++i) {
                    prof.tic_cl("Run");
                    //y = morton_code<cl_uint>::calculate(x);
                    y = morton_code30(quantize(x, 1024));
                    auto t = prof.toc("Run");
                    time += t;
                    if (min_time > t) min_time = t;
                    if (max_time < t) max_time = t;
                }
                return std::make_tuple(rc,time/double(RUNS),min_time,max_time);
            });

        run_test("Generate 60-bit morton code",
            [&]() -> std::tuple<bool, double, double, double> {
                const size_t N = 1 << 10;
                int rc = true;
                vex::vector<cl_float4> x(ctx, N);
                vex::vector<cl_ulong> y(ctx, N);
                Random<cl_float4> rand0;
                x = rand0(element_index(), rand());
                profiler prof;
                double time = 0.0;
                double min_time = std::numeric_limits<double>::infinity();
                double max_time = -min_time;
                for (auto i = 0; i < RUNS; ++i) {
                    prof.tic_cl("Run");
                    //y = morton_code<cl_ulong>::calculate(x);
                    y = morton_code60(quantize(x, 1 << 20));
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


