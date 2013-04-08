// Reference implementation from Slash Sandbox
// http://code.google.com/p/slash-sandbox

#include <iostream>
#include <random>
#include <iterator>
#include <cassert>
#include <chrono>

#include <vexcl/vexcl.hpp>

#define TESTS_ON

using namespace vex;

static bool all_passed = true;

bool run_test(const std::string &name, std::function<int()> test) {
    char fc = std::cout.fill('.');
    std::cout << name << ": " << std::setw(62 - name.size()) << "." << std::flush;
    std::cout.fill(fc);

    int rc = test();
    all_passed = all_passed && (rc >= 0);
    std::cout << (rc >= 0 ? " success." : " failed.") <<
        " [" << (rc&0x7fffffff) << "us]" << std::endl;
    return rc;
}

#define QUANTIZE "#define QUANTIZE(x,n) " \
                 "(uint)(max(min((int)(x*(float)(n)),(int)(n-1)),(int)(0)))\n"
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

VEX_FUNCTION(morton_code30, uint(cl_float3),
             QUANTIZE MORTON30
             "uint out;"
             "uint x = QUANTIZE(prm1.x, 1024u);"
             "uint y = QUANTIZE(prm1.y, 1024u);"
             "uint z = QUANTIZE(prm1.z, 1024u);"
             "MORTON30(x, y, z, out);"
             "return out;");

VEX_FUNCTION(morton_code60, cl_ulong(cl_float3),
             QUANTIZE MORTON30
             "uint x = QUANTIZE(prm1.x, 1u << 20);"
             "uint y = QUANTIZE(prm1.y, 1u << 20);"
             "uint z = QUANTIZE(prm1.z, 1u << 20);"
             "uint lo_x = x & 1023u;"
             "uint lo_y = y & 1023u;"
             "uint lo_z = z & 1023u;"
             "uint hi_x = x >> 10u;"
             "uint hi_y = y >> 10u;"
             "uint hi_z = z >> 10u;"
             "uint lo, hi;"
             "MORTON30(lo_x, lo_y, lo_z, lo);"
             "MORTON30(hi_x, hi_y, hi_z, hi);"
             "return ((ulong)hi << 30) | (ulong)lo;");

template <typename morton_type>
struct morton_code { };
template <>
struct morton_code<cl_uint> {
    template <typename T>
    static decltype(morton_code30(T())) calculate(T p) {
        return morton_code30(p);
    }
};
template <>
struct morton_code<cl_ulong> {
    template <typename T>
    static decltype(morton_code60(T())) calculate(T p) {
        return morton_code60(p);
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

        using namespace std::chrono;

        run_test("Generate 30-bit morton code", [&]() -> int {
                const size_t N = 1 << 28;
                int rc = 0;
                vex::vector<cl_float3> x(ctx, N);
                vex::vector<uint> y(ctx, N);
                Random<cl_float3> rand0;
                x = rand0(element_index(), rand());
                auto start = high_resolution_clock::now();
                y = morton_code<cl_uint>::calculate(x);
                auto end = high_resolution_clock::now();
                rc = duration_cast<microseconds>(end - start).count();
                return rc;
            });

        run_test("Generate 60-bit morton code", [&]() -> int {
                const size_t N = 1 << 28;
                int rc = 0;
                vex::vector<cl_float3> x(ctx, N);
                vex::vector<cl_ulong> y(ctx, N);
                Random<cl_float3> rand0;
                x = rand0(element_index(), rand());
                auto start = high_resolution_clock::now();
                y = morton_code<cl_ulong>::calculate(x);
                auto end = high_resolution_clock::now();
                rc = duration_cast<microseconds>(end - start).count();
                return rc;
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


