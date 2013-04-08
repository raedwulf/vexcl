#include <iostream>
#include <vector>
#include <string>
#include <vexcl/devlist.hpp>
#include <vexcl/util.hpp>
#include <boost/algorithm/string.hpp>
using namespace vex;

int main() {
    std::cout << "OpenCL devices:" << std::endl << std::endl;
    auto dev = device_list(Filter::All);
    for (auto d = dev.begin(); d != dev.end(); d++) {
        // Prettify some of the text
        typedef std::vector<std::string> split_vector_type;
        auto defines = generate_platform_defines(*d);
        auto flags = generate_platform_options(*d); flags.erase(0, 3);
        std::string pretty_defines, pretty_flags;
        split_vector_type define_lines, flag_lines;

        boost::split(define_lines, defines, boost::is_from_range('\n','\n'));
        for (auto l : define_lines)
            pretty_defines += std::string("        ") + l + "\n";

        boost::iter_split(flag_lines, flags, boost::first_finder("-D"));
        for (auto l : flag_lines)
            pretty_flags += std::string("        -D") + l + "\n";

        std::cout << "  " << d->getInfo<CL_DEVICE_NAME>() << std::endl
                  << "    CL_PLATFORM_NAME              = " << cl::Platform(d->getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>() << std::endl
                  << "    CL_DEVICE_MAX_COMPUTE_UNITS   = " << d->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl
                  << "    CL_DEVICE_HOST_UNIFIED_MEMORY = " << d->getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>() << std::endl
                  << "    CL_DEVICE_GLOBAL_MEM_SIZE     = " << d->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl
                  << "    CL_DEVICE_LOCAL_MEM_SIZE      = " << d->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl
                  << "    CL_DEVICE_MAX_MEM_ALLOC_SIZE  = " << d->getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl
                  << "    CL_DEVICE_MAX_CLOCK_FREQUENCY = " << d->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl
                  << "    Defines:          " << std::endl << pretty_defines
                  << "    Compiler Options:"  << std::endl << pretty_flags
                  << std::endl;
    }
}

// vim: et
