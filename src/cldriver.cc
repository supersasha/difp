#include <iostream>

#include "cldriver.h"
#include "utils.h"

CLDriver CLDriver::s_cldriver("cl/difp2.c");

CLDriver::CLDriver(const std::string& filename)
{
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    std::cout << "Platforms: \n";
    for(size_t i = 0; i < all_platforms.size(); i++) {
            std::cout<< all_platforms[i].getInfo<CL_PLATFORM_VERSION>()<<"\n";
    }
    
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: " <<
        default_platform.getInfo<CL_PLATFORM_VERSION>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    std::cout << "Devices: \n";
    for(size_t i = 0; i < all_devices.size(); i++) {
            std::cout<< all_devices[i].getInfo<CL_DEVICE_NAME>()<<"\n";
            std::cout<<all_devices[i].getInfo<CL_DEVICE_AVAILABLE>() << "\n";
    }
    cl::Device default_device = all_devices[0];
    std::cout<< "Using device: " << default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

    m_context = cl::Context({ default_device });
    std::cout << "Context created\n";

    m_filename = filename;
    reload();
}

void CLDriver::reload()
{
    m_program = cl::Program(m_context, read_text_file(m_filename), false);
    std::cout << "Program loaded\n";
    try {
        m_program.build("-cl-std=CL2.0"); //"-cl-fast-relaxed-math -cl-std=CL2.0");
        std::cout << "Program built\n";
    } catch (...) {
        auto buildInfo = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for(auto & pair : buildInfo) {
            std::cerr << pair.second << "\n\n";
        }
        throw -1;
    }
}
