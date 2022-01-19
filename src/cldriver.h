#ifndef __CL_DRIVER_H
#define __CL_DRIVER_H

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
//200
#define CL_HPP_TARGET_OPENCL_VERSION 120
//200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

class CLDriver
{
public:
    CLDriver(const std::string& filename);

    cl::Context& context() { return m_context; }
    const cl::Context& context() const { return m_context; }
    
    cl::Program& program() { return m_program; }
    const cl::Program& program() const { return m_program; }

    static CLDriver& get() { return s_cldriver; }

    void reload();
private:
    std::string m_filename;
    cl::Context m_context;
    cl::Program m_program;

    static CLDriver s_cldriver;
};

#endif
