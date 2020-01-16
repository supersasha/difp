#include "film.h"
#include "cldriver.h"

Image process_photo_old(const Image& img, PhotoProcessOpts& opts)
{
    const auto& drv = CLDriver::get();
    size_t len = img.width * img.height;
    int out_width = img.width + 2 * opts.extra.frame_horz;
    int out_height = img.height + 2 * opts.extra.frame_vert;
    size_t out_len = out_width * out_height;
    Image out_img(out_width, out_height);

    cl::Buffer img_buf(drv.context(), img.data,
                     img.data + len, false);
    cl::Buffer out_img_buf(drv.context(), CL_MEM_WRITE_ONLY,
                            out_len * sizeof(Color));
    cl::Buffer opts_buf(drv.context(),
        const_cast<PhotoProcessOpts*>(&opts),
        const_cast<PhotoProcessOpts*>(&opts + 1), false);
    cl::CommandQueue queue(drv.context());
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>
        kf(drv.program(), "process_photo");
    auto args = cl::EnqueueArgs(queue, cl::NDRange(out_width, out_height));
    kf(args, img_buf, opts_buf, out_img_buf);
    cl::copy(queue, out_img_buf, out_img.data, out_img.data + out_len);
    cl::copy(queue, opts_buf, &opts, &opts + 1);
    return out_img;
}

Image process_photo(const Image& img, SpectrumData& sd, ProfileData& pd, UserOptions& opts)
{
    const auto& drv = CLDriver::get();
    size_t len = img.width * img.height;
    int out_width = img.width + 2 * opts.frame_horz;
    int out_height = img.height + 2 * opts.frame_vert;
    size_t out_len = out_width * out_height;
    Image out_img(out_width, out_height);

    cl::Buffer img_buf(drv.context(), img.data,
                     img.data + len, false);
    cl::Buffer out_img_buf(drv.context(), CL_MEM_WRITE_ONLY,
                            out_len * sizeof(Color));
    cl::Buffer sd_buf(drv.context(), &sd, &sd + 1, /*readonly*/true);
    cl::Buffer pd_buf(drv.context(), &pd, &pd + 1, true);
    cl::Buffer op_buf(drv.context(), &opts, &opts + 1, true);
    cl::CommandQueue queue(drv.context());
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
        kf(drv.program(), "process_photo");
    auto args = cl::EnqueueArgs(queue, cl::NDRange(out_width, out_height));
    kf(args, img_buf, sd_buf, pd_buf, op_buf, out_img_buf);
    cl::copy(queue, out_img_buf, out_img.data, out_img.data + out_len);
    //cl::copy(queue, opts_buf, &opts, &opts + 1);
    return out_img;
}
