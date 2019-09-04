#include "film.h"
#include "cldriver.h"

void process_photo(Image& img, const PhotoProcessOpts& opts)
{
    const auto& drv = CLDriver::get();
    size_t len = img.width * img.height;

    cl::Buffer img_buf(drv.context(), img.data,
                     img.data + len, false);
    cl::Buffer opts_buf(drv.context(),
        const_cast<PhotoProcessOpts*>(&opts),
        const_cast<PhotoProcessOpts*>(&opts + 1), true);
    cl::CommandQueue queue(drv.context());
        
    cl::KernelFunctor<cl::Buffer, cl::Buffer>
        kf(drv.program(), "process_photo");
    /*
    cl::KernelFunctor<cl::Buffer, cl::Buffer>
        kf2(drv.program(), "process_photo_cont");
    */

    auto args = cl::EnqueueArgs(queue, cl::NDRange(len));
    kf(args, img_buf, opts_buf);
    cl::copy(queue, img_buf, img.data, img.data + len);
    
    /* 
    if (opts.extra.cy > 0) { 
        auto ch = Channel::from_image(img, 3);
        ch = ch.gaussian_fft(opts.extra.cy);
        //ch = ch.compress_range(opts.extra.cy);
        ch.into_image(img, 3);
    }
    */

    /*
    cl::Buffer img_buf2(drv.context(), img.data, img.data + len, false);
    kf2(args, img_buf2, opts_buf);

    cl::copy(queue, img_buf2, img.data, img.data + len);
    */
}
