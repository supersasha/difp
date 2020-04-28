/*
 * Color conversions
 */
/*
float A_1931_64[31][3] = {
    { 0.0191097, 0.0020044, 0.0860109 },
    { 0.084736, 0.008756, 0.389366 },
    { 0.204492, 0.021391, 0.972542 },
    { 0.314679, 0.038676, 1.55348 },
    { 0.383734, 0.062077, 1.96728 },
    { 0.370702, 0.089456, 1.9948 },
    { 0.302273, 0.128201, 1.74537 },
    { 0.195618, 0.18519, 1.31756 },
    { 0.080507, 0.253589, 0.772125 },
    { 0.016172, 0.339133, 0.415254 },
    { 0.003816, 0.460777, 0.218502 },
    { 0.037465, 0.606741, 0.112044 },
    { 0.117749, 0.761757, 0.060709 },
    { 0.236491, 0.875211, 0.030451 },
    { 0.376772, 0.961988, 0.013676 },
    { 0.529826, 0.991761, 0.003988 },
    { 0.705224, 0.99734, 0 },
    { 0.878655, 0.955552, 0 },
    { 1.01416, 0.868934, 0 },
    { 1.11852, 0.777405, 0 },
    { 1.12399, 0.658341, 0 },
    { 1.03048, 0.527963, 0 },
    { 0.856297, 0.398057, 0 },
    { 0.647467, 0.283493, 0 },
    { 0.431567, 0.179828, 0 },
    { 0.268329, 0.107633, 0 },
    { 0.152568, 0.060281, 0 },
    { 0.0812606, 0.0318004, 0 },
    { 0.0408508, 0.0159051, 0 },
    { 0.0199413, 0.0077488, 0 },
    { 0.00957688, 0.00371774, 0 }
};
*/

float4 xyz_to_srgb_scalar(float4 c)
{
    float x = c.x / 100.0f;
    float y = c.y / 100.0f;
    float z = c.z / 100.0f;

    float r = x *  3.2406f + y * -1.5372f + z * -0.4986f;
    float g = x * -0.9689f + y *  1.8758f + z *  0.0415f;
    float b = x *  0.0557f + y * -0.2040f + z *  1.0570f;

    if(r > 0.0031308f)
        r = 1.055f * native_powr(r, 1.0f / 2.4f) - 0.055f;
    else
        r = 12.92f * r;

    if(g > 0.0031308f)
        g = 1.055f * native_powr(g, 1.0f / 2.4f) - 0.055f;
    else
        g = 12.92f * g;

    if(b > 0.0031308f)
        b = 1.055f * native_powr(b, 1.0f / 2.4f) - 0.055f;
    else
        b = 12.92f * b;
    
    if(r < 0.0f)
        r = 0.0f;
    else if(r > 1.0f)
        r = 1.0f;

    if(g < 0.0f)
        g = 0.0f;
    else if(g > 1.0f)
        g = 1.0f;

    if(b < 0.0f)
        b = 0.0f;
    else if(b > 1.0f)
        b = 1.0f;
    return (float4) (r, g, b, 0.0f);
}

float2 chromaticity(float4 xyz)
{
    float v = xyz.x + xyz.y + xyz.z;
    if (v == 0) {
        return 0.3333f;
    }
    return (float2)(xyz.x/v, xyz.y/v);
}

/*
 * Structures
 */

struct SpectrumData
{
    float wp[2];
    float light[31];
    float base[3][31];
    float tri_to_v_mtx[3][3];
};

struct ProfileData
{
    float film_sense[3][31];
    float film_dyes[3][31];
    float paper_sense[3][31];
    float paper_dyes[3][31];

    float couplers[3][31];
    float proj_light[31];
    float dev_light[31];
    float mtx_refl[3][31];

    float neg_gammas[3];
    float paper_gammas[3];
    float film_max_qs[3];
};

enum ProcessingMode
{
    NORMAL = 0,
    NEGATIVE,
    IDENTITY,
    FILM_EXPOSURE,
    GEN_SPECTR,
    FILM_DEV,
    PAPER_EXPOSURE,
    FILM_NEG_LOG_EXP,
    PAPER_NEG_LOG_EXP
};

struct UserOptions
{
    float color_corr[3];
    float film_exposure;
    float paper_exposure;
    float paper_contrast;
    float curve_smoo;
    //int negative;
    int mode;
    int channel;
    int frame_horz;
    int frame_vert;
};

/*
 * Helpers
 */

__constant int NUM_SECTORS = 6;
__constant int NUM_BASES = NUM_SECTORS + 1;
__constant float BLUE_CHROMA_SEPARATION = 0.16f;
__constant int SPECTRUM_SIZE = 31;
__constant float MIN_REFLECTION = 1.0e-15f;

float sigma_old(float x, float _min, float _max, float gamma, float bias, float smoo)
{
    /* 
     * When bias = 0 it reaches maximum at x = 0 (when not smoothed)
     */
    float a = (_max - _min) / 2.0f;
    float y = gamma * (x + 0.5f/gamma + bias) / a;
    float w = y / pow(1.0f + pow(fabs(y), 1.0f/smoo), smoo);
    return a * (w + 1.0f);
}


float sigma(float x, float ymin, float ymax, float gamma, float bias, float smoo)
{
    float a = (ymax - ymin) / 2;
    float y = gamma * (x - bias) / a;
    return a * (y / pow(1 + pow(fabs(y), 1/smoo), smoo) + 1) + ymin;
}

float sigma_from(float x, float ymin, float ymax, float gamma, float smoo, float x0)
{
    float avg = (ymax + ymin) / 2;

    // gamma * (x0 - bias) + avg = ymin
    
    float bias = x0 - (ymin - avg) / gamma;
    return sigma(x, ymin, ymax, gamma, bias, smoo);
}

float sigma_to(float x, float ymin, float ymax, float gamma, float smoo, float x1)
{
    float avg = (ymax + ymin) / 2;

    // gamma * (x1 - bias) + avg = ymax
    
    float bias = x1 - (ymax - avg) / gamma;
    return sigma(x, ymin, ymax, gamma, bias, smoo);
}

float zigzag(float x, float gamma, float ymax)
{
    if (x >= 0) {
        return ymax;
    }
    float x0 = -ymax/gamma;
    if (x <= x0) {
        return 0;
    }
    return gamma * (x - x0);
}

float zigzag_p(float x, float gamma, float ymax)
{
    if (x < 0) {
        return 0;
    }
    float y = x * gamma;
    if (y > ymax) {
        return ymax;
    }
    return y;
}

float zigzag1(float x, float ymin, float ymax, float gamma, float bias)
{
    if (x <= bias) {
        return ymin;
    }
    float y = ymin + gamma * (x - bias);
    if (y > ymax) {
        return ymax;
    }
    return y;
}

/*
 * Process film
 */
__kernel void process_photo(
    __global float4 * img,
    __global struct SpectrumData * sd,
    __global struct ProfileData * pd,
    __global struct UserOptions * opts,
    __global float4 * out_img
    )
{
    int out_col = get_global_id(0);
    int out_row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    // Frame
    int fr_horz = opts->frame_horz;
    int fr_vert = opts->frame_vert;
    int out_idx = out_row * width + out_col;
    if (out_col < fr_horz || out_col >= width - fr_horz
        || out_row < fr_vert || out_row >= height - fr_vert)
    {
        out_img[out_idx] = (float4) (0.96f, 0.96f, 0.95f, 0);
        return;
    }
    
    // Getting input color XYZ
    int idx = (out_row - fr_vert) * (width - 2 * fr_horz) + (out_col - fr_horz); 
    float4 xyz = img[idx] * 100;

    if (opts->mode == IDENTITY) {
        out_img[out_idx] = xyz_to_srgb_scalar(xyz * pow(10, opts->film_exposure));
        return;
    }
    
    // Spectrum
#define B(i, j) (sd->base[i][j])
#define T(i, j) (sd->tri_to_v_mtx[i][j])
    float4 v = (float4)(
        T(0, 0)*xyz.x + T(0, 1)*xyz.y + T(0, 2)*xyz.z,
        T(1, 0)*xyz.x + T(1, 1)*xyz.y + T(1, 2)*xyz.z,
        T(2, 0)*xyz.x + T(2, 1)*xyz.y + T(2, 2)*xyz.z,
        0
    );

    float4 zzz = 0;
    // Film development
    float4 exposure = (float4)(0, 0, 0, 1);
    for (int i = 0; i < SPECTRUM_SIZE; i++) {
        float refl = B(0, i)*v.x + B(1, i)*v.y + B(2, i)*v.z;
        if (refl < MIN_REFLECTION) {
            refl = MIN_REFLECTION;
        } else if (refl > 1) {
            refl = 1;
        }
        float sp = refl * pd->dev_light[i]; //sd->light[i];
        
        if (opts->mode == GEN_SPECTR) {
            zzz.x += pd->mtx_refl[0][i] * refl;
            zzz.y += pd->mtx_refl[1][i] * refl;
            zzz.z += pd->mtx_refl[2][i] * refl;
        }

        exposure.x += pow(10, pd->film_sense[0][i] - 4.149f - opts->color_corr[0]) * sp;
        exposure.y += pow(10, pd->film_sense[1][i] - 5.997f - opts->color_corr[1]) * sp;
        exposure.z += pow(10, pd->film_sense[2][i] - 1.309f - opts->color_corr[2]) * sp;
    }

    if (opts->mode == GEN_SPECTR) {
        out_img[out_idx] = xyz_to_srgb_scalar(zzz);
        return;
    }

    if (opts->mode == FILM_EXPOSURE) {
        out_img[out_idx] = exposure * pow(10, opts->film_exposure);
        return;
    }

    float4 H = log10(exposure) + opts->film_exposure;

    if (opts->mode == FILM_NEG_LOG_EXP) {
        out_img[out_idx] = (float4) (
            H.x < 0 ? 1 : 0,
            H.y < 0 ? 1 : 0,
            H.z < 0 ? 1 : 0,
            0
        );
        return;
    }

    float4 dev = (float4) (
            /*
        sigma_to(H.x, 0, 1, pd->neg_gammas[0], opts->curve_smoo, 0),
        sigma_to(H.y, 0, 1, pd->neg_gammas[1], opts->curve_smoo, 0),
        sigma_to(H.z, 0, 1, pd->neg_gammas[2], opts->curve_smoo, 0),
        */
        sigma_to(H.x, 0, 4, 0.812f, /*opts->curve_smoo*/0.02f, 0),
        sigma_to(H.y, 0, 4, 0.533f, /*opts->curve_smoo*/0.02f, 0),
        sigma_to(H.z, 0, 4, 0.540f, /*opts->curve_smoo*/0.02f, 0),
        0
    );

    if (opts->mode == FILM_DEV) {
        out_img[out_idx] = dev;
        return;
    }

    float4 xyz1 = 0;

    // Paper development
    exposure = (float4)(0, 0, 0, 1);
    for (int i = 0; i < SPECTRUM_SIZE; i++) {
        float developed_dyes = pd->film_dyes[0][i] * dev.x
                             + pd->film_dyes[1][i] * dev.y
                             + pd->film_dyes[2][i] * dev.z
                             ;
        float developed_couplers = pd->couplers[0][i] * (1.0f - dev.x / 4.0f)
                                 + pd->couplers[1][i] * (1.0f - dev.y / 4.0f)
                                 + pd->couplers[2][i] * (1.0f - dev.z / 4.0f);
        float developed = developed_dyes + developed_couplers;
        float trans = pow(10, -developed);
        if (opts->mode == NEGATIVE) {
            xyz1.x += pd->mtx_refl[0][i] * trans * 1000000;
            xyz1.y += pd->mtx_refl[1][i] * trans * 1000000;
            xyz1.z += pd->mtx_refl[2][i] * trans * 1000000;
        } else {
            float sp = trans * pd->proj_light[i];
            exposure.x += pow(10, pd->paper_sense[0][i] - 0.995f) * sp;
            exposure.y += pow(10, pd->paper_sense[1][i] - 0.373f) * sp;
            exposure.z += pow(10, pd->paper_sense[2][i] + 1.420f) * sp;
        }
    }
    
    if (opts->mode == PAPER_EXPOSURE) {
        out_img[out_idx] = exposure * pow(10, opts->paper_exposure) / 500;
        return;
    }
    
    if (opts->mode != NEGATIVE) {
        H = log10(exposure) + opts->paper_exposure;
    
        if (opts->mode == PAPER_NEG_LOG_EXP) {
            out_img[out_idx] = (float4) (
                ((H.x < 0) ? 1 : 0),
                ((H.y < 0) ? 1 : 0),
                ((H.z < 0) ? 1 : 0),
                0
            );
            return;
        }

        // Viewing paper
        for (int i = 0; i < SPECTRUM_SIZE; i++) {
#define SIGMA 4
#if SIGMA == 0
            float r = H.x * pd->paper_gammas[0] * opts->paper_contrast;
            float g = H.y * pd->paper_gammas[1] * opts->paper_contrast;
            float b = H.z * pd->paper_gammas[2] * opts->paper_contrast;
#elif SIGMA == 1
            //float r = H.x * pd->paper_gammas[0] * opts->paper_contrast;
            float r = sigma_from(H.x, 0, 3,
                pd->paper_gammas[0] * opts->paper_contrast, opts->curve_smoo, 0);
            float g = sigma_from(H.y, 0, 3,
                pd->paper_gammas[1] * opts->paper_contrast, opts->curve_smoo, 0);
            float b = sigma_from(H.z, 0, 3,
                pd->paper_gammas[2] * opts->paper_contrast, opts->curve_smoo, 0);
#elif SIGMA == 2
            float r = zigzag_p(H.x, pd->paper_gammas[0] * opts->paper_contrast, 4);
            float g = zigzag_p(H.y, pd->paper_gammas[1] * opts->paper_contrast, 4);
            float b = zigzag_p(H.z, pd->paper_gammas[2] * opts->paper_contrast, 4);
#elif SIGMA == 3
            float r = sigma_old(H.x, 0, 3, opts->paper_contrast * pd->paper_gammas[0],
                                -0.5, opts->curve_smoo);
            float g = sigma_old(H.y, 0, 3, opts->paper_contrast * pd->paper_gammas[1],
                                -0.5, opts->curve_smoo);
            float b = sigma_old(H.z, 0, 3, opts->paper_contrast * pd->paper_gammas[2],
                                -0.5, opts->curve_smoo);
#elif SIGMA == 4
            float r = sigma_from(H.x, 0, 4, 5.0f * opts->paper_contrast, opts->curve_smoo, 0);
            float g = sigma_from(H.y, 0, 4, 5.0f * opts->paper_contrast, opts->curve_smoo, 0);
            float b = sigma_from(H.z, 0, 4, 5.0f * opts->paper_contrast, opts->curve_smoo, 0);
#endif
            float developed = pd->paper_dyes[0][i] * r
                            + pd->paper_dyes[1][i] * g
                            + pd->paper_dyes[2][i] * b;
            float trans = pow(10, -developed /* * opts->paper_contrast*/);
            xyz1.x += pd->mtx_refl[0][i] * trans;
            xyz1.y += pd->mtx_refl[1][i] * trans;
            xyz1.z += pd->mtx_refl[2][i] * trans;
        }
    }
    
    // Setting output color sRGB
    float4 c = xyz_to_srgb_scalar(xyz1 * 0.966f);
    out_img[out_idx] = c;
}

