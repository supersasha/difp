/*
 * Color conversions
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
    float sectors[6][2];
    float light[31];
    float bases[7][3][31];
    float tri_to_v_mtx[7][3][3];
};

struct ProfileData
{
    float film_sense[3][31];
    float film_dyes[3][31];
    float paper_sense[3][31];
    float paper_dyes[3][31];

    float couplers[3][31];
    float proj_light[31];
    float mtx_refl[3][31];

    float neg_gammas[3];
    float paper_gammas[3];
    float film_max_qs[3];
};

struct UserOptions
{
    float color_corr[3];
    float film_exposure;
    float paper_exposure;
    float paper_contrast;
    float curve_smoo;
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

bool in_sector(int sec_num, float2 xy, struct SpectrumData * sd) {
    float * v1 = sd->sectors[sec_num];
    float * v2 = sd->sectors[sec_num+1];
    float d = v1[0]*v2[1] - v2[0]*v1[1];
    float q1 = (xy.x*v2[1] - xy.y*v2[0]) / d;
    float q2 = (-xy.x*v1[1] + xy.y*v1[0]) / d;
    return q1 > 0 && q2 > 0;
}

int find_sector(float2 xy0, struct SpectrumData * sd)
{
    float2 wp = (float2)(sd->wp[0], sd->wp[1]);
    float2 xy = xy0 - wp;
    for (int i = 0; i < NUM_SECTORS - 1; i++) {
        if (in_sector(i, xy, sd)) {
            if (i == 0 && hypot(xy.x, xy.y) < BLUE_CHROMA_SEPARATION) {
                return NUM_SECTORS;
            }
            return i;
        }
    }
    return NUM_SECTORS - 1;
}

float sigma(float x, float _min, float _max, float gamma, float bias, float smoo)
{
    /* 
     * When bias = 0 it reaches maximum at x = 0 (when not smoothed)
     */
    //float smoo = 0.2f;
    float a = (_max - _min) / 2.0f;
    float y = gamma * (x + 0.5f/gamma + bias) / a;
    float w = y / pow(1.0f + pow(fabs(y), 1.0f/smoo), smoo);
    return a * (w + 1.0f);
}

float zigzag(float x, float gamma, float ymax)
{
    /*
    if (x >= 0) {
        return ymax;
    }
    float x0 = -ymax/gamma;
    if (x <= x0) {
        return 0;
    }
    return gamma * (x - x0);
    */
    return sigma(x, 0, ymax, gamma, 0, 0.2);
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

    // Spectrum
    float2 xy = chromaticity(xyz);
    int sector = find_sector(xy, sd);
    //float basis[3][31] = sd->bases[sector];
    //float t[3][3] = sd->tri_to_v_mtx[sector];
#define B(i, j) (sd->bases[sector][i][j])
#define T(i, j) (sd->tri_to_v_mtx[sector][i][j])
    float4 v = (float4)(
        T(0, 0)*xyz.x + T(0, 1)*xyz.y + T(0, 2)*xyz.z,
        T(1, 0)*xyz.x + T(1, 1)*xyz.y + T(1, 2)*xyz.z,
        T(2, 0)*xyz.x + T(2, 1)*xyz.y + T(2, 2)*xyz.z,
        0
    );

    // Film development
    float4 exposure = (float4)(0, 0, 0, 1);
    for (int i = 0; i < SPECTRUM_SIZE; i++) {
        float refl = B(0, i)*v.x + B(1, i)*v.y + B(2, i)*v.z;
        if (refl < MIN_REFLECTION) {
            refl = MIN_REFLECTION;
        } else if (refl > 1) {
            refl = 1;
        }
        float sp = refl * sd->light[i];
        exposure.x += pow(10, pd->film_sense[0][i] - opts->color_corr[0]) * sp;
        exposure.y += pow(10, pd->film_sense[1][i] - opts->color_corr[1]) * sp;
        exposure.z += pow(10, pd->film_sense[2][i] - opts->color_corr[2]) * sp;
    }
    float4 H = log10(exposure) + opts->film_exposure;
    float4 dev = (float4) (
        sigma(H.x, 0, 1.0f, pd->neg_gammas[0], 0, opts->curve_smoo),
        sigma(H.y, 0, 1.0f, pd->neg_gammas[1], 0, opts->curve_smoo),
        sigma(H.z, 0, 1.0f, pd->neg_gammas[2], 0, opts->curve_smoo),
        0
    );

    // Paper development
    exposure = (float4)(0, 0, 0, 1);
    for (int i = 0; i < SPECTRUM_SIZE; i++) {
        float developed_dyes = pd->film_dyes[0][i] * dev.x
                             + pd->film_dyes[1][i] * dev.y
                             + pd->film_dyes[2][i] * dev.z;
        float developed_couplers = pd->couplers[0][i] * (1.0f - dev.x)
                                 + pd->couplers[1][i] * (1.0f - dev.y)
                                 + pd->couplers[2][i] * (1.0f - dev.z);
        float developed = developed_dyes + developed_couplers;
        float sp = pow(10, -developed) * pd->proj_light[i];
        exposure.x += pow(10, pd->paper_sense[0][i]/* - opts->color_corr[0]*/) * sp;
        exposure.y += pow(10, pd->paper_sense[1][i]/* - opts->color_corr[1]*/) * sp;
        exposure.z += pow(10, pd->paper_sense[2][i]/* - opts->color_corr[2]*/) * sp;
    }
    
    H = log10(exposure) + opts->paper_exposure;

    // Viewing paper
    float4 xyz1 = 0;
    for (int i = 0; i < SPECTRUM_SIZE; i++) {
        /*
        float r = H.x * pd->paper_gammas[0];
        float g = H.y * pd->paper_gammas[1];
        float b = H.z * pd->paper_gammas[2];
        */
        float r = sigma(H.x, 0, 3, opts->paper_contrast * pd->paper_gammas[0],
                            -0.5, opts->curve_smoo);
        float g = sigma(H.y, 0, 3, opts->paper_contrast * pd->paper_gammas[1],
                            -0.5, opts->curve_smoo);
        float b = sigma(H.z, 0, 3, opts->paper_contrast * pd->paper_gammas[2],
                            -0.5, opts->curve_smoo);
        float developed = pd->paper_dyes[0][i] * r
                        + pd->paper_dyes[1][i] * g
                        + pd->paper_dyes[2][i] * b;
        float trans = pow(10, -developed);
        xyz1.x += pd->mtx_refl[0][i] * trans;
        xyz1.y += pd->mtx_refl[1][i] * trans;
        xyz1.z += pd->mtx_refl[2][i] * trans;
    }
    
    // Setting output color sRGB
    float4 c = xyz_to_srgb_scalar(xyz1);
    out_img[out_idx] = c;
}

/*
__kernel void process_photo(
    __global float4 * img,
    __global struct PhotoProcessOpts * opts,
    __global float4 * out_img
    )
{
    int out_col = get_global_id(0);
    int out_row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    int fr_horz = opts->extra.frame_horz;
    int fr_vert = opts->extra.frame_vert;
    
    int out_idx = out_row * width + out_col;

    if (out_col < fr_horz || out_col >= width - fr_horz
        || out_row < fr_vert || out_row >= height - fr_vert)
    {
        out_img[out_idx] = (float4) (1, 1, 1, 0);
        return;
    }

    int idx = (out_row - fr_vert) * (width - 2 * fr_horz) + (out_col - fr_horz);

    int es = opts->extra.stop;
    
    float4 px = img[idx];
    float4 lrgb = px;
    float4 exposure = rgb_to_exposure_debug(lrgb, opts, idx);

    float4 density = exposure_to_density(exposure, opts, 1.0f, 0);

    if (opts->extra.pixel == idx) {
        opts->debug.xyz_in[0] = px.x;
        opts->debug.xyz_in[1] = px.y;
        opts->debug.xyz_in[2] = px.z;

        opts->debug.film_density[0] = density.x;
        opts->debug.film_density[1] = density.y;
        opts->debug.film_density[2] = density.z;
    }

    exposure = (float4)(0, 0, 0, 0);
    float4 xyz = (float4)(0, 0, 0, 0);
    float4 z = (float4)(0, 0, 0, 0);
    float4 cf1 = dye_qty(&opts->film, density);

    float4 cf2 = cf1;
    float j = 0.7f;
    float4 cf = j*cf1 + (1-j)*cf2;
    if (opts->extra.pixel == idx) {
        opts->debug.film_tdensity[0] = cf.x;
        opts->debug.film_tdensity[1] = cf.y;
        opts->debug.film_tdensity[2] = cf.z;
    }
    for(int i = 0; i < SPECTRUM_SIZE; i++) {
        float lt;
        if (opts->extra.stop) {
            lt = opts->illuminant2.v[i] * pow(10.0f, opts->extra.light_through_film);
        } else {
            lt = opts->illuminant1.v[i];
        }

        if (opts->extra.pixel == idx) {
            opts->debug.film_fall_spectrum[i] = lt;
        }
        float rr = opts->film.red.dye.v[i];
        float gg = opts->film.green.dye.v[i];
        float bb = opts->film.blue.dye.v[i];

        lt /= pow(10.0f,
              rr*(cf.x) 
            + gg*(cf.y)
            + bb*(cf.z));
        if (opts->extra.pixel == idx) {
            opts->debug.film_pass_spectrum[i] = lt;
        }

        if (!opts->extra.stop) {
            exposure.x += lt * opts->paper.red.sense.v[i];
            exposure.y += lt * opts->paper.green.sense.v[i];
            exposure.z += lt * opts->paper.blue.sense.v[i];
            if (opts->extra.pixel == idx) {
                opts->debug.film_fltr_spectrum[i] = lt;
            }
        } else {
            xyz.x += lt * A1931_78[i][0];
            xyz.y += lt * A1931_78[i][1];
            xyz.z += lt * A1931_78[i][2];
        }
    }
    if (opts->extra.stop) {
        float4 c = xyz_to_srgb_scalar(xyz);
        out_img[out_idx] = c;
        return;
    }
    density = opts->extra.paper_contrast * exposure_to_density_paper(exposure, opts, 1.0f);
    float4 cp = dye_qty(&opts->paper, density);
    float4 cpz = dye_qty(&opts->paper, z);

    if (opts->extra.pixel == idx) {
        opts->debug.paper_exposure[0] = exposure.x;
        opts->debug.paper_exposure[1] = exposure.y;
        opts->debug.paper_exposure[2] = exposure.z;

        opts->debug.paper_density[0] = density.x;
        opts->debug.paper_density[1] = density.y;
        opts->debug.paper_density[2] = density.z;

        opts->debug.paper_tdensity[0] = cp.x;
        opts->debug.paper_tdensity[1] = cp.y;
        opts->debug.paper_tdensity[2] = cp.z;
    }

    for(int i = 0; i < SPECTRUM_SIZE; i++) {
        float lt = 
                   opts->illuminant2.v[i] *
                   pow(10.0f, opts->extra.light_on_paper);
        if (opts->extra.pixel == idx) {
            opts->debug.paper_fall_spectrum[i] = lt;
        }
        float rr = opts->paper.red.dye.v[i];
        float gg = opts->paper.green.dye.v[i];
        float bb = opts->paper.blue.dye.v[i];

        lt /= pow(10, ((cp.x) * rr + (cp.y) * gg + (cp.z) * bb));
        lt *= pow(10.0f,
                rr * opts->extra.paper_filter[0] +
                gg * opts->extra.paper_filter[1] +
                bb * opts->extra.paper_filter[2]);
    
        if (opts->extra.pixel == idx) {
            opts->debug.paper_refl_spectrum[i] = lt;
        }
        
        xyz.x += lt * A1931_78[i][0];
        xyz.y += lt * A1931_78[i][1];
        xyz.z += lt * A1931_78[i][2];
    }
    float4 c = xyz_to_srgb_scalar(xyz * pow(10.0f, opts->extra.linear_amp));
    if (opts->extra.pixel == idx) {
        opts->debug.xyz_out[0] = xyz.x;
        opts->debug.xyz_out[1] = xyz.y;
        opts->debug.xyz_out[2] = xyz.z;

        opts->debug.srgb_out[0] = c.x;
        opts->debug.srgb_out[1] = c.y;
        opts->debug.srgb_out[2] = c.z;
    }
    out_img[out_idx] = c;
}
*/
