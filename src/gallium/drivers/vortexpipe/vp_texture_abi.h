/* SPDX-License-Identifier: MIT */
#ifndef VP_TEXTURE_ABI_H
#define VP_TEXTURE_ABI_H
#include <stdint.h>

#define VP_TEXTURE_LEVELS 16
enum vp_texture_format {
   VP_TEXTURE_RGBA8_UNORM = 1,
   VP_TEXTURE_BGRA8_UNORM,
   VP_TEXTURE_RGBA8_SRGB,
   VP_TEXTURE_BGRA8_SRGB,
   VP_TEXTURE_RGBA32_FLOAT,
   VP_TEXTURE_R32_FLOAT,
};
enum vp_texture_wrap {
   VP_TEXTURE_REPEAT,
   VP_TEXTURE_MIRROR_REPEAT,
   VP_TEXTURE_CLAMP_EDGE,
   VP_TEXTURE_CLAMP_BORDER,
};
enum vp_texture_filter { VP_TEXTURE_NEAREST, VP_TEXTURE_LINEAR };
enum vp_texture_mip_filter {
   VP_TEXTURE_MIP_NONE,
   VP_TEXTURE_MIP_NEAREST,
   VP_TEXTURE_MIP_LINEAR,
};

/* Device addresses and byte strides; no host pointers or compiler objects. */
struct vp_texture_descriptor {
   uint64_t base;
   uint32_t width, height;
   uint32_t format, first_level, last_level, swizzle;
   uint32_t row_stride[VP_TEXTURE_LEVELS];
   uint32_t mip_offset[VP_TEXTURE_LEVELS];
};
struct vp_sampler_descriptor {
   uint32_t wrap_s, wrap_t, min_filter, mag_filter;
   uint32_t mip_filter, normalized_coords;
   float min_lod, max_lod, lod_bias, border_color[4];
};
/* Separate texture and sampler sources can address different descriptor slots. */
struct vp_sampled_descriptor {
   struct vp_texture_descriptor texture;
   struct vp_sampler_descriptor sampler;
};
#endif
