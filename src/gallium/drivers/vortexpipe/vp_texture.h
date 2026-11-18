/* SPDX-License-Identifier: MIT */
#ifndef VP_TEXTURE_H
#define VP_TEXTURE_H
#include "vp_texture_abi.h"
#include "gallivm/lp_bld_jit_types.h"
#include "gallivm/lp_bld_sample.h"
#include "pipe/p_defines.h"
#include <stdbool.h>
#include <string.h>

static bool
vp_texture_wrap_mode(unsigned mode, uint32_t *out)
{
   switch (mode) {
   case PIPE_TEX_WRAP_REPEAT: *out = VP_TEXTURE_REPEAT; return true;
   case PIPE_TEX_WRAP_MIRROR_REPEAT: *out = VP_TEXTURE_MIRROR_REPEAT; return true;
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE: *out = VP_TEXTURE_CLAMP_EDGE; return true;
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER: *out = VP_TEXTURE_CLAMP_BORDER; return true;
   default: return false;
   }
}

static bool
vp_sampled_descriptor_init(struct vp_sampled_descriptor *dst,
                          const struct lp_descriptor *src,
                          const struct lp_static_texture_state *texture,
                          const struct lp_static_sampler_state *sampler)
{
   if (texture->target != PIPE_TEXTURE_2D || texture->tiled ||
       src->texture.first_level > src->texture.last_level ||
       src->texture.last_level >= VP_TEXTURE_LEVELS ||
       sampler->compare_mode || sampler->reduction_mode != PIPE_TEX_REDUCTION_WEIGHTED_AVERAGE)
      return false;
   memset(dst, 0, sizeof(*dst));
   switch (texture->format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM: dst->texture.format = VP_TEXTURE_RGBA8_UNORM; break;
   case PIPE_FORMAT_B8G8R8A8_UNORM: dst->texture.format = VP_TEXTURE_BGRA8_UNORM; break;
   case PIPE_FORMAT_R8G8B8A8_SRGB: dst->texture.format = VP_TEXTURE_RGBA8_SRGB; break;
   case PIPE_FORMAT_B8G8R8A8_SRGB: dst->texture.format = VP_TEXTURE_BGRA8_SRGB; break;
   case PIPE_FORMAT_R32G32B32A32_FLOAT: dst->texture.format = VP_TEXTURE_RGBA32_FLOAT; break;
   case PIPE_FORMAT_R32_FLOAT: dst->texture.format = VP_TEXTURE_R32_FLOAT; break;
   default: return false;
   }
   if (!src->texture.width || !src->texture.height || src->texture.depth > 1)
      return false;
   dst->texture.width = src->texture.width;
   dst->texture.height = src->texture.height;
   dst->texture.first_level = src->texture.first_level;
   dst->texture.last_level = src->texture.last_level;
   dst->texture.swizzle = texture->swizzle_r | (texture->swizzle_g << 3) |
                         (texture->swizzle_b << 6) | (texture->swizzle_a << 9);
   memcpy(dst->texture.row_stride, src->texture.row_stride, sizeof(dst->texture.row_stride));
   memcpy(dst->texture.mip_offset, src->texture.mip_offsets, sizeof(dst->texture.mip_offset));
   if (!vp_texture_wrap_mode(sampler->wrap_s, &dst->sampler.wrap_s) ||
       !vp_texture_wrap_mode(sampler->wrap_t, &dst->sampler.wrap_t))
      return false;
   if (sampler->min_img_filter > PIPE_TEX_FILTER_LINEAR ||
       sampler->mag_img_filter > PIPE_TEX_FILTER_LINEAR)
      return false;
   dst->sampler.min_filter = sampler->min_img_filter == PIPE_TEX_FILTER_LINEAR ?
                            VP_TEXTURE_LINEAR : VP_TEXTURE_NEAREST;
   dst->sampler.mag_filter = sampler->mag_img_filter == PIPE_TEX_FILTER_LINEAR ?
                            VP_TEXTURE_LINEAR : VP_TEXTURE_NEAREST;
   switch (sampler->min_mip_filter) {
   case PIPE_TEX_MIPFILTER_NONE: dst->sampler.mip_filter = VP_TEXTURE_MIP_NONE; break;
   case PIPE_TEX_MIPFILTER_NEAREST: dst->sampler.mip_filter = VP_TEXTURE_MIP_NEAREST; break;
   case PIPE_TEX_MIPFILTER_LINEAR: dst->sampler.mip_filter = VP_TEXTURE_MIP_LINEAR; break;
   default: return false;
   }
   dst->sampler.normalized_coords = sampler->normalized_coords;
   dst->sampler.min_lod = src->sampler.min_lod;
   dst->sampler.max_lod = src->sampler.max_lod;
   dst->sampler.lod_bias = src->sampler.lod_bias;
   memcpy(dst->sampler.border_color, src->sampler.border_color, sizeof(dst->sampler.border_color));
   return true;
}

static bool
vp_sampled_texture_bytes(const struct lp_descriptor *src, uint32_t *out)
{
   if (src->texture.first_level > src->texture.last_level ||
       src->texture.last_level >= VP_TEXTURE_LEVELS)
      return false;
   uint64_t bytes = 0;
   for (unsigned level = src->texture.first_level; level <= src->texture.last_level; ++level) {
      uint32_t height = src->texture.height >> level;
      if (!height)
         height = 1;
      uint64_t end = (uint64_t)src->texture.mip_offsets[level] +
                     (uint64_t)height * src->texture.row_stride[level];
      if (!src->texture.row_stride[level] || end > UINT32_MAX)
         return false;
      if (end > bytes)
         bytes = end;
   }
   *out = (uint32_t)bytes;
   return bytes != 0;
}
#endif
