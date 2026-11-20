/* SPDX-License-Identifier: MIT */
#ifndef VP_SCENE_H
#define VP_SCENE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
struct vp_as_ctx;
uint64_t vp_as_resident_address(struct vp_as_ctx *ctx, const void *source, uint8_t *bytes, uint32_t size);
uint8_t *vp_build_scene(struct vp_as_ctx *ctx, const void *acceleration_structure, uint32_t *size);
#ifdef __cplusplus
}
#endif

#endif
