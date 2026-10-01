/* SPDX-License-Identifier: MIT */
#ifndef VP_SCENE_H
#define VP_SCENE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
uint8_t *vp_build_scene(const void *acceleration_structure, uint32_t *size);
#ifdef __cplusplus
}
#endif

#endif
