/* SPDX-License-Identifier: MIT */
#ifndef VP_MEMORY_H
#define VP_MEMORY_H

#include "vortex2.h"
#include <stdbool.h>
#include <stdint.h>

struct pipe_screen;
struct pipe_resource;

void vp_memory_init(struct pipe_screen *screen);
void vp_memory_finish(struct pipe_screen *screen);
void vp_memory_resource_destroy(struct pipe_screen *screen, struct pipe_resource *resource);
bool vp_memory_lookup(struct pipe_screen *screen, const void *host, uint32_t size,
                      vx_buffer_h *buffer, uint32_t *offset);
vx_result_t vp_memory_transfer(struct pipe_screen *screen, vx_queue_h queue, bool readback);

#endif
