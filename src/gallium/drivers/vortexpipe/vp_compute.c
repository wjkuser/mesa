/* SPDX-License-Identifier: MIT */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vp_private.h"
#include "vp_compile.h"
#include "vp_launch.h"
#include "util/log.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "nir.h"
#include "nir_builder.h"

static simple_mtx_t registry_lock = SIMPLE_MTX_INITIALIZER;
static struct hash_table *registry;

void vp_reg_put(const void *key, void *data)
{
   simple_mtx_lock(&registry_lock);
   if (!registry)
      registry = _mesa_pointer_hash_table_create(NULL);
   _mesa_hash_table_insert(registry, key, data);
   simple_mtx_unlock(&registry_lock);
}

void *vp_reg_get(const void *key)
{
   simple_mtx_lock(&registry_lock);
   struct hash_entry *entry = registry ? _mesa_hash_table_search(registry, key) : NULL;
   void *data = entry ? entry->data : NULL;
   simple_mtx_unlock(&registry_lock);
   return data;
}

void vp_reg_del(const void *key)
{
   simple_mtx_lock(&registry_lock);
   struct hash_entry *entry = registry ? _mesa_hash_table_search(registry, key) : NULL;
   if (entry)
      _mesa_hash_table_remove(registry, entry);
   simple_mtx_unlock(&registry_lock);
}

void vp_dbg(const char *fmt, ...)
{
   if (!getenv("VORTEXPIPE_DEBUG"))
      return;
   va_list args;
   va_start(args, fmt);
   mesa_log_v(MESA_LOG_INFO, "MESA", fmt, args);
   va_end(args);
}

static void release_module(struct vp_cso *cso)
{
   if (cso && cso->vx_module) {
      vx_kernel_release(cso->vx_kernel);
      vx_module_release(cso->vx_module);
      cso->vx_kernel = NULL;
      cso->vx_module = NULL;
   }
}

static void *create_compute_state(struct pipe_context *pipe,
                                  const struct pipe_compute_state *state)
{
   struct vp_cso *cso = CALLOC_STRUCT(vp_cso);
   if (!cso)
      return NULL;
   nir_shader *nir = vp_screen_take_dev_nir(pipe->screen, (nir_shader *)state->prog);
   if (!nir)
      nir = nir_shader_clone(NULL, (nir_shader *)state->prog);
   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size) {
      nir->info.shared_size = ALIGN(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size, 16);
   }
   cso->lmem_size = nir->info.shared_size;
   cso->scratch_size = nir->scratch_size;
   vp_dbg("vortexpipe: compile %p: %s", (void *)cso,
          nir->info.name ? nir->info.name : "compute");
   vp_scan_descriptors(nir, cso->descs, &cso->num_descs);
   char *ir = NULL;
   bool ok = vp_nir_to_llvm(nir, &ir, NULL, NULL, 1, 0) &&
             vp_compile_vxbin(ir, VP_STARTUP_FS, false, &cso->vxbin, &cso->vxbin_size);
   vp_free_ir(ir);
   ralloc_free(nir);
   if (!ok) {
      free(cso);
      return NULL;
   }
   return cso;
}

static void bind_compute_state(struct pipe_context *pipe, void *state)
{
   struct vp_context *ctx = vp_reg_get(pipe);
   if (ctx->cur_cso != state)
      release_module(ctx->cur_cso);
   ctx->cur_cso = state;
}

static void delete_compute_state(struct pipe_context *pipe, void *state)
{
   struct vp_context *ctx = vp_reg_get(pipe);
   struct vp_cso *cso = state;
   if (ctx->cur_cso == cso)
      ctx->cur_cso = NULL;
   release_module(cso);
   vp_free_blob(cso->vxbin);
   free(cso);
}

static void set_constant_buffer(struct pipe_context *pipe, enum pipe_shader_type shader,
                                unsigned index, bool ownership,
                                const struct pipe_constant_buffer *cb)
{
   struct vp_context *ctx = vp_reg_get(pipe);
   if (shader == PIPE_SHADER_COMPUTE) {
      pipe_resource_reference(&ctx->cbuf[index].buffer, cb ? cb->buffer : NULL);
      ctx->cbuf[index].buffer_offset = cb ? cb->buffer_offset : 0;
      ctx->cbuf[index].buffer_size = cb ? cb->buffer_size : 0;
      ctx->cbuf[index].user_buffer = cb ? cb->user_buffer : NULL;
   }
   ctx->lp_set_constant_buffer(pipe, shader, index, ownership, cb);
}

static void set_shader_buffers(struct pipe_context *pipe, enum pipe_shader_type shader,
                               unsigned start, unsigned count,
                               const struct pipe_shader_buffer *buffers, unsigned writable)
{
   struct vp_context *ctx = vp_reg_get(pipe);
   if (shader == PIPE_SHADER_COMPUTE) {
      for (unsigned i = 0; i < count; ++i) {
         struct pipe_shader_buffer *dst = &ctx->sbuf[start + i];
         const struct pipe_shader_buffer *src = buffers ? &buffers[i] : NULL;
         pipe_resource_reference(&dst->buffer, src ? src->buffer : NULL);
         dst->buffer_offset = src ? src->buffer_offset : 0;
         dst->buffer_size = src ? src->buffer_size : 0;
      }
   }
   ctx->lp_set_shader_buffers(pipe, shader, start, count, buffers, writable);
}

static void launch_grid(struct pipe_context *pipe, const struct pipe_grid_info *info)
{
   struct vp_context *ctx = vp_reg_get(pipe);
   struct vp_screen *screen = vp_reg_get(pipe->screen);
   struct vp_cso *cso = ctx->cur_cso;
   uint32_t grid[3];
   memcpy(grid, info->grid, sizeof(grid));
   if (info->indirect)
      pipe_buffer_read(pipe, info->indirect, info->indirect_offset, sizeof(grid), grid);
   if (!grid[0] || !grid[1] || !grid[2])
      return;
   struct pipe_transfer *maps[VP_MAX_CBUFS] = { 0 };
   struct vp_const_buffer cbufs[VP_MAX_CBUFS] = { 0 };
   for (unsigned i = 0; i < VP_MAX_CBUFS; i++) {
      const struct pipe_constant_buffer *cb = &ctx->cbuf[i];
      if (!cb->buffer_size)
         continue;
      const void *host = cb->user_buffer;
      if (cb->buffer)
         host = pipe_buffer_map(pipe, cb->buffer, PIPE_MAP_READ, &maps[i]);
      cbufs[i].host = (const char *)host + cb->buffer_offset;
      cbufs[i].size = cb->buffer_size;
   }
   struct pipe_transfer *smap[VP_MAX_SSBO] = { 0 };
   struct vp_ssbo ssbos[VP_MAX_SSBO];
   unsigned count = 0;
   for (unsigned i = 0; i < VP_MAX_SSBO; ++i) {
      const struct pipe_shader_buffer *buffer = &ctx->sbuf[i];
      if (!buffer->buffer || !buffer->buffer_size)
         continue;
      const char *host = pipe_buffer_map(pipe, buffer->buffer, PIPE_MAP_READ, &smap[i]);
      ssbos[count++] = (struct vp_ssbo) {
         .host = host + buffer->buffer_offset,
         .size = buffer->buffer_size,
         .slot = i,
      };
   }
   vp_screen_resident_dirty_all(pipe->screen);
   vp_dbg("vortexpipe: dispatch %u shader=%p grid=%u,%u,%u block=%u,%u,%u shared=%u",
          ctx->launches, (void *)cso, grid[0], grid[1], grid[2],
          info->block[0], info->block[1], info->block[2], cso->lmem_size);
   bool ok = cso && vp_launch(pipe->screen, ctx->dev, cso->vxbin, cso->vxbin_size,
                              &cso->vx_module, &cso->vx_kernel,
                              cbufs, cso->descs, cso->num_descs,
                              ssbos, count, grid, info->block, info->grid_base,
                              cso->lmem_size, cso->scratch_size, screen->has_rtu);
   for (unsigned i = 0; i < VP_MAX_CBUFS; i++)
      if (maps[i])
         pipe_buffer_unmap(pipe, maps[i]);
   for (unsigned i = 0; i < VP_MAX_SSBO; ++i)
      if (smap[i])
         pipe_buffer_unmap(pipe, smap[i]);
   if (!ok) {
      mesa_loge("vortexpipe: compute dispatch failed");
      abort();
   }
   ctx->launches++;
   vp_dbg("vortexpipe: completed dispatch %u", ctx->launches);
}

static void destroy_context(struct pipe_context *pipe)
{
   struct vp_context *ctx = vp_reg_get(pipe);
   for (unsigned i = 0; i < PIPE_MAX_CONSTANT_BUFFERS; i++)
      pipe_resource_reference(&ctx->cbuf[i].buffer, NULL);
   for (unsigned i = 0; i < PIPE_MAX_SHADER_BUFFERS; ++i)
      pipe_resource_reference(&ctx->sbuf[i].buffer, NULL);
   vp_dbg("vortexpipe: device dispatches=%u", ctx->launches);
   ctx->lp_context_destroy(pipe);
   vp_reg_del(pipe);
   free(ctx);
}

struct pipe_context *vp_context_create(struct pipe_screen *screen, void *priv, unsigned flags)
{
   struct vp_screen *vps = vp_reg_get(screen);
   struct pipe_context *pipe = vps->lp_context_create(screen, priv, flags);
   if (!pipe)
      return NULL;
   struct vp_context *ctx = CALLOC_STRUCT(vp_context);
   if (!ctx) {
      pipe->destroy(pipe);
      return NULL;
   }
   ctx->dev = vps->dev;
   ctx->lp_set_constant_buffer = pipe->set_constant_buffer;
   ctx->lp_set_shader_buffers = pipe->set_shader_buffers;
   ctx->lp_context_destroy = pipe->destroy;
   vp_reg_put(pipe, ctx);
   pipe->create_compute_state = create_compute_state;
   pipe->bind_compute_state = bind_compute_state;
   pipe->delete_compute_state = delete_compute_state;
   pipe->set_constant_buffer = set_constant_buffer;
   pipe->set_shader_buffers = set_shader_buffers;
   pipe->launch_grid = launch_grid;
   pipe->destroy = destroy_context;
   return pipe;
}
