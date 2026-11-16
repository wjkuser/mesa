/* SPDX-License-Identifier: MIT */
#ifndef VP_COMPUTE_H
#define VP_COMPUTE_H

struct vp_cso {
   void *vxbin;
   size_t vxbin_size;
   unsigned lmem_size;
   unsigned scratch_size;
   bool scratch_private;
   struct vp_desc descs[VP_MAX_DESCS];
   unsigned num_descs;
   vx_module_h vx_module;
   vx_kernel_h vx_kernel;
};

struct vp_context {
   vx_device_h dev;
   struct vp_cso *cur_cso;
   struct pipe_constant_buffer cbuf[PIPE_MAX_CONSTANT_BUFFERS];
   struct pipe_shader_buffer sbuf[PIPE_MAX_SHADER_BUFFERS];
   void (*lp_set_shader_buffers)(struct pipe_context *, enum pipe_shader_type,
                                unsigned, unsigned, const struct pipe_shader_buffer *, unsigned);
   void (*lp_set_constant_buffer)(struct pipe_context *, enum pipe_shader_type,
                                 unsigned, bool, const struct pipe_constant_buffer *);
   void (*lp_context_destroy)(struct pipe_context *);
   void (*lp_resource_copy_region)(struct pipe_context *, struct pipe_resource *,
                                   unsigned, unsigned, unsigned, unsigned,
                                   struct pipe_resource *, unsigned, const struct pipe_box *);
   void (*lp_clear_buffer)(struct pipe_context *, struct pipe_resource *,
                           unsigned, unsigned, const void *, int);
   unsigned launches;
};

#endif
