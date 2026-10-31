/* SPDX-License-Identifier: MIT */
#include "vp_memory.h"
#include "vp_private.h"
#include "llvmpipe/lp_texture.h"
#include "util/log.h"
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

/* Buffer device addresses retain their value when stored in another buffer.
 * Give each allocation the same address in the host and RV32 address spaces;
 * shaders can then follow pointer graphs without descriptor-specific fixups.
 * Host/device copies are performed at queue dispatch boundaries.
 *
 * Allocations come from one fixed host window in first-fit order. Device
 * addresses then depend only on the application's allocation sequence, so
 * simulated cache and bank mapping is reproducible across processes. */
#define VP_MEMORY_WINDOW_BASE 0x40000000ull
#define VP_MEMORY_WINDOW_SIZE 0x40000000ull
struct vp_memory {
   struct llvmpipe_memory_allocation backing;
   vx_buffer_h buffer;
   struct pipe_resource *owner;
   struct vp_memory *next;
};

/* Return the pages to the window: zero-filled and inaccessible until reuse. */
static void
release_backing(struct vp_screen *vps, struct vp_memory *mem)
{
   mmap(mem->backing.cpu_addr, mem->backing.size, PROT_NONE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
   simple_mtx_lock(&vps->resident_lock);
   util_vma_heap_free(&vps->memory_heap, (uintptr_t)mem->backing.cpu_addr, mem->backing.size);
   simple_mtx_unlock(&vps->resident_lock);
}

static struct pipe_memory_allocation *
allocate_memory(struct pipe_screen *screen, uint64_t size)
{
   struct vp_screen *vps = vp_reg_get(screen);
   struct vp_memory *mem = calloc(1, sizeof(*mem));
   if (!mem)
      return NULL;
   uint64_t page = sysconf(_SC_PAGESIZE);
   mem->backing.size = (size + page - 1) & ~(page - 1);
   simple_mtx_lock(&vps->resident_lock);
   uint64_t addr = util_vma_heap_alloc(&vps->memory_heap, mem->backing.size, page);
   simple_mtx_unlock(&vps->resident_lock);
   if (!addr) {
      mesa_loge("vortexpipe: device memory window exhausted");
      free(mem);
      return NULL;
   }
   mem->backing.cpu_addr = (void *)(uintptr_t)addr;
   if (mprotect(mem->backing.cpu_addr, mem->backing.size, PROT_READ | PROT_WRITE)) {
      simple_mtx_lock(&vps->resident_lock);
      util_vma_heap_free(&vps->memory_heap, addr, mem->backing.size);
      simple_mtx_unlock(&vps->resident_lock);
      free(mem);
      return NULL;
   }
   vx_result_t result = vx_buffer_reserve(vps->dev, (uintptr_t)mem->backing.cpu_addr,
                                          mem->backing.size, 0, &mem->buffer);
   if (result != VX_SUCCESS) {
      mesa_loge("vortexpipe: reserve buffer address: %s", vx_result_string(result));
      release_backing(vps, mem);
      free(mem);
      return NULL;
   }
   simple_mtx_lock(&vps->resident_lock);
   mem->next = vps->memory;
   vps->memory = mem;
   simple_mtx_unlock(&vps->resident_lock);
   return (struct pipe_memory_allocation *)mem;
}

static void
free_memory(struct pipe_screen *screen, struct pipe_memory_allocation *allocation)
{
   struct vp_screen *vps = vp_reg_get(screen);
   struct vp_memory *mem = (struct vp_memory *)allocation;
   simple_mtx_lock(&vps->resident_lock);
   struct vp_memory **entry = &vps->memory;
   while (*entry != mem)
      entry = &(*entry)->next;
   *entry = mem->next;
   simple_mtx_unlock(&vps->resident_lock);
   vx_buffer_release(mem->buffer);
   release_backing(vps, mem);
   free(mem);
}

static struct pipe_resource *
resource_create(struct pipe_screen *screen, const struct pipe_resource *templ)
{
   uint64_t size;
   struct pipe_resource *res = screen->resource_create_unbacked(screen, templ, &size);
   if (!res)
      return NULL;
   struct pipe_memory_allocation *allocation = allocate_memory(screen, size);
   if (!allocation || !screen->resource_bind_backing(screen, res, allocation, 0, 0, 0)) {
      if (allocation)
         free_memory(screen, allocation);
      screen->resource_destroy(screen, res);
      return NULL;
   }
   ((struct vp_memory *)allocation)->owner = res;
   return res;
}

void
vp_memory_resource_destroy(struct pipe_screen *screen, struct pipe_resource *resource)
{
   struct vp_screen *vps = vp_reg_get(screen);
   struct vp_memory *mem = vps->memory;
   while (mem && mem->owner != resource)
      mem = mem->next;
   if (mem)
      free_memory(screen, (struct pipe_memory_allocation *)mem);
}

bool
vp_memory_lookup(struct pipe_screen *screen, const void *host, uint32_t size,
                  vx_buffer_h *buffer, uint32_t *offset)
{
   struct vp_screen *vps = vp_reg_get(screen);
   uintptr_t addr = (uintptr_t)host;
   simple_mtx_lock(&vps->resident_lock);
   for (struct vp_memory *mem = vps->memory; mem; mem = mem->next) {
      uintptr_t base = (uintptr_t)mem->backing.cpu_addr;
      if (addr >= base && addr + size <= base + mem->backing.size) {
         *buffer = mem->buffer;
         *offset = addr - base;
         simple_mtx_unlock(&vps->resident_lock);
         return true;
      }
   }
   simple_mtx_unlock(&vps->resident_lock);
   return false;
}

vx_result_t
vp_memory_transfer(struct pipe_screen *screen, vx_queue_h queue, bool readback)
{
   struct vp_screen *vps = vp_reg_get(screen);
   vx_result_t result = VX_SUCCESS;
   simple_mtx_lock(&vps->resident_lock);
   for (struct vp_memory *mem = vps->memory; mem; mem = mem->next) {
      if (readback)
         result = vx_enqueue_read(queue, mem->backing.cpu_addr, mem->buffer, 0,
                                   mem->backing.size, 0, NULL, NULL);
      else
         result = vx_enqueue_write(queue, mem->buffer, 0, mem->backing.cpu_addr,
                                    mem->backing.size, 0, NULL, NULL);
      if (result != VX_SUCCESS)
         break;
   }
   simple_mtx_unlock(&vps->resident_lock);
   return result;
}

bool
vp_memory_init(struct pipe_screen *screen)
{
   struct vp_screen *vps = vp_reg_get(screen);
   vps->memory_window = mmap((void *)(uintptr_t)VP_MEMORY_WINDOW_BASE, VP_MEMORY_WINDOW_SIZE,
                             PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                             MAP_FIXED_NOREPLACE, -1, 0);
   if (vps->memory_window != (void *)(uintptr_t)VP_MEMORY_WINDOW_BASE) {
      mesa_loge("vortexpipe: device memory window %#llx is unavailable",
                (unsigned long long)VP_MEMORY_WINDOW_BASE);
      if (vps->memory_window != MAP_FAILED)
         munmap(vps->memory_window, VP_MEMORY_WINDOW_SIZE);
      vps->memory_window = NULL;
      return false;
   }
   util_vma_heap_init(&vps->memory_heap, VP_MEMORY_WINDOW_BASE, VP_MEMORY_WINDOW_SIZE);
   vps->memory_heap.alloc_high = false;
   screen->allocate_memory = allocate_memory;
   screen->free_memory = free_memory;
   screen->resource_create = resource_create;
   return true;
}

void
vp_memory_finish(struct pipe_screen *screen)
{
   struct vp_screen *vps = vp_reg_get(screen);
   while (vps->memory)
      free_memory(screen, (struct pipe_memory_allocation *)vps->memory);
   if (vps->memory_window) {
      util_vma_heap_finish(&vps->memory_heap);
      munmap(vps->memory_window, VP_MEMORY_WINDOW_SIZE);
      vps->memory_window = NULL;
   }
}
