/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * vp_compile -- Phase 2 #4b: LLVM IR -> Vortex .vxbin.
 *
 * Drives the existing Vortex device toolchain on the kernel module
 * vp_nir_to_llvm emits: llvm-vortex's clang compiles + links the IR
 * (riscv32 or riscv64, +xvortex) against the KMU device kernel
 * library (libvortex2.a) and the baremetal libc/compiler-rt, then
 * vxbin.py packages the ELF into a .vxbin. First cut: fork/exec the
 * tools via system(); moving in-process is a later optimization
 * (§3 n.4).
 *
 * Paths come from VORTEX_HOME (the Vortex tree) and TOOLDIR (the
 * $HOME/tools install root) -- the meson-baked defaults below,
 * overridable by the VORTEX_HOME / VORTEX_TOOLDIR env vars. The
 * Vortex out-of-tree build dir (where libvortex2.a is built) is
 * VORTEX_HOME/build by convention; VORTEX_BUILD overrides it for
 * trees configured under a differently named build dir (e.g. the
 * CI build32/build64).
 *
 * Target resources and XLEN come from the screen that owns the shader.
 */

#define _GNU_SOURCE
#include "vp_compile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "vp_private.h"   /* vp_dbg */
#include "util/log.h"

#ifndef VP_VORTEX_HOME
#define VP_VORTEX_HOME ""
#endif
#ifndef VP_TOOLDIR
#define VP_TOOLDIR ""
#endif

static bool
write_text(const char *path, const char *data)
{
   FILE *f = fopen(path, "wb");
   if (!f)
      return false;
   size_t n = strlen(data);
   bool ok = fwrite(data, 1, n, f) == n;
   fclose(f);
   return ok;
}

static void *
read_blob(const char *path, size_t *out_size)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      return NULL;
   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   void *buf = (sz > 0) ? malloc((size_t)sz) : NULL;
   if (buf && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
      free(buf);
      buf = NULL;
   }
   fclose(f);
   if (buf)
      *out_size = (size_t)sz;
   return buf;
}

/* Use the requested shader target when supplied. A standalone driver can
 * discover it from the device; an explicit target must agree with that device. */
bool
vp_init_compiler_target(struct vp_compiler_target *gpu, unsigned device_warp, unsigned device_xlen)
{
   const char *width = getenv("VX_CFG_NUM_THREADS");
   const char *bits = getenv("MESA_VORTEX_XLEN");
   char *end;
   unsigned long warp = width ? strtoul(width, &end, 10) : device_warp;
   if ((width && (*width < '0' || *width > '9' || *end)) ||
       warp != device_warp || !warp || warp > 32 || (warp & (warp - 1))) {
      mesa_loge("vortexpipe: invalid or incompatible warp width (requested %s, device %u)",
                width ? width : "auto", device_warp);
      return false;
   }
   unsigned long xlen = bits ? strtoul(bits, &end, 10) : device_xlen;
   if ((bits && (*bits < '0' || *bits > '9' || *end)) ||
       xlen != device_xlen || (xlen != 32 && xlen != 64)) {
      mesa_loge("vortexpipe: invalid or incompatible XLEN (requested %s, device %u)",
                bits ? bits : "auto", device_xlen);
      return false;
   }
   *gpu = (struct vp_compiler_target){ .warp_size = warp, .xlen = xlen };
   const char *redux = getenv("HACKRTCORE_NATIVE_WARP_REDUX");
   gpu->native_redux = redux && strcmp(redux, "1") == 0;
   /* Without a known partition count, omit the entire extension budget.
    * LLVM then uses its normal architectural registers and spills. */
   if (!getenv("VX_CFG_REGISTER_PARTITION_COUNT"))
      return true;
   const struct { unsigned *dst; const char *key; } fields[] = {
#define VP_RESOURCE(field, key, option) { &gpu->field, #key },
#include "vp_register_resources.def"
#undef VP_RESOURCE
   };
   for (unsigned i = 0; i < ARRAY_SIZE(fields); ++i) {
      if (fields[i].dst == &gpu->warp_size)
         continue;
      const char *s = getenv(fields[i].key);
      char *end;
      unsigned long n = s ? strtoul(s, &end, 10) : 0;
      if (!s || *s < '0' || *s > '9' || *end || !n || n > UINT32_MAX) {
         mesa_loge("vortexpipe: invalid public target property %s", fields[i].key);
         return false;
      }
      *fields[i].dst = n;
   }
   return true;
}

/* ---- shader disk cache ------------------------------------------------ *
 * Compiling a shader forks clang and vxbin.py, which dominates the cost of a
 * short test. The product is a pure function of the IR and the toolchain
 * invocation, so key the binary on both and reuse it across processes.
 *
 * The key covers the whole command string, so a toolchain, flag or link-layout
 * change can never serve a stale binary. VORTEXPIPE_NO_CACHE=1 bypasses the
 * cache entirely when a compile-path bug is suspected. */

/* FNV-1a, 64-bit. */
static uint64_t
vp_hash(uint64_t h, const void *data, size_t len)
{
   const uint8_t *p = data;
   for (size_t i = 0; i < len; i++) {
      h ^= p[i];
      h *= 0x100000001b3ull;
   }
   return h;
}

/* Cache directory, honoring $VORTEXPIPE_CACHE_DIR then $XDG_CACHE_HOME then
 * $HOME. Returns false when no writable home is known, which disables the
 * cache rather than falling back to a shared temp location. */
static bool
vp_cache_dir(char *out, size_t out_size)
{
   const char *d = getenv("VORTEXPIPE_CACHE_DIR");
   if (d && *d) {
      snprintf(out, out_size, "%s", d);
      return true;
   }
   d = getenv("XDG_CACHE_HOME");
   if (d && *d) {
      snprintf(out, out_size, "%s/vortexpipe", d);
      return true;
   }
   d = getenv("HOME");
   if (d && *d) {
      snprintf(out, out_size, "%s/.cache/vortexpipe", d);
      return true;
   }
   return false;
}

bool
vp_compile_vxbin(const struct vp_compiler_target *gpu, const char *llvm_ir, unsigned long long startup_addr,
                 bool link_gfx_sw, void **out_blob, size_t *out_size)
{
   *out_blob = NULL;
   *out_size = 0;

   const char *vh = getenv("VORTEX_HOME");
   if (!vh || !*vh) vh = VP_VORTEX_HOME;
   const char *td = getenv("VORTEX_TOOLDIR");
   if (!td || !*td) td = VP_TOOLDIR;
   if (!*vh || !*td) {
      mesa_loge("vortexpipe: VORTEX_HOME / TOOLDIR not configured");
      return false;
   }
   const bool is64 = gpu->xlen == 64;
   const char *target  = is64 ? "riscv64-unknown-elf" : "riscv32-unknown-elf";
   const char *gnu_dir = is64 ? "riscv64-gnu-toolchain" : "riscv32-gnu-toolchain";
   const char *march   = is64 ? "rv64imafd" : "rv32imaf";
   const char *mabi    = is64 ? "lp64d"     : "ilp32f";
   /* RV32 medlow covers all addresses and lets LLVM rematerialize constant
    * bases. Kernels are statically linked at their actual launch address. */
   const char *code_model = is64 ? "medany" : "medlow";
   const char *linker  = is64 ? "link64.ld" : "link32.ld";
   const char *libc    = is64 ? "libc64"    : "libc32";
   const char *libcrt  = is64 ? "libcrt64"  : "libcrt32";
   const char *crt_a   = is64 ? "libclang_rt.builtins-riscv64.a"
                              : "libclang_rt.builtins-riscv32.a";

   /* Build dir holding libvortex2.a -- VORTEX_HOME/build by default. */
   char bd_buf[512];
   const char *bd = getenv("VORTEX_BUILD");
   if (!bd || !*bd) {
      snprintf(bd_buf, sizeof bd_buf, "%s/build", vh);
      bd = bd_buf;
   }

   /* Preserve branch-local stores so Vortex split/join regions stay nested. */
   char simt_options[2048] = "-mllvm -vortex-divergence-max-bbs=65536 "
                              "-mllvm -simplifycfg-sink-common=false "
                              "-funroll-loops -mllvm -unroll-runtime "
                              "-mllvm -enable-post-misched "
                              "-mllvm -riscv-use-rematerializable-movimm";
   if (gpu->partitions) {
      const struct { const char *option; unsigned value; } options[] = {
#define VP_RESOURCE(field, key, option) { option, gpu->field },
#include "vp_register_resources.def"
#undef VP_RESOURCE
      };
      for (unsigned i = 0; i < ARRAY_SIZE(options); ++i) {
         size_t used = strlen(simt_options);
         int added = snprintf(simt_options + used, sizeof(simt_options) - used,
                              " -mllvm -%s=%u", options[i].option, options[i].value);
         if (added < 0 || (size_t)added >= sizeof(simt_options) - used)
            return false;
      }
   }
   char cache_path[640];
   bool cache_enabled = !getenv("VORTEXPIPE_NO_CACHE");
   cache_path[0] = '\0';
   if (cache_enabled) {
      char cfg[4096];
      snprintf(cfg, sizeof cfg, "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%llx|%d|%s",
               simt_options, td, vh, bd, target, gnu_dir, march, mabi,
               linker, libc, libcrt, startup_addr, link_gfx_sw ? 1 : 0, code_model);
      uint64_t key = vp_hash(0xcbf29ce484222325ull, cfg, strlen(cfg));
      key = vp_hash(key, llvm_ir, strlen(llvm_ir));
      char inputs[3][1024];
      snprintf(inputs[0], sizeof inputs[0], "%s/sw/kernel/scripts/vxbin.py", vh);
      snprintf(inputs[1], sizeof inputs[1], "%s/llvm-vortex/lib/libLLVMRISCVCodeGen.so", td);
      snprintf(inputs[2], sizeof inputs[2], "%s/llvm-vortex/lib/libLLVMRISCVDesc.so", td);
      for (unsigned i = 0; i < ARRAY_SIZE(inputs); ++i) {
         size_t size;
         void *data = read_blob(inputs[i], &size);
         if (!data) {
            mesa_loge("vortexpipe: cannot read compiler input %s", inputs[i]);
            return false;
         }
         key = vp_hash(key, data, size);
         free(data);
      }

      char cdir[512];
      if (vp_cache_dir(cdir, sizeof cdir)) {
         char *mk = NULL;
         int mk_rc = -1;
         if (asprintf(&mk, "mkdir -p %s", cdir) >= 0) {
            mk_rc = system(mk);
            free(mk);
         }
         if (mk_rc == 0) {
            snprintf(cache_path, sizeof cache_path, "%s/%016llx.vxbin", cdir,
                     (unsigned long long)key);
            *out_blob = read_blob(cache_path, out_size);
            if (*out_blob) {
               vp_dbg("vortexpipe: shader cache hit (%s)", cache_path);
               return true;
            }
         }
      }
   }

   /* Compile scratch dir. Honor TMPDIR (falling back to /tmp) so the transient
    * kernel.ll/.elf/.vxbin files land on a roomy filesystem -- a full /tmp (the
    * default) intermittently fails the clang write and looks like a shader bug. */
   char dir[512];
   const char *tmpdir = getenv("TMPDIR");
   if (!tmpdir || !*tmpdir)
      tmpdir = "/tmp";
   snprintf(dir, sizeof dir, "%s/vortexpipe.XXXXXX", tmpdir);
   if (!mkdtemp(dir)) {
      mesa_loge("vortexpipe: mkdtemp failed");
      return false;
   }

   char p_ll[256], p_elf[256], p_bin[256];
   snprintf(p_ll,  sizeof p_ll,  "%s/kernel.ll",   dir);
   snprintf(p_elf, sizeof p_elf, "%s/kernel.elf",  dir);
   snprintf(p_bin, sizeof p_bin, "%s/kernel.vxbin", dir);

   bool ok = write_text(p_ll, llvm_ir);
   char *cmd = NULL;

   /* §5: when a unit is routed to software, co-compile the gfx_sw_abi C ABI into
    * the kernel (compiled from the SSOT C++ headers, so the divergent OM merge is
    * processed by the Vortex divergence pass over the whole kernel; --gc-sections
    * drops unused entry points). VX_types.h is the generated copy under $bd/sw. */
   char gfx_seg[1024] = "";
   if (ok && link_gfx_sw) {
      /* -DNDEBUG: the shared tex/OM headers (and cocogfx) use assert(); the
       * baremetal link has no __assert_func, so compile assertions out. */
      snprintf(gfx_seg, sizeof gfx_seg,
         "-std=c++17 -DNDEBUG -D__VORTEX__ -DGFX_SW_DIVERGENCE_OK "
         "-I%s/sw/gfx -I%s/sw/common -I%s/third_party -I%s/sw %s/sw/gfx/gfx_sw_abi.cpp",
         vh, vh, vh, bd, vh);
   }

   /* compile + link the IR into a Vortex KMU kernel ELF */
   if (ok) {
      /* Device flags mirror the canonical Vortex kernel toolchain
       * invocation in tests/regression/common.mk: the llvm-vortex
       * clang with +xvortex (the Vortex ISA extension) and +zicond,
       * --sysroot / --gcc-toolchain pointing at the GNU toolchain
       * matching the target XLEN, and -disable-loop-idiom-all. The
       * Vortex branch-divergence pass is part of the +xvortex backend
       * and is left at its default (enabled): it is what lowers
       * divergent SIMT control flow into correct masked execution, so
       * we must never pass -mllvm -vortex-branch-divergence=0. */
      if (asprintf(&cmd,
            "%s/llvm-vortex/bin/clang --target=%s "
            "--sysroot=%s/%s/%s "
            "--gcc-toolchain=%s/%s "
            "-march=%s -mabi=%s "
            "-Xclang -target-feature -Xclang +xvortex "
            "-Xclang -target-feature -Xclang +zicond "
            "-mllvm -disable-loop-idiom-all %s "
            "-Wno-unused-command-line-argument -Wno-override-module "
            "-O3 -mcmodel=%s -nostartfiles -nostdlib "
            "-fdata-sections -ffunction-sections -fuse-ld=lld "
            "%s %s "
            "-Wl,-Bstatic,--gc-sections,-T,%s/sw/kernel/scripts/%s,"
            "--defsym=STARTUP_ADDR=0x%llx "
            "%s/sw/kernel/libvortex2.a "
            "-L%s/%s/lib -lm -lc "
            "%s/%s/lib/baremetal/%s "
            "-o %s 2>%s/clang.log",
            td, target,
            td, gnu_dir, target,
            td, gnu_dir,
            march, mabi,
            simt_options,
            code_model,
            p_ll, gfx_seg,
            vh, linker,
            startup_addr,
            bd,
            td, libc,
            td, libcrt, crt_a,
            p_elf, dir) < 0) {
         cmd = NULL;
         ok = false;
      }
      if (ok && system(cmd) != 0) {
         mesa_loge("vortexpipe: device clang/link failed (kept %s/clang.log)",
                   dir);
         ok = false;
      }
      free(cmd);
      cmd = NULL;
   }

   /* package the ELF into a .vxbin */
   if (ok) {
      if (asprintf(&cmd,
            "OBJCOPY=%s/llvm-vortex/bin/llvm-objcopy python3 "
            "%s/sw/kernel/scripts/vxbin.py %s %s 2>%s/vxbin.log",
            td, vh, p_elf, p_bin, dir) < 0) {
         cmd = NULL;
         ok = false;
      }
      if (ok && system(cmd) != 0) {
         mesa_loge("vortexpipe: vxbin.py failed (kept %s/vxbin.log)", dir);
         ok = false;
      }
      free(cmd);
      cmd = NULL;
   }

   if (ok) {
      *out_blob = read_blob(p_bin, out_size);
      ok = (*out_blob != NULL);
   }

   /* Populate the cache. Copy to a unique name in the cache directory and
    * rename into place, so a concurrent reader never observes a torn file. */
   if (ok && cache_path[0]) {
      char *cp = NULL;
      if (asprintf(&cp, "cp %s %s.%d.tmp && mv %s.%d.tmp %s",
                   p_bin, cache_path, (int)getpid(),
                   cache_path, (int)getpid(), cache_path) >= 0) {
         if (system(cp) != 0)
            vp_dbg("vortexpipe: shader cache store failed (%s)", cache_path);
         free(cp);
      }
   }

   /* clean up on success; keep the temp dir for debugging on failure or
    * when VORTEXPIPE_KEEP_TMP is set (lets a developer inspect kernel
    * ELF / .vxbin contents, symbol tables, etc.). */
   if (ok && !getenv("VORTEXPIPE_KEEP_TMP")) {
      if (asprintf(&cmd, "rm -rf %s", dir) >= 0) {
         system(cmd);
         free(cmd);
      }
   } else if (ok) {
      fprintf(stderr, "vortexpipe: VORTEXPIPE_KEEP_TMP set; kept %s\n", dir);
   }
   return ok;
}

void
vp_free_blob(void *blob)
{
   free(blob);
}
