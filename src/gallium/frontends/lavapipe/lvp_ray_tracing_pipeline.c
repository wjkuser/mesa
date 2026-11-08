/*
 * Copyright © 2024 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "lvp_private.h"
#include "lvp_acceleration_structure.h"
#include "nir/lvp_nir.h"

/* Vortex RTU register-file slot ABI + wait-status codes (generated from
 * VX_types.toml). Used by the hardware ray-traversal path below, which lowers
 * VkTraceRaysKHR onto the RTU via the vortex_rt_* NIR intrinsics — the same
 * intrinsics vortexpipe already emits for VK_KHR_ray_query. */
#ifndef VP_HACKRTCORE
#include "VX_types.h"
#else
#include "scene_layout.h"
#include "registers.h"
#endif

#include "vk_pipeline.h"

#include "spirv/spirv.h"

#include "util/mesa-sha1.h"
#include "util/simple_mtx.h"

static void
lvp_init_ray_tracing_groups(struct lvp_pipeline *pipeline,
                            const VkRayTracingPipelineCreateInfoKHR *create_info)
{
   uint32_t i = 0;
   for (; i < create_info->groupCount; i++) {
      const VkRayTracingShaderGroupCreateInfoKHR *group_info = create_info->pGroups + i;
      struct lvp_ray_tracing_group *dst = pipeline->rt.groups + i;

      dst->recursive_index = VK_SHADER_UNUSED_KHR;
      dst->ahit_index = VK_SHADER_UNUSED_KHR;
      dst->isec_index = VK_SHADER_UNUSED_KHR;

      switch (group_info->type) {
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR:
         if (group_info->generalShader != VK_SHADER_UNUSED_KHR) {
            dst->recursive_index = group_info->generalShader;
         }
         break;
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR:
         if (group_info->closestHitShader != VK_SHADER_UNUSED_KHR) {
            dst->recursive_index = group_info->closestHitShader;
         }
         if (group_info->anyHitShader != VK_SHADER_UNUSED_KHR) {
            dst->ahit_index = group_info->anyHitShader;
         }
         break;
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR:
         if (group_info->closestHitShader != VK_SHADER_UNUSED_KHR) {
            dst->recursive_index = group_info->closestHitShader;
         }
         if (group_info->intersectionShader != VK_SHADER_UNUSED_KHR) {
            dst->isec_index = group_info->intersectionShader;

            if (group_info->anyHitShader != VK_SHADER_UNUSED_KHR)
               dst->ahit_index = group_info->anyHitShader;
         }
         break;
      default:
         unreachable("Unimplemented VkRayTracingShaderGroupTypeKHR");
      }

      dst->handle.index = p_atomic_inc_return(&pipeline->device->group_handle_alloc);
   }

   if (!create_info->pLibraryInfo)
      return;

   uint32_t stage_base_index = create_info->stageCount;
   for (uint32_t library_index = 0; library_index < create_info->pLibraryInfo->libraryCount; library_index++) {
      VK_FROM_HANDLE(lvp_pipeline, library, create_info->pLibraryInfo->pLibraries[library_index]);
      for (uint32_t group_index = 0; group_index < library->rt.group_count; group_index++) {
         const struct lvp_ray_tracing_group *src = library->rt.groups + group_index;
         struct lvp_ray_tracing_group *dst = pipeline->rt.groups + i;

         dst->handle = src->handle;

         if (src->recursive_index != VK_SHADER_UNUSED_KHR)
            dst->recursive_index = stage_base_index + src->recursive_index;
         else
            dst->recursive_index = VK_SHADER_UNUSED_KHR;

         if (src->ahit_index != VK_SHADER_UNUSED_KHR)
            dst->ahit_index = stage_base_index + src->ahit_index;
         else
            dst->ahit_index = VK_SHADER_UNUSED_KHR;

         if (src->isec_index != VK_SHADER_UNUSED_KHR)
            dst->isec_index = stage_base_index + src->isec_index;
         else
            dst->isec_index = VK_SHADER_UNUSED_KHR;

         i++;
      }
      stage_base_index += library->rt.stage_count;
   }
}

static bool
lvp_lower_ray_tracing_derefs(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   bool progress = false;

   nir_builder _b = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &_b;

   nir_def *arg_offset = nir_load_shader_call_data_offset_lvp(b);

   nir_foreach_block (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (!nir_deref_mode_is_one_of(deref, nir_var_shader_call_data |
                                       nir_var_ray_hit_attrib))
            continue;

         bool is_shader_call_data = nir_deref_mode_is(deref, nir_var_shader_call_data);

         deref->modes = nir_var_function_temp;
         progress = true;

         if (deref->deref_type == nir_deref_type_var) {
            b->cursor = nir_before_instr(&deref->instr);
            nir_def *offset = is_shader_call_data ? arg_offset : nir_imm_int(b, 0);
            nir_deref_instr *replacement =
               nir_build_deref_cast(b, offset, nir_var_function_temp, deref->var->type, 0);
            nir_def_replace(&deref->def, &replacement->def);
         }
      }
   }

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

static bool
lvp_move_ray_tracing_intrinsic(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_load_shader_record_ptr:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_t_max:
      nir_instr_move(nir_before_impl(b->impl), &instr->instr);
      return true;
   default:
      return false;
   }
}

#ifdef VP_HACKRTCORE
/* Suspend only after leaving the search loops. Native shader continuations
 * execute their terminal preparation after those loop barriers, before the
 * final dispatcher return. Capture external SSA inputs on each original
 * invocation path and keep the complete terminal region in program order. */
static bool
lvp_terminal_region_safe(nir_cf_node *node, bool allow_halt)
{
   if (node->type == nir_cf_node_loop)
      return false;
   nir_foreach_block_in_cf_node(block, node) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump) {
            if (!allow_halt || nir_instr_as_jump(instr)->type != nir_jump_halt)
               return false;
         } else if (instr->type == nir_instr_type_intrinsic) {
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_scratch:
            case nir_intrinsic_store_scratch:
            case nir_intrinsic_shader_clock:
            case nir_intrinsic_rt_trace_ray:
               break;
            default:
               return false;
            }
         } else if (instr->type != nir_instr_type_alu &&
                    instr->type != nir_instr_type_load_const &&
                    instr->type != nir_instr_type_phi &&
                    instr->type != nir_instr_type_undef) {
            return false;
         }
      }
   }
   return true;
}

static bool
lvp_collect_terminal_src(nir_src *src, void *data)
{
   util_dynarray_append((struct util_dynarray *)data, nir_src *, src);
   return true;
}

static void
lvp_defer_terminal_regions(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   struct util_dynarray sites;
   util_dynarray_init(&sites, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_rt_trace_ray)
            util_dynarray_append(&sites, nir_instr *, instr);
      }
   }
   unsigned count = util_dynarray_num_elements(&sites, nir_instr *);
   if (!count) {
      util_dynarray_fini(&sites);
      return;
   }
   nir_cf_list *tails = rzalloc_array(shader, nir_cf_list, count);
   struct hash_table **maps = rzalloc_array(shader, struct hash_table *, count);
   struct util_dynarray *srcs = rzalloc_array(shader, struct util_dynarray, count);
   struct util_dynarray *input_order = rzalloc_array(shader, struct util_dynarray, count);
   nir_variable *selected = nir_local_variable_create(impl, glsl_uint_type(), "terminal_region");
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_store_var(&b, selected, nir_imm_int(&b, 0), 1);
   struct hash_table *shared_inputs = _mesa_pointer_hash_table_create(NULL);
   unsigned moved = 0;
   util_dynarray_foreach(&sites, nir_instr *, site) {
      nir_block *last = (*site)->block;
      if (!nir_cf_node_is_last(&last->cf_node) ||
          !lvp_terminal_region_safe(&last->cf_node, true))
         continue;
      nir_cf_node *first = &last->cf_node;
      while (nir_cf_node_prev(first)) {
         nir_cf_node *prev = nir_cf_node_prev(first);
         if (!lvp_terminal_region_safe(prev, false)) break;
         first = prev;
      }
      if (first->type != nir_cf_node_block) first = nir_cf_node_next(first);
      if (first == &last->cf_node) continue;
      struct set *blocks = _mesa_pointer_set_create(NULL);
      struct util_dynarray sources;
      util_dynarray_init(&sources, NULL);
      for (nir_cf_node *node = first; node; node = nir_cf_node_next(node)) {
         nir_foreach_block_in_cf_node(block, node) {
            _mesa_set_add(blocks, block);
            nir_foreach_instr(instr, block)
               nir_foreach_src(instr, lvp_collect_terminal_src, &sources);
            nir_if *nif = nir_block_get_following_if(block);
            if (nif) util_dynarray_append(&sources, nir_src *, &nif->condition);
         }
      }
      struct hash_table *inputs = _mesa_pointer_hash_table_create(NULL);
      util_dynarray_init(&input_order[moved], NULL);
      b.cursor = nir_before_block(nir_cf_node_as_block(first));
      unsigned input_index = 0;
      util_dynarray_foreach(&sources, nir_src *, src) {
         nir_def *def = (*src)->ssa;
         if (_mesa_set_search(blocks, def->parent_instr->block) ||
             _mesa_hash_table_search(inputs, def)) continue;
         util_dynarray_append(&input_order[moved], nir_def *, def);
         if (def->parent_instr->type == nir_instr_type_load_const) {
            _mesa_hash_table_insert(inputs, def, NULL);
            continue;
         }
         const struct glsl_type *scalar = def->bit_size == 1 ?
            glsl_bool_type() : glsl_uintN_t_type(def->bit_size);
         const struct glsl_type *type = glsl_replace_vector_type(scalar, def->num_components);
         uintptr_t key = ((uintptr_t)input_index++ << 16) |
                         (def->num_components << 8) | def->bit_size;
         struct hash_entry *shared = _mesa_hash_table_search(shared_inputs, (void *)key);
         nir_variable *var = shared ? shared->data :
            nir_local_variable_create(impl, type, "terminal_input");
         if (!shared) _mesa_hash_table_insert(shared_inputs, (void *)key, var);
         nir_store_var(&b, var, def, BITFIELD_MASK(def->num_components));
         _mesa_hash_table_insert(inputs, def, var);
      }
      /* Insert captures before the first old instruction, then split after
       * them so only original terminal instructions move. */
      nir_store_var(&b, selected, nir_imm_int(&b, moved + 1), 1);
      nir_cursor begin = b.cursor;
      b.cursor = nir_cf_extract(&tails[moved], begin, nir_after_block(last));
      nir_jump(&b, nir_jump_return);
      /* Detached regions must not appear in the live implementation's SSA
       * use traversal while lower_returns repairs that implementation. */
      util_dynarray_foreach(&sources, nir_src *, src)
         if (_mesa_hash_table_search(inputs, (*src)->ssa))
            list_delinit(&(*src)->use_link);
      maps[moved] = inputs;
      srcs[moved] = sources;
      _mesa_set_destroy(blocks, NULL);
      moved++;
   }
   if (moved) {
      nir_progress(true, impl, nir_metadata_none);
      nir_lower_returns(shader);
      b.cursor = nir_after_impl(impl);
      for (unsigned i = 0; i < moved; ++i) {
         nir_push_if(&b, nir_ieq_imm(&b, nir_load_var(&b, selected), i + 1));
         util_dynarray_foreach(&input_order[i], nir_def *, input) {
            nir_def *old = *input;
            struct hash_entry *entry = _mesa_hash_table_search(maps[i], old);
            nir_def *replacement;
            if (entry->data) {
               replacement = nir_load_var(&b, (nir_variable *)entry->data);
            } else {
               nir_instr *constant = nir_instr_clone(shader, old->parent_instr);
               nir_builder_instr_insert(&b, constant);
               replacement = nir_instr_def(constant);
            }
            util_dynarray_foreach(&srcs[i], nir_src *, src)
               if ((*src)->ssa == old) nir_src_rewrite(*src, replacement);
         }
         util_dynarray_fini(&srcs[i]);
         util_dynarray_fini(&input_order[i]);
         _mesa_hash_table_destroy(maps[i], NULL);
         b.cursor = nir_cf_reinsert(&tails[i], b.cursor);
         nir_pop_if(&b, NULL);
      }
      nir_progress(true, impl, nir_metadata_none);
      nir_validate_shader(shader, "deferred terminal regions");
      nir_validate_ssa_dominance(shader, "deferred terminal regions");
   }
   _mesa_hash_table_destroy(shared_inputs, NULL);

   util_dynarray_fini(&sites);
}

/* A continuation may contain several copies of the same suspension site after
 * loop peeling. Funnel those terminal paths through one call epilogue, as the
 * native raygen code does before returning to the shader dispatcher. */
static void
lvp_merge_raygen_call_tail(nir_shader *shader)
{
   lvp_defer_terminal_regions(shader);
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   struct util_dynarray calls;
   util_dynarray_init(&calls, NULL);
   nir_intrinsic_instr *sample = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_rt_trace_ray)
            continue;
         if (sample && nir_intrinsic_call_idx(sample) != nir_intrinsic_call_idx(intr)) {
            util_dynarray_fini(&calls);
            return;
         }
         sample = intr;
         util_dynarray_append(&calls, nir_intrinsic_instr *, intr);
      }
   }
   if (!sample) {
      util_dynarray_fini(&calls);
      return;
   }
   nir_intrinsic_instr *call = nir_instr_as_intrinsic(nir_instr_clone(shader, &sample->instr));
   nir_variable *args[11];
   for (unsigned i = 0; i < ARRAY_SIZE(args); ++i)
      args[i] = nir_local_variable_create(impl,
         glsl_vector_type(sample->src[i].ssa->bit_size == 64 ? GLSL_TYPE_UINT64 : GLSL_TYPE_UINT,
                          sample->src[i].ssa->num_components), "call_argument");
   nir_variable *pending = nir_local_variable_create(impl, glsl_bool_type(), "call_pending");
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_store_var(&b, pending, nir_imm_false(&b), 1);
   /* Terminal preparation has already been moved past the search-loop exits.
    * Merge only the suspension operation at the final return boundary. */
   util_dynarray_foreach(&calls, nir_intrinsic_instr *, site) {
      b.cursor = nir_before_instr(&(*site)->instr);
      for (unsigned i = 0; i < ARRAY_SIZE(args); ++i)
         nir_store_var(&b, args[i], (*site)->src[i].ssa,
                       BITFIELD_MASK((*site)->src[i].ssa->num_components));
      nir_store_var(&b, pending, nir_imm_true(&b), 1);
      nir_instr_remove(&(*site)->instr);
   }
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block)
         if (instr->type == nir_instr_type_jump && nir_instr_as_jump(instr)->type == nir_jump_halt)
            nir_instr_as_jump(instr)->type = nir_jump_return;
   }
   nir_progress(true, impl, nir_metadata_none);
   nir_lower_returns(shader);
   b.cursor = nir_after_impl(impl);
   nir_push_if(&b, nir_load_var(&b, pending));
   for (unsigned i = 0; i < ARRAY_SIZE(args); ++i)
      call->src[i] = nir_src_for_ssa(nir_load_var(&b, args[i]));
   nir_builder_instr_insert(&b, &call->instr);
   nir_jump(&b, nir_jump_halt);
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);
   util_dynarray_fini(&calls);
}
#endif

static VkResult
lvp_compile_ray_tracing_stages(struct lvp_pipeline *pipeline,
                               const VkRayTracingPipelineCreateInfoKHR *create_info)
{
   VkResult result = VK_SUCCESS;

   uint32_t i = 0;
   for (; i < create_info->stageCount; i++) {
      nir_shader *nir;
      result = lvp_spirv_to_nir(pipeline, create_info->pNext, create_info->pStages + i, &nir);
      if (result != VK_SUCCESS)
         return result;

      assert(!nir->scratch_size);
      if (nir->info.stage == MESA_SHADER_ANY_HIT ||
          nir->info.stage == MESA_SHADER_CLOSEST_HIT ||
          nir->info.stage == MESA_SHADER_INTERSECTION)
         nir->scratch_size = LVP_RAY_HIT_ATTRIBS_SIZE;

      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
               nir_var_function_temp | nir_var_shader_call_data | nir_var_ray_hit_attrib,
               glsl_get_natural_size_align_bytes);

      NIR_PASS(_, nir, lvp_lower_ray_tracing_derefs);

      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_function_temp, nir_address_format_32bit_offset);

      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lvp_move_ray_tracing_intrinsic,
               nir_metadata_control_flow, NULL);

      pipeline->rt.stages[i] = lvp_create_pipeline_nir(nir);
      if (!pipeline->rt.stages[i]) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         ralloc_free(nir);
         return result;
      }
      if (pipeline->layout)
         pipeline->shaders[nir->info.stage].push_constant_size = pipeline->layout->push_constant_size;
   }

   if (!create_info->pLibraryInfo)
      return result;

   for (uint32_t library_index = 0; library_index < create_info->pLibraryInfo->libraryCount; library_index++) {
      VK_FROM_HANDLE(lvp_pipeline, library, create_info->pLibraryInfo->pLibraries[library_index]);
      for (uint32_t stage_index = 0; stage_index < library->rt.stage_count; stage_index++) {
         lvp_pipeline_nir_ref(pipeline->rt.stages + i, library->rt.stages[stage_index]);
         if (pipeline->layout) {
            gl_shader_stage stage = pipeline->rt.stages[i]->nir->info.stage;
            pipeline->shaders[stage].push_constant_size = pipeline->layout->push_constant_size;
         }
         i++;
      }
   }

   return result;
}

static nir_def *
lvp_load_trace_ray_command_field(nir_builder *b, uint32_t command_offset,
                                 uint32_t num_components, uint32_t bit_size)
{
   return nir_load_ssbo(b, num_components, bit_size, nir_imm_int(b, 0),
                        nir_imm_int(b, command_offset), .access = ACCESS_NON_WRITEABLE);
}

struct lvp_sbt_entry {
   nir_def *value;
   nir_def *shader_record_ptr;
};

static struct lvp_sbt_entry
lvp_load_sbt_entry(nir_builder *b, nir_def *index,
                   uint32_t command_offset, uint32_t index_offset)
{
   nir_def *addr = lvp_load_trace_ray_command_field(b, command_offset, 1, 64);

   if (index) {
      /* The 32 high bits of stride can be ignored. */
      nir_def *stride = lvp_load_trace_ray_command_field(
         b, command_offset + sizeof(VkDeviceSize) * 2, 1, 32);
      addr = nir_iadd(b, addr, nir_u2u64(b, nir_imul(b, index, stride)));
   }

   return (struct lvp_sbt_entry) {
      .value = nir_build_load_global(b, 1, 32, nir_iadd_imm(b, addr, index_offset)),
      .shader_record_ptr = nir_iadd_imm(b, addr, LVP_RAY_TRACING_GROUP_HANDLE_SIZE),
   };
}

struct lvp_ray_traversal_state {
   nir_variable *origin;
   nir_variable *dir;
   nir_variable *inv_dir;
   nir_variable *bvh_base;
   nir_variable *current_node;
   nir_variable *stack_base;
   nir_variable *stack_ptr;
   nir_variable *stack;
   nir_variable *hit;

   nir_variable *instance_addr;
   nir_variable *sbt_offset_and_flags;
};

struct lvp_ray_tracing_state {
   nir_variable *bvh_base;
   nir_variable *flags;
   nir_variable *cull_mask;
   nir_variable *sbt_offset;
   nir_variable *sbt_stride;
   nir_variable *miss_index;
   nir_variable *origin;
   nir_variable *tmin;
   nir_variable *dir;
   nir_variable *tmax;

   nir_variable *instance_addr;
   nir_variable *primitive_id;
   nir_variable *geometry_id_and_flags;
   nir_variable *hit_kind;
   nir_variable *sbt_index;

   /* Hardware-traversal (RTU) path only: the RTU returns the instance custom
    * index as a value, not the instance-node device address the software path
    * reads it from, so stage it here and short-circuit the intrinsic. */
   nir_variable *rtu_instance_custom_index;
   nir_variable *rtu_instance_id;

   nir_variable *shader_record_ptr;
   nir_variable *stack_ptr;
   nir_variable *shader_call_data_offset;

   nir_variable *accept;
   nir_variable *terminate;
   nir_variable *opaque;

   struct lvp_ray_traversal_state traversal;
};

struct lvp_rt_function {
   nir_shader *stage;
   nir_function *function;
   unsigned trace_depth;
   nir_shader *ahit;
   unsigned resume_base;
};

struct lvp_callable_stage {
   nir_shader *stage;
   nir_shader **resumes;
   unsigned resume_count;
   unsigned entry;
   unsigned spill_base;
   nir_variable *live_registers;
};

struct lvp_ray_tracing_pipeline_compiler {
   struct lvp_pipeline *pipeline;
   VkPipelineCreateFlags2KHR flags;

   struct lvp_ray_tracing_state state;

   struct util_dynarray functions;
   unsigned trace_depth;
   unsigned max_trace_depth;
   nir_shader *isec_ahit;
   struct lvp_callable_stage *callables;
   nir_variable *callable_pc;
#ifdef VP_HACKRTCORE
   nir_variable *raygen_pc, *raygen_resume;
   nir_variable *trace_args[11];
#endif

   uint32_t raygen_size;
   uint32_t ahit_size;
   uint32_t chit_size;
   uint32_t miss_size;
   uint32_t isec_size;
   uint32_t callable_size;
};

static uint32_t
lvp_ray_tracing_pipeline_compiler_get_stack_size(
   struct lvp_ray_tracing_pipeline_compiler *compiler, nir_function *function)
{
   util_dynarray_foreach(&compiler->functions, struct lvp_rt_function, entry)
      if (entry->function == function)
         return entry->stage->scratch_size;
#ifdef VP_HACKRTCORE
   if (function->is_entrypoint)
      return compiler->raygen_size;
#endif
   return 0;
}

static void
lvp_ray_tracing_state_init(nir_shader *nir, struct lvp_ray_tracing_state *state)
{
   state->bvh_base = nir_variable_create(nir, nir_var_shader_temp, glsl_uint64_t_type(), "bvh_base");
   state->flags = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "flags");
   state->cull_mask = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "cull_mask");
   state->sbt_offset = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "sbt_offset");
   state->sbt_stride = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "sbt_stride");
   state->miss_index = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "miss_index");
   state->origin = nir_variable_create(nir, nir_var_shader_temp, glsl_vec_type(3), "origin");
   state->tmin = nir_variable_create(nir, nir_var_shader_temp, glsl_float_type(), "tmin");
   state->dir = nir_variable_create(nir, nir_var_shader_temp, glsl_vec_type(3), "dir");
   state->tmax = nir_variable_create(nir, nir_var_shader_temp, glsl_float_type(), "tmax");

   state->instance_addr = nir_variable_create(nir, nir_var_shader_temp, glsl_uint64_t_type(), "instance_addr");
   state->primitive_id = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "primitive_id");
   state->geometry_id_and_flags = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "geometry_id_and_flags");
   state->hit_kind = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "hit_kind");
   state->sbt_index = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "sbt_index");
   state->rtu_instance_custom_index = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "rtu_instance_custom_index");
   state->rtu_instance_id = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "rtu_instance_id");

   state->shader_record_ptr = nir_variable_create(nir, nir_var_shader_temp, glsl_uint64_t_type(), "shader_record_ptr");
   state->stack_ptr = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "stack_ptr");
   state->shader_call_data_offset = nir_variable_create(nir, nir_var_shader_temp, glsl_uint_type(), "shader_call_data_offset");

   state->accept = nir_variable_create(nir, nir_var_shader_temp, glsl_bool_type(), "accept");
   state->terminate = nir_variable_create(nir, nir_var_shader_temp, glsl_bool_type(), "terminate");
   state->opaque = nir_variable_create(nir, nir_var_shader_temp, glsl_bool_type(), "opaque");
}

static void
lvp_ray_traversal_state_init(nir_function_impl *impl, struct lvp_ray_traversal_state *state)
{
   state->origin = nir_local_variable_create(impl, glsl_vec_type(3), "traversal.origin");
   state->dir = nir_local_variable_create(impl, glsl_vec_type(3), "traversal.dir");
   state->inv_dir = nir_local_variable_create(impl, glsl_vec_type(3), "traversal.inv_dir");
   state->bvh_base = nir_local_variable_create(impl, glsl_uint64_t_type(), "traversal.bvh_base");
   state->current_node = nir_local_variable_create(impl, glsl_uint_type(), "traversal.current_node");
   state->stack_base = nir_local_variable_create(impl, glsl_uint_type(), "traversal.stack_base");
   state->stack_ptr = nir_local_variable_create(impl, glsl_uint_type(), "traversal.stack_ptr");
   state->stack = nir_local_variable_create(impl, glsl_array_type(glsl_uint_type(), 24 * 2, 0), "traversal.stack");
   state->hit = nir_local_variable_create(impl, glsl_bool_type(), "traversal.hit");

   state->instance_addr = nir_local_variable_create(impl, glsl_uint64_t_type(), "traversal.instance_addr");
   state->sbt_offset_and_flags = nir_local_variable_create(impl, glsl_uint_type(), "traversal.sbt_offset_and_flags");
}

static nir_function *
lvp_call_ray_tracing_stage(nir_builder *b, struct lvp_ray_tracing_pipeline_compiler *compiler, nir_shader *stage)
{
   nir_function *function = NULL;
   nir_shader *ahit = stage->info.stage == MESA_SHADER_INTERSECTION ? compiler->isec_ahit : NULL;
   util_dynarray_foreach(&compiler->functions, struct lvp_rt_function, entry) {
      if (entry->stage == stage && entry->trace_depth == compiler->trace_depth &&
          entry->ahit == ahit) {
         function = entry->function;
         break;
      }
   }
   if (!function) {
      nir_function_impl *stage_entrypoint = nir_shader_get_entrypoint(stage);
      nir_function_impl *copy = nir_function_impl_clone(b->shader, stage_entrypoint);

      struct hash_table *var_remap = _mesa_pointer_hash_table_create(NULL);

      nir_foreach_block(block, copy) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type != nir_deref_type_var ||
                deref->var->data.mode == nir_var_function_temp)
               continue;

            struct hash_entry *entry =
               _mesa_hash_table_search(var_remap, deref->var);
            if (!entry) {
               nir_variable *new_var = nir_variable_clone(deref->var, b->shader);
               nir_shader_add_variable(b->shader, new_var);
               entry = _mesa_hash_table_insert(var_remap,
                                               deref->var, new_var);
            }
            deref->var = entry->data;
         }
      }

      function = nir_function_create(
         b->shader, _mesa_shader_stage_to_string(stage->info.stage));
      nir_function_set_impl(function, copy);

      ralloc_free(var_remap);

      struct lvp_rt_function entry = {
         .stage = stage, .function = function,
         .trace_depth = compiler->trace_depth, .ahit = ahit,
      };
      for (unsigned i = 0; i < compiler->pipeline->rt.stage_count; i++) {
         struct lvp_callable_stage *callable = &compiler->callables[i];
         if (callable->stage == stage)
            entry.resume_base = callable->entry + 1;
         for (unsigned j = 0; j < callable->resume_count; j++)
            if (callable->resumes[j] == stage)
               entry.resume_base = callable->entry + 1;
      }
      util_dynarray_append(&compiler->functions, struct lvp_rt_function, entry);
   }

   nir_build_call(b, function, 0, NULL);

   switch(stage->info.stage) {
   case MESA_SHADER_RAYGEN:
      compiler->raygen_size = MAX2(compiler->raygen_size, stage->scratch_size);
      break;
   case MESA_SHADER_ANY_HIT:
      compiler->ahit_size = MAX2(compiler->ahit_size, stage->scratch_size);
      break;
   case MESA_SHADER_CLOSEST_HIT:
      compiler->chit_size = MAX2(compiler->chit_size, stage->scratch_size);
      break;
   case MESA_SHADER_MISS:
      compiler->miss_size = MAX2(compiler->miss_size, stage->scratch_size);
      break;
   case MESA_SHADER_INTERSECTION:
      compiler->isec_size = MAX2(compiler->isec_size, stage->scratch_size);
      break;
   case MESA_SHADER_CALLABLE:
      compiler->callable_size = MAX2(compiler->callable_size, stage->scratch_size);
      break;
   default:
      unreachable("Invalid ray tracing stage");
      break;
   }
   return function;
}

/* Frames contain the caller's base, continuation, shader record and argument.
 * The compiler-provided scratch offsets occupy the preceding frame body. */
#define LVP_CALL_FRAME_HEADER_SIZE 32

static void
lvp_callable_push(nir_builder *b, struct lvp_ray_tracing_pipeline_compiler *compiler,
                  nir_intrinsic_instr *instr, unsigned stack_size, unsigned resume)
{
   struct lvp_ray_tracing_state *state = &compiler->state;
   nir_def *base = nir_load_var(b, state->stack_ptr);
   nir_def *header = nir_iadd_imm(b, base, ALIGN(stack_size, 16));
   nir_store_scratch(b, nir_vec2(b, base, nir_imm_int(b, resume)), header,
                     .align_mul = 4, .write_mask = 3);
   nir_store_scratch(b, nir_load_var(b, state->shader_record_ptr),
                     nir_iadd_imm(b, header, 8), .align_mul = 8, .write_mask = 1);
   nir_store_scratch(b, nir_load_var(b, state->shader_call_data_offset),
                     nir_iadd_imm(b, header, 16), .align_mul = 4, .write_mask = 1);
   nir_def *next = nir_iadd_imm(b, header, LVP_CALL_FRAME_HEADER_SIZE);
   nir_store_var(b, state->stack_ptr, next, 1);
   nir_store_var(b, state->shader_call_data_offset,
                 nir_isub(b, nir_iadd(b, base, instr->src[1].ssa), next), 1);
   struct lvp_sbt_entry callable_entry = lvp_load_sbt_entry(
      b, instr->src[0].ssa,
      offsetof(VkTraceRaysIndirectCommand2KHR, callableShaderBindingTableAddress),
      offsetof(struct lvp_ray_tracing_group_handle, index));
   nir_store_var(b, state->shader_record_ptr, callable_entry.shader_record_ptr, 1);
   for (uint32_t i = 0; i < compiler->pipeline->rt.group_count; i++) {
      struct lvp_ray_tracing_group *group = compiler->pipeline->rt.groups + i;
      if (group->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;
      struct lvp_callable_stage *callable = &compiler->callables[group->recursive_index];
      if (!callable->stage || callable->stage->info.stage != MESA_SHADER_CALLABLE)
         continue;
      nir_push_if(b, nir_ieq_imm(b, callable_entry.value, group->handle.index));
      nir_store_var(b, compiler->callable_pc, nir_imm_int(b, callable->entry), 1);
      nir_pop_if(b, NULL);
   }
}

static void
lvp_callable_pop(nir_builder *b, struct lvp_ray_tracing_pipeline_compiler *compiler)
{
   struct lvp_ray_tracing_state *state = &compiler->state;
   nir_def *header = nir_iadd_imm(b, nir_load_var(b, state->stack_ptr), -LVP_CALL_FRAME_HEADER_SIZE);
   nir_def *saved = nir_load_scratch(b, 2, 32, header, .align_mul = 4);
   nir_store_var(b, state->shader_record_ptr,
                 nir_load_scratch(b, 1, 64, nir_iadd_imm(b, header, 8), .align_mul = 8), 1);
   nir_store_var(b, state->shader_call_data_offset,
                 nir_load_scratch(b, 1, 32, nir_iadd_imm(b, header, 16), .align_mul = 4), 1);
   nir_store_var(b, state->stack_ptr, nir_channel(b, saved, 0), 1);
   nir_store_var(b, compiler->callable_pc, nir_channel(b, saved, 1), 1);
}

static void
lvp_execute_callable(nir_builder *b, struct lvp_ray_tracing_pipeline_compiler *compiler,
                     nir_intrinsic_instr *instr)
{
   unsigned size = lvp_ray_tracing_pipeline_compiler_get_stack_size(compiler, b->impl->function);
   lvp_callable_push(b, compiler, instr, size, 0);
   nir_push_loop(b);
   nir_def *pc = nir_load_var(b, compiler->callable_pc);
   nir_push_if(b, nir_ieq_imm(b, pc, 0));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   for (unsigned i = 0; i < compiler->pipeline->rt.stage_count; i++) {
      struct lvp_callable_stage *callable = &compiler->callables[i];
      if (!callable->stage || callable->stage->info.stage != MESA_SHADER_CALLABLE)
         continue;
      for (unsigned j = 0; j <= callable->resume_count; j++) {
         nir_push_if(b, nir_ieq_imm(b, pc, callable->entry + j));
         lvp_call_ray_tracing_stage(b, compiler,
                                   j ? callable->resumes[j - 1] : callable->stage);
         nir_pop_if(b, NULL);
      }
   }
   nir_pop_loop(b, NULL);
}

struct lvp_lower_isec_intrinsic_state {
   struct lvp_ray_tracing_pipeline_compiler *compiler;
   nir_shader *ahit;
   nir_function *function;
};

static bool
lvp_lower_isec_intrinsic(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   if (instr->intrinsic != nir_intrinsic_report_ray_intersection)
      return false;

   struct lvp_lower_isec_intrinsic_state *isec_state = data;
   if (b->impl->function != isec_state->function)
      return false;
   struct lvp_ray_tracing_pipeline_compiler *compiler = isec_state->compiler;
   struct lvp_ray_tracing_state *state = &compiler->state;

   b->cursor = nir_after_instr(&instr->instr);

   nir_def *t = instr->src[0].ssa;
   nir_def *hit_kind = instr->src[1].ssa;

   nir_def *prev_accept = nir_load_var(b, state->accept);
   nir_def *prev_tmax = nir_load_var(b, state->tmax);
   nir_def *prev_hit_kind = nir_load_var(b, state->hit_kind);

   nir_variable *commit = nir_local_variable_create(b->impl, glsl_bool_type(), "commit");
   nir_store_var(b, commit, nir_imm_false(b), 0x1);

   nir_def *in_range = nir_iand(b, nir_fge(b, t, nir_load_var(b, state->tmin)), nir_fge(b, nir_load_var(b, state->tmax), t));
   nir_def *terminated = nir_iand(b, nir_load_var(b, state->terminate), nir_load_var(b, state->accept));
   nir_push_if(b, nir_iand(b, in_range, nir_inot(b, terminated)));
   {
      nir_store_var(b, state->accept, nir_imm_true(b), 0x1);

      nir_store_var(b, state->tmax, t, 1);
      nir_store_var(b, state->hit_kind, hit_kind, 1);

      if (isec_state->ahit) {
         nir_def *prev_terminate = nir_load_var(b, state->terminate);
         nir_store_var(b, state->terminate, nir_imm_false(b), 0x1);

         nir_push_if(b, nir_inot(b, nir_load_var(b, state->opaque)));
         {
            lvp_call_ray_tracing_stage(b, compiler, isec_state->ahit);
         }
         nir_pop_if(b, NULL);

         nir_def *terminate = nir_load_var(b, state->terminate);
         nir_store_var(b, state->terminate, nir_ior(b, terminate, prev_terminate), 0x1);

         nir_push_if(b, terminate);
         nir_jump(b, nir_jump_return);
         nir_pop_if(b, NULL);
      }

      nir_push_if(b, nir_load_var(b, state->accept));
      {
         nir_store_var(b, commit, nir_imm_true(b), 0x1);
      }
      nir_push_else(b, NULL);
      {
         nir_store_var(b, state->accept, prev_accept, 0x1);
         nir_store_var(b, state->tmax, prev_tmax, 1);
         nir_store_var(b, state->hit_kind, prev_hit_kind, 1);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   nir_def_replace(&instr->def, nir_load_var(b, commit));

   return true;
}

static void
lvp_handle_aabb_intersection(nir_builder *b, struct lvp_leaf_intersection *intersection,
                             const struct lvp_ray_traversal_args *args,
                             const struct lvp_ray_flags *ray_flags)
{
   struct lvp_ray_tracing_pipeline_compiler *compiler = args->data;
   struct lvp_ray_tracing_state *state = &compiler->state;

   bool has_isec = false;
   for (uint32_t i = 0; i < compiler->pipeline->rt.group_count; i++) {
      struct lvp_ray_tracing_group *group = compiler->pipeline->rt.groups + i;
      if (group->isec_index != VK_SHADER_UNUSED_KHR) {
         has_isec = true;
         break;
      }
   }

   if (!has_isec)
      return;

   nir_store_var(b, state->accept, nir_imm_false(b), 0x1);
   nir_store_var(b, state->terminate, ray_flags->terminate_on_first_hit, 0x1);
   nir_store_var(b, state->opaque, intersection->opaque, 0x1);

   nir_def *prev_instance_addr = nir_load_var(b, state->instance_addr);
   nir_def *prev_primitive_id = nir_load_var(b, state->primitive_id);
   nir_def *prev_geometry_id_and_flags = nir_load_var(b, state->geometry_id_and_flags);

   nir_store_var(b, state->instance_addr, nir_load_var(b, state->traversal.instance_addr), 0x1);
   nir_store_var(b, state->primitive_id, intersection->primitive_id, 0x1);
   nir_store_var(b, state->geometry_id_and_flags, intersection->geometry_id_and_flags, 0x1);

   nir_def *geometry_id = nir_iand_imm(b, intersection->geometry_id_and_flags, 0xfffffff);
   nir_def *sbt_index =
      nir_iadd(b,
               nir_iadd(b, nir_load_var(b, state->sbt_offset),
                        nir_iand_imm(b, nir_load_var(b, state->traversal.sbt_offset_and_flags), 0xffffff)),
               nir_imul(b, nir_load_var(b, state->sbt_stride), geometry_id));

   struct lvp_sbt_entry isec_entry = lvp_load_sbt_entry(
      b,
      sbt_index,
      offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress),
      offsetof(struct lvp_ray_tracing_group_handle, index));
   nir_store_var(b, compiler->state.shader_record_ptr, isec_entry.shader_record_ptr, 0x1);

   for (uint32_t i = 0; i < compiler->pipeline->rt.group_count; i++) {
      struct lvp_ray_tracing_group *group = compiler->pipeline->rt.groups + i;
      if (group->isec_index == VK_SHADER_UNUSED_KHR)
         continue;

      nir_shader *stage = compiler->pipeline->rt.stages[group->isec_index]->nir;

      nir_shader *ahit_stage = group->ahit_index == VK_SHADER_UNUSED_KHR ? NULL :
         compiler->pipeline->rt.stages[group->ahit_index]->nir;
      compiler->isec_ahit = ahit_stage;

      nir_push_if(b, nir_ieq_imm(b, isec_entry.value, group->handle.index));
      nir_function *function = lvp_call_ray_tracing_stage(b, compiler, stage);
      nir_pop_if(b, NULL);
      compiler->isec_ahit = NULL;

      struct lvp_lower_isec_intrinsic_state isec_state = {
         .compiler = compiler,
         .ahit = ahit_stage,
         .function = function,
      };
      nir_shader_intrinsics_pass(b->shader, lvp_lower_isec_intrinsic,
                                 nir_metadata_none, &isec_state);
   }

   nir_push_if(b, nir_load_var(b, state->accept));
   {
      nir_store_var(b, state->sbt_index, sbt_index, 0x1);
      nir_store_var(b, state->traversal.hit, nir_imm_true(b), 0x1);

#ifndef VP_HACKRTCORE
      nir_break_if(b, nir_load_var(b, state->terminate));
#endif
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, state->instance_addr, prev_instance_addr, 0x1);
      nir_store_var(b, state->primitive_id, prev_primitive_id, 0x1);
      nir_store_var(b, state->geometry_id_and_flags, prev_geometry_id_and_flags, 0x1);
   }
   nir_pop_if(b, NULL);
}

static void
lvp_handle_triangle_intersection(nir_builder *b,
                                 struct lvp_triangle_intersection *intersection,
                                 const struct lvp_ray_traversal_args *args,
                                 const struct lvp_ray_flags *ray_flags)
{
   struct lvp_ray_tracing_pipeline_compiler *compiler = args->data;
   struct lvp_ray_tracing_state *state = &compiler->state;

   nir_store_var(b, state->accept, nir_imm_true(b), 0x1);
   nir_store_var(b, state->terminate, ray_flags->terminate_on_first_hit, 0x1);

   nir_def *barycentrics_offset = nir_load_var(b, state->stack_ptr);

   nir_def *prev_tmax = nir_load_var(b, state->tmax);
   nir_def *prev_instance_addr = nir_load_var(b, state->instance_addr);
   nir_def *prev_primitive_id = nir_load_var(b, state->primitive_id);
   nir_def *prev_geometry_id_and_flags = nir_load_var(b, state->geometry_id_and_flags);
   nir_def *prev_hit_kind = nir_load_var(b, state->hit_kind);
   nir_def *prev_barycentrics = nir_load_scratch(b, 2, 32, barycentrics_offset);

   nir_store_var(b, state->tmax, intersection->t, 0x1);
   nir_store_var(b, state->instance_addr, nir_load_var(b, state->traversal.instance_addr), 0x1);
   nir_store_var(b, state->primitive_id, intersection->base.primitive_id, 0x1);
   nir_store_var(b, state->geometry_id_and_flags, intersection->base.geometry_id_and_flags, 0x1);
   nir_store_var(b, state->hit_kind,
                 nir_bcsel(b, intersection->frontface, nir_imm_int(b, 0xFE), nir_imm_int(b, 0xFF)), 0x1);

   nir_store_scratch(b, intersection->barycentrics, barycentrics_offset);

   nir_def *geometry_id = nir_iand_imm(b, intersection->base.geometry_id_and_flags, 0xfffffff);
   nir_def *sbt_index =
      nir_iadd(b,
               nir_iadd(b, nir_load_var(b, state->sbt_offset),
                        nir_iand_imm(b, nir_load_var(b, state->traversal.sbt_offset_and_flags), 0xffffff)),
               nir_imul(b, nir_load_var(b, state->sbt_stride), geometry_id));

   nir_push_if(b, nir_inot(b, intersection->base.opaque));
   {
      struct lvp_sbt_entry ahit_entry = lvp_load_sbt_entry(
         b,
         sbt_index,
         offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress),
         offsetof(struct lvp_ray_tracing_group_handle, index));
      nir_store_var(b, compiler->state.shader_record_ptr, ahit_entry.shader_record_ptr, 0x1);

      for (uint32_t i = 0; i < compiler->pipeline->rt.group_count; i++) {
         struct lvp_ray_tracing_group *group = compiler->pipeline->rt.groups + i;
         if (group->ahit_index == VK_SHADER_UNUSED_KHR)
            continue;

         nir_shader *stage = compiler->pipeline->rt.stages[group->ahit_index]->nir;

         nir_push_if(b, nir_ieq_imm(b, ahit_entry.value, group->handle.index));
         lvp_call_ray_tracing_stage(b, compiler, stage);
         nir_pop_if(b, NULL);
      }
   }
   nir_pop_if(b, NULL);

   nir_push_if(b, nir_load_var(b, state->accept));
   {
      nir_store_var(b, state->sbt_index, sbt_index, 0x1);
      nir_store_var(b, state->traversal.hit, nir_imm_true(b), 0x1);

#ifndef VP_HACKRTCORE
      nir_break_if(b, nir_load_var(b, state->terminate));
#endif
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, state->tmax, prev_tmax, 0x1);
      nir_store_var(b, state->instance_addr, prev_instance_addr, 0x1);
      nir_store_var(b, state->primitive_id, prev_primitive_id, 0x1);
      nir_store_var(b, state->geometry_id_and_flags, prev_geometry_id_and_flags, 0x1);
      nir_store_var(b, state->hit_kind, prev_hit_kind, 0x1);
      nir_store_scratch(b, prev_barycentrics, barycentrics_offset);
   }
   nir_pop_if(b, NULL);
}

/* Hardware ray traversal on the Vortex RTU.
 *
 * Replaces the software BVH walk with one RTU trace+wait macro-op and writes
 * the SAME closest-hit state the software path produces, so the closest-hit /
 * miss dispatch that follows is identical. This is the RT-pipeline analogue of
 * vortexpipe's VK_KHR_ray_query lowering (vp_nir_lower_ray_tracing_to_rtu):
 * both resolve the whole ray in the RTU. Scope: opaque triangle geometry (the
 * bulk of the suite); any-hit and procedural intersection remain on the
 * software path for now. The acceleration structure is transcoded to the RTU
 * CW-BVH format at launch (vp_launch, has_rtu) exactly as for ray queries, so
 * `accel_struct` is the same handle the software path would walk. */
static void
lvp_trace_ray_rtu(nir_builder *b, struct lvp_ray_tracing_pipeline_compiler *compiler,
                  nir_def *accel_struct, nir_def *flags, nir_def *cull_mask,
                  nir_def *origin, nir_def *tmin, nir_def *dir, nir_def *tmax)
{
   struct lvp_ray_tracing_state *state = &compiler->state;

   /* One XLEN scene register carries the TLAS pointer (low 32 bits). */
   nir_def *scene = accel_struct->bit_size == 32 ? accel_struct : nir_u2u32(b, accel_struct);
   /* Config word: ray_flags (low 16) | cull_mask (high 16), matching the
    * ray-query lowering's lane-3 packing. */
   nir_def *flags_cull =
      nir_ior(b, nir_iand_imm(b, flags, 0xffff),
                 nir_ishl_imm(b, nir_iand_imm(b, cull_mask, 0xff), 16));

   nir_def *handle = nir_vortex_rt_wtrace(b, 32, scene, flags_cull, origin, dir, tmin, tmax);
   nir_def *status = nir_vortex_rt_wait(b, 32, handle);
   nir_def *is_hit = nir_ieq_imm(b, status, VX_RT_STS_DONE_HIT);

   nir_def *t     = nir_vortex_rt_get(b, 32, status, .base = VX_RT_HIT_T);
   nir_def *bary_u = nir_vortex_rt_get(b, 32, status, .base = VX_RT_HIT_BARY_U);
   nir_def *bary_v = nir_vortex_rt_get(b, 32, status, .base = VX_RT_HIT_BARY_V);
   nir_def *prim  = nir_vortex_rt_get(b, 32, status, .base = VX_RT_HIT_PRIMITIVE_ID);
   nir_def *geom  = nir_vortex_rt_get(b, 32, status, .base = VX_RT_HIT_GEOMETRY_INDEX);
   nir_def *cust  = nir_vortex_rt_get(b, 32, status, .base = VX_RT_HIT_INSTANCE_CUSTOM);

   /* tmax holds the committed hit distance on a hit; keep the ray's tmax on a
    * miss (the miss shader reads gl_RayTmaxEXT). */
   nir_store_var(b, state->tmax, nir_bcsel(b, is_hit, t, tmax), 0x1);
   nir_store_var(b, state->primitive_id, prim, 0x1);
   nir_store_var(b, state->geometry_id_and_flags, geom, 0x1);
   nir_store_var(b, state->rtu_instance_custom_index, cust, 0x1);
   /* The RTU does not report triangle facing, so opaque-triangle hits are
    * always reported front-facing (gl_HitKindEXT 0xFE; back-face would be 0xFF). */
   nir_store_var(b, state->hit_kind, nir_imm_int(b, 0xFE), 0x1);
   nir_store_var(b, state->traversal.hit, is_hit, 0x1);

   /* Barycentrics live in scratch at the traversal stack pointer — the exact
    * slot the software triangle handler uses, so the closest-hit's
    * hitAttributeEXT read is unchanged. */
   nir_def *bary_offset = nir_load_var(b, state->stack_ptr);
   nir_store_scratch(b, nir_vec2(b, bary_u, bary_v), bary_offset);

   /* SBT record = sbt_offset + sbt_stride * geometry_index. Per-instance
    * shaderBindingTableRecordOffset (uniform for the single-hit-group opaque
    * scenes targeted here) is not yet plumbed from the RTU. */
   nir_def *sbt_index =
      nir_iadd(b, nir_load_var(b, state->sbt_offset),
               nir_imul(b, nir_load_var(b, state->sbt_stride), geom));
   nir_store_var(b, state->sbt_index, sbt_index, 0x1);
}

#ifdef VP_HACKRTCORE
static nir_def *
lvp_rtu_query_value(nir_builder *b, nir_def *query, nir_ray_query_value value,
                    bool committed)
{
   unsigned components = value == nir_ray_query_value_intersection_barycentrics ? 2 : 1;
   unsigned bits = value == nir_ray_query_value_intersection_front_face ||
                   value == nir_ray_query_value_intersection_candidate_aabb_opaque ? 1 : 32;
   return nir_rq_load(b, components, bits, query,
                     .ray_query_value = value, .committed = committed);
}

static void
lvp_rtu_instance(nir_builder *b, struct lvp_ray_tracing_state *state,
                 nir_def *query, nir_def *scene, bool committed)
{
   nir_def *id = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_instance_id, committed);
   nir_def *table = nir_build_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, scene, RTU_SCENE_INSTANCE_TABLE_OFFSET));
   nir_def *address = nir_iadd(b, scene, nir_u2u64(b,
      nir_iadd(b, table, nir_imul_imm(b, id, RTU_INSTANCE_METADATA_BYTES))));
   nir_store_var(b, state->rtu_instance_id, id, 1);
   nir_store_var(b, state->rtu_instance_custom_index,
      lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_instance_custom_index, committed), 1);
   nir_store_var(b, state->traversal.instance_addr, address, 1);
   nir_store_var(b, state->traversal.sbt_offset_and_flags,
      lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_instance_sbt_index, committed), 1);
}

static bool
lvp_can_keep_traversal_resident(struct lvp_pipeline *pipeline)
{
   for (unsigned i = 0; i < pipeline->rt.stage_count; i++) {
      nir_shader *stage = pipeline->rt.stages[i]->nir;
      if (stage->info.stage != MESA_SHADER_INTERSECTION &&
          stage->info.stage != MESA_SHADER_ANY_HIT &&
          stage->info.stage != MESA_SHADER_CALLABLE)
         continue;
      nir_foreach_function_impl(impl, stage) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
               if (op == nir_intrinsic_trace_ray || nir_intrinsic_is_ray_query(op))
                  return false;
            }
         }
      }
   }
   return true;
}

static void
lvp_trace_ray_query(nir_builder *b, struct lvp_ray_tracing_pipeline_compiler *compiler,
                    nir_def *scene, nir_def *flags, nir_def *mask,
                    nir_def *origin, nir_def *tmin, nir_def *direction, nir_def *tmax)
{
   struct lvp_ray_tracing_state *state = &compiler->state;
   nir_variable *variable = nir_local_variable_create(b->impl, glsl_uint_type(), "traversal_query");
   variable->data.ray_query = true;
   variable->data.rtu_resident_query = lvp_can_keep_traversal_resident(compiler->pipeline);
   nir_def *query = &nir_build_deref_var(b, variable)->def;
   nir_rq_initialize(b, query, scene,
                    nir_ior_imm(b, flags, VX_RT_FLAG_TRACE_PIPELINE),
                    mask, origin, tmin, direction, tmax);
   struct lvp_ray_traversal_args args = { .data = compiler };
   struct lvp_ray_flags ray_flags = {
      .terminate_on_first_hit = nir_test_mask(b, flags, SpvRayFlagsTerminateOnFirstHitKHRMask),
   };
   nir_push_loop(b);
   nir_break_if(b, nir_inot(b, nir_rq_proceed(b, 1, query)));
   lvp_rtu_instance(b, state, query, scene, false);
   struct lvp_leaf_intersection leaf = {
      .primitive_id = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_primitive_index, false),
      .geometry_id_and_flags = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_geometry_index, false),
      .opaque = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_candidate_aabb_opaque, false),
   };
   nir_store_var(b, state->accept, nir_imm_false(b), 1);
   nir_def *type = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_type, false);
   nir_push_if(b, nir_ieq_imm(b, type, 0));
   {
      struct lvp_triangle_intersection triangle = {
         .base = leaf,
         .t = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_t, false),
         .frontface = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_front_face, false),
         .barycentrics = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_barycentrics, false),
      };
      triangle.base.opaque = nir_imm_false(b);
      lvp_handle_triangle_intersection(b, &triangle, &args, &ray_flags);
      nir_push_if(b, nir_load_var(b, state->accept));
      nir_rq_confirm_intersection(b, query);
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      nir_def *attributes[2];
      for (unsigned i = 0; i < 2; ++i)
         attributes[i] = nir_load_scratch(b, 4, 32,
            nir_iadd_imm(b, nir_load_var(b, state->stack_ptr), i * 16));
      lvp_handle_aabb_intersection(b, &leaf, &args, &ray_flags);
      nir_push_if(b, nir_load_var(b, state->accept));
      nir_rq_generate_intersection(b, query, nir_load_var(b, state->tmax));
      nir_push_else(b, NULL);
      for (unsigned i = 0; i < 2; ++i)
         nir_store_scratch(b, attributes[i],
            nir_iadd_imm(b, nir_load_var(b, state->stack_ptr), i * 16));
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
   nir_push_if(b, nir_iand(b, nir_load_var(b, state->accept), nir_load_var(b, state->terminate)));
   nir_rq_terminate(b, query);
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_pop_loop(b, NULL);

   nir_def *committed_type = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_type, true);
   nir_def *hit = nir_ine_imm(b, committed_type, 0);
   nir_store_var(b, state->traversal.hit, hit, 1);
   nir_push_if(b, hit);
   lvp_rtu_instance(b, state, query, scene, true);
   nir_store_var(b, state->instance_addr, nir_load_var(b, state->traversal.instance_addr), 1);
   nir_store_var(b, state->primitive_id,
      lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_primitive_index, true), 1);
   nir_def *geometry = lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_geometry_index, true);
   nir_store_var(b, state->geometry_id_and_flags, geometry, 1);
   nir_store_var(b, state->tmax,
      lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_t, true), 1);
   nir_store_var(b, state->sbt_index, nir_iadd(b,
      nir_iadd(b, nir_load_var(b, state->sbt_offset), nir_load_var(b, state->traversal.sbt_offset_and_flags)),
      nir_imul(b, nir_load_var(b, state->sbt_stride), geometry)), 1);
   nir_push_if(b, nir_ieq_imm(b, committed_type, 1));
   nir_store_var(b, state->hit_kind, nir_bcsel(b,
      lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_front_face, true),
      nir_imm_int(b, 0xfe), nir_imm_int(b, 0xff)), 1);
   nir_store_scratch(b,
      lvp_rtu_query_value(b, query, nir_ray_query_value_intersection_barycentrics, true),
      nir_load_var(b, state->stack_ptr));
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}
#endif

static void
lvp_trace_ray(nir_builder *b, struct lvp_ray_tracing_pipeline_compiler *compiler,
              nir_intrinsic_instr *instr)
{
   struct lvp_ray_tracing_state *state = &compiler->state;
   if (compiler->trace_depth == compiler->max_trace_depth)
      return;
   compiler->trace_depth++;
   nir_variable *saved_vars[] = {
      state->bvh_base, state->flags, state->cull_mask, state->sbt_offset,
      state->sbt_stride, state->miss_index, state->origin, state->tmin,
      state->dir, state->tmax, state->instance_addr, state->primitive_id,
      state->geometry_id_and_flags, state->hit_kind, state->sbt_index,
      state->rtu_instance_custom_index, state->rtu_instance_id,
      state->shader_record_ptr, state->shader_call_data_offset,
      state->accept, state->terminate, state->opaque,
   };
   /* Raygen has no incoming ray state; retain its record and call data. */
   bool raygen = b->impl->function->is_entrypoint;
   util_dynarray_foreach(&compiler->functions, struct lvp_rt_function, entry) {
      if (entry->function == b->impl->function) {
         raygen = entry->stage->info.stage == MESA_SHADER_RAYGEN;
         break;
      }
   }
   nir_def *saved[ARRAY_SIZE(saved_vars)] = {0};
   for (unsigned i = 0; i < ARRAY_SIZE(saved_vars); ++i) {
      if (!raygen || saved_vars[i] == state->shader_record_ptr ||
          saved_vars[i] == state->shader_call_data_offset)
         saved[i] = nir_load_var(b, saved_vars[i]);
   }

   nir_def *accel_struct = instr->src[0].ssa;
   nir_def *flags = instr->src[1].ssa;
   nir_def *cull_mask = instr->src[2].ssa;
   nir_def *sbt_offset = nir_iand_imm(b, instr->src[3].ssa, 0xF);
   nir_def *sbt_stride = nir_iand_imm(b, instr->src[4].ssa, 0xF);
   nir_def *miss_index = nir_iand_imm(b, instr->src[5].ssa, 0xFFFF);
   nir_def *origin = instr->src[6].ssa;
   nir_def *tmin = instr->src[7].ssa;
   nir_def *dir = instr->src[8].ssa;
   nir_def *tmax = instr->src[9].ssa;
   nir_def *payload = instr->src[10].ssa;

   uint32_t stack_size =
      lvp_ray_tracing_pipeline_compiler_get_stack_size(compiler, b->impl->function);
   nir_def *stack_ptr = nir_load_var(b, state->stack_ptr);
   nir_store_var(b, state->stack_ptr, nir_iadd_imm(b, stack_ptr, stack_size), 0x1);

   nir_store_var(b, state->shader_call_data_offset, nir_iadd_imm(b, payload, -stack_size), 0x1);

   lvp_ray_traversal_state_init(b->impl, &state->traversal);

   nir_store_var(b, state->bvh_base, accel_struct, 0x1);
   nir_store_var(b, state->flags, flags, 0x1);
   nir_store_var(b, state->cull_mask, cull_mask, 0x1);
   nir_store_var(b, state->sbt_offset, sbt_offset, 0x1);
   nir_store_var(b, state->sbt_stride, sbt_stride, 0x1);
   nir_store_var(b, state->miss_index, miss_index, 0x1);
   nir_store_var(b, state->origin, origin, 0x7);
   nir_store_var(b, state->tmin, tmin, 0x1);
   nir_store_var(b, state->dir, dir, 0x7);
   nir_store_var(b, state->tmax, tmax, 0x1);

   nir_store_var(b, state->traversal.hit, nir_imm_false(b), 0x1);

#ifdef VP_HACKRTCORE
   nir_push_if(b, nir_ine_imm(b, accel_struct, 0));
   bool rtu_trace = true;
   for (unsigned i = 0; i < compiler->pipeline->rt.group_count; i++) {
      struct lvp_ray_tracing_group *group = &compiler->pipeline->rt.groups[i];
      if (group->ahit_index != VK_SHADER_UNUSED_KHR ||
          group->isec_index != VK_SHADER_UNUSED_KHR) {
         rtu_trace = false;
         break;
      }
   }
   if (rtu_trace)
      lvp_trace_ray_rtu(b, compiler, accel_struct, flags, cull_mask, origin, tmin, dir, tmax);
   else
      lvp_trace_ray_query(b, compiler, accel_struct, flags, cull_mask, origin, tmin, dir, tmax);
   nir_pop_if(b, NULL);
#else
   if (compiler->pipeline->device->pscreen->caps.driver_ray_queries) {
      /* Vortex RTU present: resolve the ray in hardware. */
      nir_push_if(b, nir_ine_imm(b, accel_struct, 0));
      lvp_trace_ray_rtu(b, compiler, accel_struct, flags, cull_mask, origin, tmin, dir, tmax);
      nir_pop_if(b, NULL);
   } else
   {
      nir_store_var(b, state->traversal.bvh_base, accel_struct, 0x1);
      nir_store_var(b, state->traversal.origin, origin, 0x7);
      nir_store_var(b, state->traversal.dir, dir, 0x7);
      nir_store_var(b, state->traversal.inv_dir, nir_frcp(b, dir), 0x7);
      nir_store_var(b, state->traversal.current_node, nir_imm_int(b, LVP_BVH_ROOT_NODE), 0x1);
      nir_store_var(b, state->traversal.stack_base, nir_imm_int(b, -1), 0x1);
      nir_store_var(b, state->traversal.stack_ptr, nir_imm_int(b, 0), 0x1);

      struct lvp_ray_traversal_vars vars = {
         .tmax = nir_build_deref_var(b, state->tmax),
         .origin = nir_build_deref_var(b, state->traversal.origin),
         .dir = nir_build_deref_var(b, state->traversal.dir),
         .inv_dir = nir_build_deref_var(b, state->traversal.inv_dir),
         .bvh_base = nir_build_deref_var(b, state->traversal.bvh_base),
         .current_node = nir_build_deref_var(b, state->traversal.current_node),
         .stack_base = nir_build_deref_var(b, state->traversal.stack_base),
         .stack_ptr = nir_build_deref_var(b, state->traversal.stack_ptr),
         .stack = nir_build_deref_var(b, state->traversal.stack),
         .instance_addr = nir_build_deref_var(b, state->traversal.instance_addr),
         .sbt_offset_and_flags = nir_build_deref_var(b, state->traversal.sbt_offset_and_flags),
      };

      struct lvp_ray_traversal_args args = {
         .root_bvh_base = accel_struct,
         .flags = flags,
         .cull_mask = nir_ishl_imm(b, cull_mask, 24),
         .origin = origin,
         .tmin = tmin,
         .dir = dir,
         .vars = vars,
         .aabb_cb = (compiler->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR) ?
                    NULL : lvp_handle_aabb_intersection,
         .triangle_cb = (compiler->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR) ?
                        NULL : lvp_handle_triangle_intersection,
         .data = compiler,
      };

      nir_push_if(b, nir_ine_imm(b, accel_struct, 0));
      lvp_build_ray_traversal(b, &args);
      nir_pop_if(b, NULL);
   }
#endif

#ifdef VP_HACKRTCORE
   /* The shader dispatcher selects the minimum pending shader ID across the
    * warp, runs that group, and rejoins it before selecting the next group.
    * Keeping this outside the hit/miss branch preserves the shared return
    * boundary when lanes select different stages or skip closest-hit. */
   nir_variable *pending_shader = nir_local_variable_create(
      b->impl, glsl_uint_type(), "pending_shader");
   nir_store_var(b, pending_shader, nir_imm_int(b, UINT32_MAX), 1);
   nir_push_if(b, nir_load_var(b, state->traversal.hit));
   {
      nir_push_if(b, nir_inot(b, nir_test_mask(
         b, flags, SpvRayFlagsSkipClosestHitShaderKHRMask)));
      struct lvp_sbt_entry entry = lvp_load_sbt_entry(
         b, nir_load_var(b, state->sbt_index),
         offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress),
         offsetof(struct lvp_ray_tracing_group_handle, index));
      nir_store_var(b, state->shader_record_ptr, entry.shader_record_ptr, 1);
      nir_store_var(b, pending_shader, entry.value, 1);
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      struct lvp_sbt_entry entry = lvp_load_sbt_entry(
         b, miss_index,
         offsetof(VkTraceRaysIndirectCommand2KHR, missShaderBindingTableAddress),
         offsetof(struct lvp_ray_tracing_group_handle, index));
      nir_store_var(b, state->shader_record_ptr, entry.shader_record_ptr, 1);
      nir_store_var(b, pending_shader, entry.value, 1);
   }
   nir_pop_if(b, NULL);

   nir_push_loop(b);
   nir_def *pending = nir_load_var(b, pending_shader);
   nir_def *selected = nir_reduce(b, pending, .reduction_op = nir_op_umin,
                                  .cluster_size = 0);
   nir_push_if(b, nir_ieq_imm(b, selected, UINT32_MAX));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   for (uint32_t i = 0; i < compiler->pipeline->rt.group_count; i++) {
      struct lvp_ray_tracing_group *group = compiler->pipeline->rt.groups + i;
      if (group->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;
      nir_shader *stage = compiler->pipeline->rt.stages[group->recursive_index]->nir;
      if (stage->info.stage != MESA_SHADER_CLOSEST_HIT &&
          stage->info.stage != MESA_SHADER_MISS)
         continue;
      nir_push_if(b, nir_ieq_imm(b, selected, group->handle.index));
      nir_push_if(b, nir_ieq(b, pending, selected));
      lvp_call_ray_tracing_stage(b, compiler, stage);
      nir_store_var(b, pending_shader, nir_imm_int(b, UINT32_MAX), 1);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);
#else
   nir_push_if(b, nir_load_var(b, state->traversal.hit));
   {
      nir_def *skip_chit = nir_test_mask(b, flags, SpvRayFlagsSkipClosestHitShaderKHRMask);
      nir_push_if(b, nir_inot(b, skip_chit));

      struct lvp_sbt_entry chit_entry = lvp_load_sbt_entry(
         b,
         nir_load_var(b, state->sbt_index),
         offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress),
         offsetof(struct lvp_ray_tracing_group_handle, index));
      nir_store_var(b, compiler->state.shader_record_ptr, chit_entry.shader_record_ptr, 0x1);

      for (uint32_t i = 0; i < compiler->pipeline->rt.group_count; i++) {
         struct lvp_ray_tracing_group *group = compiler->pipeline->rt.groups + i;
         if (group->recursive_index == VK_SHADER_UNUSED_KHR)
            continue;

         nir_shader *stage = compiler->pipeline->rt.stages[group->recursive_index]->nir;
         if (stage->info.stage != MESA_SHADER_CLOSEST_HIT)
            continue;

         nir_push_if(b, nir_ieq_imm(b, chit_entry.value, group->handle.index));
         lvp_call_ray_tracing_stage(b, compiler, stage);
         nir_pop_if(b, NULL);
      }

      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      struct lvp_sbt_entry miss_entry = lvp_load_sbt_entry(
         b,
         miss_index,
         offsetof(VkTraceRaysIndirectCommand2KHR, missShaderBindingTableAddress),
         offsetof(struct lvp_ray_tracing_group_handle, index));
      nir_store_var(b, compiler->state.shader_record_ptr, miss_entry.shader_record_ptr, 0x1);

      for (uint32_t i = 0; i < compiler->pipeline->rt.group_count; i++) {
         struct lvp_ray_tracing_group *group = compiler->pipeline->rt.groups + i;
         if (group->recursive_index == VK_SHADER_UNUSED_KHR)
            continue;

         nir_shader *stage = compiler->pipeline->rt.stages[group->recursive_index]->nir;
         if (stage->info.stage != MESA_SHADER_MISS)
            continue;

         nir_push_if(b, nir_ieq_imm(b, miss_entry.value, group->handle.index));
         lvp_call_ray_tracing_stage(b, compiler, stage);
         nir_pop_if(b, NULL);
      }
   }
   nir_pop_if(b, NULL);

#endif

   nir_store_var(b, state->stack_ptr, stack_ptr, 0x1);
   for (unsigned i = 0; i < ARRAY_SIZE(saved_vars); ++i)
      if (saved[i])
         nir_store_var(b, saved_vars[i], saved[i], BITFIELD_MASK(saved[i]->num_components));
   compiler->trace_depth--;
}

static void
lvp_shader_wto_matrix(nir_builder *b, nir_def *address, nir_def *rows[3])
{
#ifdef VP_HACKRTCORE
   for (unsigned r = 0; r < 3; ++r)
      rows[r] = nir_build_load_global_constant(
         b, 4, 32, nir_iadd_imm(b, address, 48 + r * 16));
#else
   lvp_load_wto_matrix(b, address, NULL, rows);
#endif
}

static bool
lvp_lower_ray_tracing_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct lvp_ray_tracing_pipeline_compiler *compiler = data;
   struct lvp_ray_tracing_state *state = &compiler->state;

   compiler->trace_depth = 0;
   unsigned resume_base = 0;
   bool raygen = false;
   struct lvp_callable_stage *raygen_storage = NULL;
   util_dynarray_foreach(&compiler->functions, struct lvp_rt_function, entry) {
      if (entry->function == b->impl->function) {
         compiler->trace_depth = entry->trace_depth;
         resume_base = entry->resume_base;
         raygen = entry->stage->info.stage == MESA_SHADER_RAYGEN;
         if (raygen) {
            for (unsigned i = 0; i < compiler->pipeline->rt.stage_count; ++i) {
               struct lvp_callable_stage *candidate = &compiler->callables[i];
               if (candidate->stage == entry->stage)
                  raygen_storage = candidate;
               for (unsigned j = 0; j < candidate->resume_count; ++j)
                  if (candidate->resumes[j] == entry->stage)
                     raygen_storage = candidate;
            }
         }
         break;
      }
   }

   if (instr->type == nir_instr_type_jump) {
      nir_jump_instr *jump = nir_instr_as_jump(instr);
      if (jump->type == nir_jump_halt) {
         jump->type = nir_jump_return;
         return true;
      }
      return false;
   } else if (instr->type != nir_instr_type_intrinsic) {
      return false;
   }

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   nir_def *def = NULL;

   b->cursor = nir_before_instr(instr);

#ifdef VP_HACKRTCORE
   /* Raygen live values persist in registers across a shader dispatch. The
    * generic continuation pass names them with constant scratch offsets;
    * turn those slots into SSA-promotable variables before merging stages. */
   if (raygen_storage && raygen_storage->live_registers &&
       (intr->intrinsic == nir_intrinsic_load_scratch ||
        intr->intrinsic == nir_intrinsic_store_scratch)) {
      bool load = intr->intrinsic == nir_intrinsic_load_scratch;
      nir_src offset = intr->src[load ? 0 : 1];
      if (nir_src_is_const(offset) && nir_src_as_uint(offset) >= raygen_storage->spill_base) {
         unsigned word = (nir_src_as_uint(offset) - raygen_storage->spill_base) / 4;
         unsigned components = load ? intr->def.num_components : intr->src[0].ssa->num_components;
         unsigned bits = load ? intr->def.bit_size : intr->src[0].ssa->bit_size;
         nir_def *values[NIR_MAX_VEC_COMPONENTS];
         for (unsigned c = 0; c < components; ++c) {
            if (!load && !(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)))
               continue;
            nir_deref_instr *slot = nir_build_deref_array_imm(b,
               nir_build_deref_var(b, raygen_storage->live_registers), word + c * (bits / 32));
            nir_deref_instr *high = bits == 64 ? nir_build_deref_array_imm(b,
               nir_build_deref_var(b, raygen_storage->live_registers), word + c * 2 + 1) : NULL;
            if (load) {
               values[c] = nir_load_deref(b, slot);
               if (high)
                  values[c] = nir_pack_64_2x32_split(b, values[c], nir_load_deref(b, high));
               else if (bits != 32)
                  values[c] = nir_u2uN(b, values[c], bits);
            } else {
               nir_def *value = nir_channel(b, intr->src[0].ssa, c);
               nir_store_deref(b, slot, high ? nir_unpack_64_2x32_split_x(b, value) : nir_u2u32(b, value), 1);
               if (high)
                  nir_store_deref(b, high, nir_unpack_64_2x32_split_y(b, value), 1);
            }
         }
         if (load)
            nir_def_rewrite_uses(&intr->def, nir_vec(b, values, components));
         nir_instr_remove(instr);
         return true;
      }
   }
#endif

   switch (intr->intrinsic) {
   /* Ray tracing instructions */
#ifdef VP_HACKRTCORE
   case nir_intrinsic_rt_trace_ray:
      if (!raygen) {
         lvp_trace_ray(b, compiler, intr);
         nir_store_var(b, compiler->callable_pc,
            nir_imm_int(b, resume_base + nir_intrinsic_call_idx(intr)), 1);
         break;
      }
      for (unsigned i = 0; i < ARRAY_SIZE(compiler->trace_args); ++i)
         nir_store_var(b, compiler->trace_args[i], intr->src[i].ssa,
                       BITFIELD_MASK(intr->src[i].ssa->num_components));
      nir_store_var(b, compiler->raygen_resume,
         nir_imm_int(b, resume_base + nir_intrinsic_call_idx(intr)), 1);
      nir_store_var(b, compiler->raygen_pc, nir_imm_int(b, UINT32_MAX - 1), 1);
      break;
#endif
   case nir_intrinsic_rt_execute_callable:
#ifdef VP_HACKRTCORE
      if (raygen) {
         lvp_execute_callable(b, compiler, intr);
         nir_store_var(b, compiler->raygen_pc,
            nir_imm_int(b, resume_base + nir_intrinsic_call_idx(intr)), 1);
         break;
      }
#endif
      lvp_callable_push(b, compiler, intr, nir_intrinsic_stack_size(intr),
                        resume_base + nir_intrinsic_call_idx(intr));
      break;
   case nir_intrinsic_rt_resume:
      break;
   case nir_intrinsic_rt_return_amd:
#ifdef VP_HACKRTCORE
      if (raygen) {
         nir_store_var(b, compiler->raygen_pc, nir_imm_int(b, UINT32_MAX), 1);
         break;
      }
#endif
      lvp_callable_pop(b, compiler);
      break;
   case nir_intrinsic_execute_callable:
      lvp_execute_callable(b, compiler, intr);
      break;
   case nir_intrinsic_trace_ray:
      lvp_trace_ray(b, compiler, intr);
      break;
   case nir_intrinsic_ignore_ray_intersection: {
      nir_store_var(b, state->accept, nir_imm_false(b), 0x1);

      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_terminate_ray: {
      nir_store_var(b, state->accept, nir_imm_true(b), 0x1);
      nir_store_var(b, state->terminate, nir_imm_true(b), 0x1);

      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;
   }
   /* Ray tracing system values */
   case nir_intrinsic_load_ray_launch_id:
      def = nir_load_global_invocation_id(b, 32);
      break;
   case nir_intrinsic_load_ray_launch_size:
      def = lvp_load_trace_ray_command_field(
         b, offsetof(VkTraceRaysIndirectCommand2KHR, width), 3, 32);
      break;
   case nir_intrinsic_load_shader_record_ptr:
      def = nir_load_var(b, state->shader_record_ptr);
      break;
   case nir_intrinsic_load_ray_t_min:
      def = nir_load_var(b, state->tmin);
      break;
   case nir_intrinsic_load_ray_t_max:
      def = nir_load_var(b, state->tmax);
      break;
   case nir_intrinsic_load_ray_world_origin:
      def = nir_load_var(b, state->origin);
      break;
   case nir_intrinsic_load_ray_world_direction:
      def = nir_load_var(b, state->dir);
      break;
   case nir_intrinsic_load_ray_instance_custom_index: {
      if (compiler->pipeline->device->pscreen->caps.driver_ray_queries) {
         /* RTU path: the custom index was staged from the RTU hit, not fetched
          * from an instance node the software walker would have visited. */
         def = nir_iand_imm(b, nir_load_var(b, state->rtu_instance_custom_index), 0xFFFFFF);
         break;
      }
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *custom_instance_and_mask = nir_build_load_global(
         b, 1, 32,
         nir_iadd_imm(b, instance_node_addr, offsetof(struct lvp_bvh_instance_node, custom_instance_and_mask)));
      def = nir_iand_imm(b, custom_instance_and_mask, 0xFFFFFF);
      break;
   }
   case nir_intrinsic_load_primitive_id:
      def = nir_load_var(b, state->primitive_id);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      def = nir_load_var(b, state->geometry_id_and_flags);
      def = nir_iand_imm(b, def, 0xFFFFFFF);
      break;
   case nir_intrinsic_load_instance_id: {
#ifdef VP_HACKRTCORE
      def = nir_load_var(b, state->rtu_instance_id);
#else
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      def = nir_build_load_global(
         b, 1, 32, nir_iadd_imm(b, instance_node_addr, offsetof(struct lvp_bvh_instance_node, instance_id)));
#endif
      break;
   }
   case nir_intrinsic_load_ray_flags:
      def = nir_load_var(b, state->flags);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      def = nir_load_var(b, state->hit_kind);
      break;
   case nir_intrinsic_load_ray_world_to_object: {
      unsigned c = nir_intrinsic_column(intr);
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *wto_matrix[3];
      lvp_shader_wto_matrix(b, instance_node_addr, wto_matrix);

      nir_def *vals[3];
      for (unsigned i = 0; i < 3; ++i)
         vals[i] = nir_channel(b, wto_matrix[i], c);

      def = nir_vec(b, vals, 3);
      break;
   }
   case nir_intrinsic_load_ray_object_to_world: {
      unsigned c = nir_intrinsic_column(intr);
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *rows[3];
      unsigned matrix_offset = 0;
#ifndef VP_HACKRTCORE
      matrix_offset = offsetof(struct lvp_bvh_instance_node, otw_matrix);
#endif
      for (unsigned r = 0; r < 3; ++r)
         rows[r] = nir_build_load_global_constant(
            b, 4, 32,
            nir_iadd_imm(b, instance_node_addr, matrix_offset + r * 16));
      def = nir_vec3(b, nir_channel(b, rows[0], c), nir_channel(b, rows[1], c), nir_channel(b, rows[2], c));
      break;
   }
   case nir_intrinsic_load_ray_object_origin: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *wto_matrix[3];
      lvp_shader_wto_matrix(b, instance_node_addr, wto_matrix);
      def = lvp_mul_vec3_mat(b, nir_load_var(b, state->origin), wto_matrix, true);
      break;
   }
   case nir_intrinsic_load_ray_object_direction: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *wto_matrix[3];
      lvp_shader_wto_matrix(b, instance_node_addr, wto_matrix);
      def = lvp_mul_vec3_mat(b, nir_load_var(b, state->dir), wto_matrix, false);
      break;
   }
   case nir_intrinsic_load_cull_mask:
      def = nir_iand_imm(b, nir_load_var(b, state->cull_mask), 0xFF);
      break;
   /* Ray tracing stack lowering */
   case nir_intrinsic_load_scratch: {
      nir_src_rewrite(&intr->src[0], nir_iadd(b, nir_load_var(b, state->stack_ptr), intr->src[0].ssa));
      return true;
   }
   case nir_intrinsic_store_scratch: {
      nir_src_rewrite(&intr->src[1], nir_iadd(b, nir_load_var(b, state->stack_ptr), intr->src[1].ssa));
      return true;
   }
   case nir_intrinsic_load_ray_triangle_vertex_positions: {
      def = lvp_load_vertex_position(
         b, nir_load_var(b, state->instance_addr), nir_load_var(b, state->primitive_id),
         nir_intrinsic_column(intr));
      break;
   }
   /* Internal system values */
   case nir_intrinsic_load_shader_call_data_offset_lvp:
      def = nir_load_var(b, state->shader_call_data_offset);
      break;
   default:
      return false;
   }

   if (def)
      nir_def_rewrite_uses(&intr->def, def);
   nir_instr_remove(instr);

   return true;
}

static bool
lvp_lower_ray_tracing_stack_base(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   if (instr->intrinsic != nir_intrinsic_load_ray_tracing_stack_base_lvp)
      return false;

   b->cursor = nir_after_instr(&instr->instr);

   nir_def_replace(&instr->def, nir_imm_int(b, b->shader->scratch_size));

   return true;
}

static void
lvp_compile_ray_tracing_pipeline(struct lvp_pipeline *pipeline,
                                 const VkRayTracingPipelineCreateInfoKHR *create_info)
{
   nir_builder _b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE,
      pipeline->device->pscreen->get_compiler_options(pipeline->device->pscreen, PIPE_SHADER_IR_NIR, MESA_SHADER_COMPUTE),
      "ray tracing pipeline");
   nir_builder *b = &_b;

#ifdef VP_HACKRTCORE
   b->shader->info.workgroup_size[0] = 32;
#else
   b->shader->info.workgroup_size[0] = 8;
#endif

   struct lvp_ray_tracing_pipeline_compiler compiler = {
      .pipeline = pipeline,
      .flags = vk_rt_pipeline_create_flags(create_info),
      .max_trace_depth = create_info->maxPipelineRayRecursionDepth,
   };
   lvp_ray_tracing_state_init(b->shader, &compiler.state);
   util_dynarray_init(&compiler.functions, NULL);
   compiler.callable_pc = nir_variable_create(b->shader, nir_var_shader_temp,
                                              glsl_uint_type(), "callable_pc");
#ifdef VP_HACKRTCORE
   compiler.raygen_pc = nir_variable_create(b->shader, nir_var_shader_temp,
                                            glsl_uint_type(), "raygen_pc");
   compiler.raygen_resume = nir_variable_create(b->shader, nir_var_shader_temp,
                                                glsl_uint_type(), "raygen_resume");
   const struct glsl_type *argument_types[] = {
      glsl_uint64_t_type(), glsl_uint_type(), glsl_uint_type(),
      glsl_uint_type(), glsl_uint_type(), glsl_uint_type(),
      glsl_vec_type(3), glsl_float_type(), glsl_vec_type(3),
      glsl_float_type(), glsl_uint_type(),
   };
   for (unsigned i = 0; i < ARRAY_SIZE(argument_types); ++i)
      compiler.trace_args[i] = nir_variable_create(b->shader, nir_var_shader_temp,
                                                    argument_types[i], "trace_arg");
#endif
   compiler.callables = calloc(pipeline->rt.stage_count, sizeof(*compiler.callables));
   unsigned next_entry = 1;
   for (unsigned i = 0; i < pipeline->rt.stage_count; i++) {
      nir_shader *stage = pipeline->rt.stages[i]->nir;
      if (stage->info.stage != MESA_SHADER_CALLABLE
#ifdef VP_HACKRTCORE
          && stage->info.stage != MESA_SHADER_RAYGEN
#endif
          )
         continue;
      struct lvp_callable_stage *callable = &compiler.callables[i];
      callable->stage = nir_shader_clone(NULL, stage);
      /* Merge ordinary source-language returns before appending the shader
       * return; call suspension will be introduced by the common NIR pass. */
      NIR_PASS(_, callable->stage, nir_lower_returns);
      nir_builder cb = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(callable->stage)));
      nir_rt_return_amd(&cb);
      const nir_lower_shader_calls_options options = {
         .address_format = nir_address_format_32bit_offset,
         .stack_alignment = 16,
         .localized_loads = true,
      };
      callable->spill_base = callable->stage->scratch_size;
      nir_lower_shader_calls(callable->stage, &options, &callable->resumes,
                             &callable->resume_count, callable->stage);
#ifdef VP_HACKRTCORE
      if (stage->info.stage == MESA_SHADER_RAYGEN) {
         lvp_merge_raygen_call_tail(callable->stage);
         for (unsigned j = 0; j < callable->resume_count; ++j)
            lvp_merge_raygen_call_tail(callable->resumes[j]);
      }
#endif
      callable->entry = next_entry;
      next_entry += 1 + callable->resume_count;
#ifdef VP_HACKRTCORE
      if (stage->info.stage == MESA_SHADER_RAYGEN) {
         unsigned bytes = callable->stage->scratch_size - callable->spill_base;
         if (bytes)
            callable->live_registers = nir_variable_create(b->shader, nir_var_shader_temp,
               glsl_array_type(glsl_uint_type(), DIV_ROUND_UP(bytes, 4), 0), "raygen_live_registers");
         compiler.raygen_size = MAX2(compiler.raygen_size, callable->spill_base);
         continue;
      }
#endif
      pipeline->rt.stage_stack_sizes[i] = ALIGN(callable->stage->scratch_size, 16) + LVP_CALL_FRAME_HEADER_SIZE;
      compiler.callable_size = MAX2(compiler.callable_size, pipeline->rt.stage_stack_sizes[i]);
   }

   nir_def *launch_id = nir_load_ray_launch_id(b);
   nir_def *launch_size = nir_load_ray_launch_size(b);
   nir_def *oob = nir_ige(b, nir_channel(b, launch_id, 0), nir_channel(b, launch_size, 0));
   oob = nir_ior(b, oob, nir_ige(b, nir_channel(b, launch_id, 1), nir_channel(b, launch_size, 1)));
   oob = nir_ior(b, oob, nir_ige(b, nir_channel(b, launch_id, 2), nir_channel(b, launch_size, 2)));

   nir_push_if(b, oob);
   nir_jump(b, nir_jump_return);
   nir_pop_if(b, NULL);

   nir_store_var(b, compiler.state.stack_ptr, nir_load_ray_tracing_stack_base_lvp(b), 0x1);

   struct lvp_sbt_entry raygen_entry = lvp_load_sbt_entry(
      b,
      NULL,
      offsetof(VkTraceRaysIndirectCommand2KHR, raygenShaderRecordAddress),
      offsetof(struct lvp_ray_tracing_group_handle, index));
   nir_store_var(b, compiler.state.shader_record_ptr, raygen_entry.shader_record_ptr, 0x1);

#ifdef VP_HACKRTCORE
   /* Suspend raygen at trace calls so divergent lanes reach the common
    * dispatcher before traversal and before any continuation resumes. */
   nir_store_var(b, compiler.raygen_pc, nir_imm_int(b, UINT32_MAX), 1);
   nir_shader *single_raygen = NULL;
   unsigned raygen_groups = 0;
   for (uint32_t i = 0; i < pipeline->rt.group_count; ++i) {
      struct lvp_ray_tracing_group *group = pipeline->rt.groups + i;
      if (group->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;
      struct lvp_callable_stage *raygen = &compiler.callables[group->recursive_index];
      if (!raygen->stage || raygen->stage->info.stage != MESA_SHADER_RAYGEN)
         continue;
      single_raygen = raygen->stage;
      raygen_groups++;
   }
   if (raygen_groups == 1) {
      /* Raygen is the launch entry, not a continuation selected by the
       * dispatcher. Native hardware enters it directly and reduces only
       * pending continuation IDs after a suspension. */
      lvp_call_ray_tracing_stage(b, &compiler, single_raygen);
   } else {
   for (uint32_t i = 0; i < pipeline->rt.group_count; ++i) {
      struct lvp_ray_tracing_group *group = pipeline->rt.groups + i;
      if (group->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;
      struct lvp_callable_stage *raygen = &compiler.callables[group->recursive_index];
      if (!raygen->stage || raygen->stage->info.stage != MESA_SHADER_RAYGEN)
         continue;
      nir_push_if(b, nir_ieq_imm(b, raygen_entry.value, group->handle.index));
      nir_store_var(b, compiler.raygen_pc, nir_imm_int(b, raygen->entry), 1);
      nir_pop_if(b, NULL);
   }
   }
   nir_push_loop(b);
   nir_def *pc = nir_load_var(b, compiler.raygen_pc);
   nir_def *selected = nir_reduce(b, pc, .reduction_op = nir_op_umin, .cluster_size = 0);
   nir_push_if(b, nir_ieq_imm(b, selected, UINT32_MAX));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, selected, UINT32_MAX - 1));
   nir_push_if(b, nir_ieq(b, pc, selected));
   nir_def *args[11];
   for (unsigned i = 0; i < ARRAY_SIZE(args); ++i)
      args[i] = nir_load_var(b, compiler.trace_args[i]);
   nir_trace_ray(b, args[0], args[1], args[2], args[3], args[4], args[5],
                    args[6], args[7], args[8], args[9], args[10]);
   nir_store_var(b, compiler.raygen_pc, nir_load_var(b, compiler.raygen_resume), 1);
   nir_pop_if(b, NULL);
   nir_push_else(b, NULL);
   for (unsigned i = 0; i < pipeline->rt.stage_count; ++i) {
      struct lvp_callable_stage *raygen = &compiler.callables[i];
      if (!raygen->stage || raygen->stage->info.stage != MESA_SHADER_RAYGEN)
         continue;
      /* A single raygen entry was already executed before this loop. Only
       * suspension continuations can select it again; retaining the initial
       * shader here duplicates its entire CFG and live-state PHIs. */
      for (unsigned j = raygen_groups == 1 ? 1 : 0;
           j <= raygen->resume_count; ++j) {
         nir_push_if(b, nir_ieq_imm(b, selected, raygen->entry + j));
         nir_push_if(b, nir_ieq(b, pc, selected));
         lvp_call_ray_tracing_stage(b, &compiler,
                                   j ? raygen->resumes[j - 1] : raygen->stage);
         nir_pop_if(b, NULL);
         nir_pop_if(b, NULL);
      }
   }
   nir_pop_if(b, NULL);
   nir_pop_loop(b, NULL);
#else
   for (uint32_t i = 0; i < pipeline->rt.group_count; i++) {
      struct lvp_ray_tracing_group *group = pipeline->rt.groups + i;
      if (group->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;

      nir_shader *stage = pipeline->rt.stages[group->recursive_index]->nir;

      if (stage->info.stage != MESA_SHADER_RAYGEN)
         continue;

      nir_push_if(b, nir_ieq_imm(b, raygen_entry.value, group->handle.index));
      lvp_call_ray_tracing_stage(b, &compiler, stage);
      nir_pop_if(b, NULL);
   }

#endif

   nir_shader_instructions_pass(b->shader, lvp_lower_ray_tracing_instr, nir_metadata_none, &compiler);

   NIR_PASS(_, b->shader, nir_lower_returns);

#ifdef VP_HACKRTCORE
   /* Inline depth-1 TraceRay stages before vars-to-SSA. Keeping stage
    * boundaries as NIR functions until after SSA leaves the merged ray state
    * live across too many control-flow joins and forces it to scratch. A
    * depth-1 TraceRay pipeline needs no nested TraceRay continuation. Callable
    * continuations retain their explicit nir_lower_shader_calls frames. */
   if (create_info->maxPipelineRayRecursionDepth <= 1) {
      nir_foreach_function(func, b->shader) {
         if (!func->is_entrypoint)
            func->should_inline = true;
      }
      NIR_PASS(_, b->shader, nir_inline_functions);
      nir_remove_non_entrypoints(b->shader);
      NIR_PASS(_, b->shader, nir_opt_copy_prop_vars);
      lvp_shader_optimize(b->shader);
   }
#endif

   const struct nir_lower_compute_system_values_options compute_system_values = {0};
   NIR_PASS(_, b->shader, nir_lower_compute_system_values, &compute_system_values);
   NIR_PASS(_, b->shader, nir_lower_global_vars_to_local);
   NIR_PASS(_, b->shader, nir_lower_vars_to_ssa);

   lvp_shader_optimize(b->shader);

   NIR_PASS(_, b->shader, nir_lower_vars_to_explicit_types,
            nir_var_shader_temp,
            glsl_get_natural_size_align_bytes);

   NIR_PASS(_, b->shader, nir_lower_explicit_io, nir_var_shader_temp,
            nir_address_format_32bit_offset);

   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass, lvp_lower_ray_tracing_stack_base,
            nir_metadata_control_flow, NULL);

   pipeline->rt.stack_size =
      compiler.raygen_size +
      MIN2(create_info->maxPipelineRayRecursionDepth, 1) * MAX3(compiler.chit_size, compiler.miss_size, compiler.isec_size + compiler.ahit_size) +
      MAX2(0, (int)create_info->maxPipelineRayRecursionDepth - 1) * MAX2(compiler.chit_size, compiler.miss_size) + 2 * compiler.callable_size;
   b->shader->scratch_size += pipeline->rt.stack_size;

   lvp_shader_optimize(b->shader);

#ifdef VP_HACKRTCORE
   /* Keep cross-stage state in scratch before inlining stage functions. */
   if (create_info->maxPipelineRayRecursionDepth > 1 &&
       !compiler.callable_size && !pipeline->rt.dynamic_stack_size) {
      nir_foreach_function(func, b->shader) {
         if (!func->is_entrypoint)
            func->should_inline = true;
      }
      NIR_PASS(_, b->shader, nir_inline_functions);
      nir_remove_non_entrypoints(b->shader);
      NIR_PASS(_, b->shader, nir_opt_copy_prop_vars);
      lvp_shader_optimize(b->shader);
   }
#endif

#ifdef VP_HACKRTCORE
   /* Fully inlined shader-call data is private to this invocation. Recover
    * scalar variables so constant payload/attribute slots can remain in
    * registers; indirect state keeps ordinary private-memory semantics. */
   unsigned implementations = 0;
   bool scratch_private = true;
   nir_foreach_function_impl(impl, b->shader) {
      implementations++;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_scratch_base_ptr)
               scratch_private = false;
         }
      }
   }
   if (implementations == 1 && scratch_private && !pipeline->rt.dynamic_stack_size) {
      NIR_PASS(_, b->shader, nir_lower_scratch_to_var);
      lvp_shader_optimize(b->shader);
   }
#endif

   struct lvp_shader *shader = &pipeline->shaders[MESA_SHADER_RAYGEN];
   lvp_shader_init(shader, b->shader);
   shader->shader_cso = lvp_shader_compile(pipeline->device, shader, nir_shader_clone(NULL, shader->pipeline_nir->nir), false);

   util_dynarray_fini(&compiler.functions);
   for (unsigned i = 0; i < pipeline->rt.stage_count; i++)
      ralloc_free(compiler.callables[i].stage);
   free(compiler.callables);
}

static VkResult
lvp_create_ray_tracing_pipeline(VkDevice _device, const VkAllocationCallbacks *allocator,
                                const VkRayTracingPipelineCreateInfoKHR *create_info,
                                VkPipeline *out_pipeline)
{
   VK_FROM_HANDLE(lvp_device, device, _device);
   VK_FROM_HANDLE(lvp_pipeline_layout, layout, create_info->layout);

   VkResult result = VK_SUCCESS;

   struct lvp_pipeline *pipeline = vk_zalloc2(&device->vk.alloc, allocator, sizeof(struct lvp_pipeline), 8,
                                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!pipeline)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   vk_object_base_init(&device->vk, &pipeline->base,
                       VK_OBJECT_TYPE_PIPELINE);

   vk_pipeline_layout_ref(&layout->vk);

   pipeline->device = device;
   pipeline->layout = layout;
   pipeline->type = LVP_PIPELINE_RAY_TRACING;
   pipeline->flags = vk_rt_pipeline_create_flags(create_info);

   pipeline->rt.stage_count = create_info->stageCount;
   pipeline->rt.group_count = create_info->groupCount;
   if (create_info->pDynamicState) {
      for (uint32_t i = 0; i < create_info->pDynamicState->dynamicStateCount; i++) {
         if (create_info->pDynamicState->pDynamicStates[i] ==
             VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR)
            pipeline->rt.dynamic_stack_size = true;
      }
   }
   if (create_info->pLibraryInfo) {
      for (uint32_t i = 0; i < create_info->pLibraryInfo->libraryCount; i++) {
         VK_FROM_HANDLE(lvp_pipeline, library, create_info->pLibraryInfo->pLibraries[i]);
         pipeline->rt.stage_count += library->rt.stage_count;
         pipeline->rt.group_count += library->rt.group_count;
      }
   }

   pipeline->rt.stages = calloc(pipeline->rt.stage_count, sizeof(struct lvp_pipeline_nir *));
   pipeline->rt.groups = calloc(pipeline->rt.group_count, sizeof(struct lvp_ray_tracing_group));
   if (!pipeline->rt.stages || !pipeline->rt.groups) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail;
   }

   result = lvp_compile_ray_tracing_stages(pipeline, create_info);
   if (result != VK_SUCCESS)
      goto fail;

   pipeline->rt.stage_stack_sizes = calloc(pipeline->rt.stage_count, sizeof(uint32_t));
   if (!pipeline->rt.stage_stack_sizes) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail;
   }
   for (unsigned i = 0; i < pipeline->rt.stage_count; i++)
      pipeline->rt.stage_stack_sizes[i] = pipeline->rt.stages[i]->nir->scratch_size;

   lvp_init_ray_tracing_groups(pipeline, create_info);

   if (!(pipeline->flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)) {
      lvp_compile_ray_tracing_pipeline(pipeline, create_info);
   }

   *out_pipeline = lvp_pipeline_to_handle(pipeline);

   return VK_SUCCESS;

fail:
   lvp_pipeline_destroy(device, pipeline, false);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_CreateRayTracingPipelinesKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   VkPipelineCache pipelineCache,
   uint32_t createInfoCount,
   const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
   const VkAllocationCallbacks *pAllocator,
   VkPipeline *pPipelines)
{
   VkResult result = VK_SUCCESS;

   uint32_t i = 0;
   for (; i < createInfoCount; i++) {
      VkResult tmp_result = lvp_create_ray_tracing_pipeline(
         device, pAllocator, pCreateInfos + i, pPipelines + i);

      if (tmp_result != VK_SUCCESS) {
         result = tmp_result;
         pPipelines[i] = VK_NULL_HANDLE;

         if (vk_rt_pipeline_create_flags(&pCreateInfos[i]) &
             VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR)
            break;
      }
   }

   for (; i < createInfoCount; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}


VKAPI_ATTR VkResult VKAPI_CALL
lvp_GetRayTracingShaderGroupHandlesKHR(
    VkDevice _device,
    VkPipeline _pipeline,
    uint32_t firstGroup,
    uint32_t groupCount,
    size_t dataSize,
    void *pData)
{
   VK_FROM_HANDLE(lvp_pipeline, pipeline, _pipeline);

   uint8_t *data = pData;
   memset(data, 0, dataSize);

   for (uint32_t i = 0; i < groupCount; i++) {
      memcpy(data + i * LVP_RAY_TRACING_GROUP_HANDLE_SIZE,
             pipeline->rt.groups + firstGroup + i,
             sizeof(struct lvp_ray_tracing_group_handle));
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_GetRayTracingCaptureReplayShaderGroupHandlesKHR(
   VkDevice device,
   VkPipeline pipeline,
   uint32_t firstGroup,
   uint32_t groupCount,
   size_t dataSize,
   void *pData)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
lvp_GetRayTracingShaderGroupStackSizeKHR(
   VkDevice device,
   VkPipeline pipeline,
   uint32_t group,
   VkShaderGroupShaderKHR groupShader)
{
   VK_FROM_HANDLE(lvp_pipeline, rt_pipeline, pipeline);
   const struct lvp_ray_tracing_group *shader_group = rt_pipeline->rt.groups + group;
   uint32_t stage_index;
   switch (groupShader) {
   case VK_SHADER_GROUP_SHADER_GENERAL_KHR:
      stage_index = shader_group->recursive_index;
      break;
   case VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR:
      stage_index = shader_group->recursive_index;
      break;
   case VK_SHADER_GROUP_SHADER_ANY_HIT_KHR:
      stage_index = shader_group->ahit_index;
      break;
   case VK_SHADER_GROUP_SHADER_INTERSECTION_KHR:
      stage_index = shader_group->isec_index;
      break;
   default:
      unreachable("Invalid ray tracing shader group selection");
   }
   return rt_pipeline->rt.stage_stack_sizes[stage_index];
}
