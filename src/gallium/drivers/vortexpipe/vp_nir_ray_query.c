/* SPDX-License-Identifier: MIT */
#include "vp_nir_to_llvm.h"
#include "nir_builder.h"
#include "util/hash_table.h"
#include "registers.h"
#include "scene_layout.h"

/* Per-query shader state. Arrays retain the application's array dimensions. */
enum query_word {
   Q_SCENE, Q_FLAGS, Q_MASK, Q_ORIGIN, Q_DIRECTION = Q_ORIGIN + 3,
   Q_MIN = Q_DIRECTION + 3, Q_MAX, Q_HANDLE, Q_PHASE, Q_ACTION, Q_GENERATED_T,
   Q_CANDIDATE = 32, Q_COMMITTED = Q_CANDIDATE + VX_RT_SLOT_COUNT,
   Q_WORDS = Q_COMMITTED + VX_RT_SLOT_COUNT,
};
enum query_phase { Q_INITIALIZED, Q_HAS_CANDIDATE, Q_COMPLETE };

static const struct glsl_type *
state_type(const struct glsl_type *type, unsigned words)
{
   if (glsl_type_is_array(type))
      return glsl_array_type(state_type(glsl_get_array_element(type), words),
                             glsl_get_length(type), 0);
   return glsl_array_type(glsl_uint_type(), words, 0);
}

static nir_deref_instr *
state_deref(nir_builder *b, nir_deref_instr *query, nir_variable *state)
{
   if (query->deref_type == nir_deref_type_var)
      return nir_build_deref_var(b, state);
   assert(query->deref_type == nir_deref_type_array);
   return nir_build_deref_array(b,
      state_deref(b, nir_deref_instr_parent(query), state), query->arr.index.ssa);
}

static nir_def *
load(nir_builder *b, nir_deref_instr *state, unsigned word)
{
   return nir_load_deref(b, nir_build_deref_array_imm(b, state, word));
}

static void
store(nir_builder *b, nir_deref_instr *state, unsigned word, nir_def *value)
{
   for (unsigned c = 0; c < value->num_components; ++c)
      nir_store_deref(b, nir_build_deref_array_imm(b, state, word + c),
                      nir_channel(b, value, c), 1);
}

static nir_def *
load_vector(nir_builder *b, nir_deref_instr *state, unsigned word, unsigned count)
{
   nir_def *v[4];
   for (unsigned c = 0; c < count; ++c)
      v[c] = load(b, state, word + c);
   return nir_vec(b, v, count);
}

static const unsigned hit_slots[] = {
   VX_RT_OBJECT_RAY_ORIGIN, VX_RT_OBJECT_RAY_ORIGIN + 1, VX_RT_OBJECT_RAY_ORIGIN + 2,
   VX_RT_OBJECT_RAY_DIRECTION, VX_RT_OBJECT_RAY_DIRECTION + 1, VX_RT_OBJECT_RAY_DIRECTION + 2,
   VX_RT_HIT_T, VX_RT_HIT_BARY_U, VX_RT_HIT_BARY_V, VX_RT_HIT_ATTR_0,
   VX_RT_HIT_PRIMITIVE_ID, VX_RT_HIT_INSTANCE_ID, VX_RT_HIT_GEOMETRY_INDEX,
   VX_RT_HIT_INSTANCE_CUSTOM, VX_RT_HIT_FLAGS,
};

static void
snapshot(nir_builder *b, nir_deref_instr *state, unsigned bank,
         nir_def *token, bool committed_registers)
{
   for (unsigned i = 0; i < ARRAY_SIZE(hit_slots); ++i) {
      unsigned slot = hit_slots[i];
      nir_def *value = committed_registers
         ? nir_vortex_rt_get_committed(b, 32, token, .base = slot)
         : nir_vortex_rt_get(b, 32, token, .base = slot);
      store(b, state, bank + slot, value);
   }
}

static void
commit_candidate(nir_builder *b, nir_deref_instr *state)
{
   for (unsigned i = 0; i < ARRAY_SIZE(hit_slots); ++i)
      store(b, state, Q_COMMITTED + hit_slots[i],
            load(b, state, Q_CANDIDATE + hit_slots[i]));
}

static void
initialize(nir_builder *b, nir_intrinsic_instr *in, nir_deref_instr *state)
{
   store(b, state, Q_SCENE, nir_u2u32(b, in->src[1].ssa));
   store(b, state, Q_FLAGS, in->src[2].ssa);
   store(b, state, Q_MASK, in->src[3].ssa);
   store(b, state, Q_ORIGIN, in->src[4].ssa);
   store(b, state, Q_MIN, in->src[5].ssa);
   store(b, state, Q_DIRECTION, in->src[6].ssa);
   store(b, state, Q_MAX, in->src[7].ssa);
   store(b, state, Q_PHASE, nir_imm_int(b, Q_INITIALIZED));
   store(b, state, Q_ACTION, nir_imm_int(b, VX_RT_ACTION_START));
   store(b, state, Q_GENERATED_T, nir_imm_int(b, 0));
   store(b, state, Q_COMMITTED + VX_RT_HIT_FLAGS, nir_imm_int(b, 0));
   store(b, state, Q_COMMITTED + VX_RT_HIT_T, in->src[7].ssa);
}

static nir_def *
proceed(nir_builder *b, nir_deref_instr *state, nir_def *continuation)
{
   nir_push_if(b, nir_ine_imm(b, load(b, state, Q_PHASE), Q_COMPLETE));
   nir_vortex_rt_set(b, continuation, .base = VX_RT_QUERY_CONTEXT);
   nir_vortex_rt_set(b, load(b, state, Q_ACTION), .base = VX_RT_QUERY_ACTION);
   nir_vortex_rt_set(b, load(b, state, Q_GENERATED_T), .base = VX_RT_QUERY_DISTANCE);
   nir_def *flags = nir_ior(b, load(b, state, Q_FLAGS),
       nir_ishl_imm(b, nir_iand_imm(b, load(b, state, Q_MASK), 255), 16));
   store(b, state, Q_HANDLE, nir_vortex_rt_wtrace(b, 32,
       load(b, state, Q_SCENE), flags, load_vector(b, state, Q_ORIGIN, 3),
       load_vector(b, state, Q_DIRECTION, 3),
       load(b, state, Q_MIN), load(b, state, Q_MAX)));

   /* A pending lane stays inside proceed, before the application's loop body. */
   nir_variable *event = nir_local_variable_create(b->impl, glsl_uint_type(), "rq_event");
   nir_push_loop(b);
   nir_def *status = nir_vortex_rt_event_wait(b, 32, load(b, state, Q_HANDLE));
   nir_store_var(b, event, status, 1);
   nir_push_if(b, nir_ine_imm(b, status, VX_RT_EVENT_PENDING));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_pop_loop(b, NULL);
   nir_def *token = nir_load_var(b, event);
   nir_push_if(b, nir_ieq_imm(b, token, VX_RT_EVENT_COMPLETE));
   nir_def *terminal = nir_vortex_rt_wait(b, 32, load(b, state, Q_HANDLE));
   snapshot(b, state, Q_COMMITTED, terminal, false);
   store(b, state, Q_PHASE, nir_imm_int(b, Q_COMPLETE));
   nir_push_else(b, NULL);
   snapshot(b, state, Q_CANDIDATE, token, false);
   snapshot(b, state, Q_COMMITTED, token, true);
   nir_vortex_rt_wait(b, 32, load(b, state, Q_HANDLE));
   store(b, state, Q_PHASE, nir_imm_int(b, Q_HAS_CANDIDATE));
   store(b, state, Q_ACTION, nir_imm_int(b, VX_RT_ACTION_IGNORE));
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   return nir_ieq_imm(b, load(b, state, Q_PHASE), Q_HAS_CANDIDATE);
}

static void
terminate(nir_builder *b, nir_deref_instr *state)
{
   store(b, state, Q_PHASE, nir_imm_int(b, Q_COMPLETE));
}

static nir_def *
instance_record(nir_builder *b, nir_deref_instr *state, unsigned bank)
{
   nir_def *scene = load(b, state, Q_SCENE);
   nir_def *table = nir_load_global(b,
      nir_u2u64(b, nir_iadd_imm(b, scene, RTU_SCENE_INSTANCE_TABLE_OFFSET)), 4, 1, 32);
   return nir_iadd(b, nir_iadd(b, scene, table), nir_imul_imm(b,
      load(b, state, bank + VX_RT_HIT_INSTANCE_ID), RTU_INSTANCE_METADATA_BYTES));
}

static nir_def *
query_value(nir_builder *b, nir_intrinsic_instr *in, nir_deref_instr *state)
{
   bool committed = nir_intrinsic_committed(in);
   unsigned bank = committed ? Q_COMMITTED : Q_CANDIDATE;
   nir_def *hit_flags = load(b, state, bank + VX_RT_HIT_FLAGS);
   switch (nir_intrinsic_ray_query_value(in)) {
   case nir_ray_query_value_intersection_type: {
      nir_def *procedural = nir_ine_imm(b, nir_iand_imm(b, hit_flags, VX_RT_HIT_PROCEDURAL), 0);
      if (!committed) return nir_b2i32(b, procedural);
      return nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, hit_flags, VX_RT_HIT_VALID), 0),
                      nir_bcsel(b, procedural, nir_imm_int(b, 2), nir_imm_int(b, 1)),
                      nir_imm_int(b, 0));
   }
   case nir_ray_query_value_intersection_t: return load(b, state, bank + VX_RT_HIT_T);
   case nir_ray_query_value_intersection_barycentrics:
      return load_vector(b, state, bank + VX_RT_HIT_BARY_U, 2);
   case nir_ray_query_value_intersection_primitive_index:
      return load(b, state, bank + VX_RT_HIT_PRIMITIVE_ID);
   case nir_ray_query_value_intersection_geometry_index:
      return load(b, state, bank + VX_RT_HIT_GEOMETRY_INDEX);
   case nir_ray_query_value_intersection_instance_id:
      return load(b, state, bank + VX_RT_HIT_INSTANCE_ID);
   case nir_ray_query_value_intersection_instance_custom_index:
      return load(b, state, bank + VX_RT_HIT_INSTANCE_CUSTOM);
   case nir_ray_query_value_intersection_front_face:
      return nir_ieq_imm(b, load(b, state, bank + VX_RT_HIT_ATTR_0), 0xfe);
   case nir_ray_query_value_intersection_candidate_aabb_opaque:
      return nir_ine_imm(b, nir_iand_imm(b, hit_flags, VX_RT_HIT_OPAQUE), 0);
   case nir_ray_query_value_intersection_object_ray_origin:
      return load_vector(b, state, bank + VX_RT_OBJECT_RAY_ORIGIN, 3);
   case nir_ray_query_value_intersection_object_ray_direction:
      return load_vector(b, state, bank + VX_RT_OBJECT_RAY_DIRECTION, 3);
   case nir_ray_query_value_world_ray_origin: return load_vector(b, state, Q_ORIGIN, 3);
   case nir_ray_query_value_world_ray_direction: return load_vector(b, state, Q_DIRECTION, 3);
   case nir_ray_query_value_flags: return load(b, state, Q_FLAGS);
   case nir_ray_query_value_tmin: return load(b, state, Q_MIN);
   case nir_ray_query_value_intersection_instance_sbt_index:
      return nir_load_global(b, nir_u2u64(b, nir_iadd_imm(b,
         instance_record(b, state, bank), RTU_INSTANCE_METADATA_SBT_OFFSET)), 4, 1, 32);
   case nir_ray_query_value_intersection_object_to_world:
   case nir_ray_query_value_intersection_world_to_object: {
      unsigned offset = nir_intrinsic_column(in) * 4;
      if (nir_intrinsic_ray_query_value(in) == nir_ray_query_value_intersection_world_to_object)
         offset += 48;
      nir_def *base = instance_record(b, state, bank), *rows[3];
      for (unsigned r = 0; r < 3; ++r)
         rows[r] = nir_load_global(b, nir_u2u64(b,
            nir_iadd_imm(b, base, offset + r * 16)), 4, 1, 32);
      return nir_vec3(b, rows[0], rows[1], rows[2]);
   }
   default: unreachable("unsupported ray query value");
   }
}

static unsigned
continuation_size(const struct glsl_type *type)
{
   return glsl_type_is_array(type)
      ? glsl_get_length(type) * continuation_size(glsl_get_array_element(type))
      : VX_RT_QUERY_CONTEXT_BYTES + VX_RT_QUERY_SPILL_BYTES;
}

static nir_def *
continuation_address(nir_builder *b, nir_deref_instr *query, unsigned base)
{
   if (query->deref_type == nir_deref_type_var)
      return nir_iadd_imm(b, nir_load_scratch_base_ptr(b, 1, 32), base);
   return nir_iadd(b, continuation_address(b, nir_deref_instr_parent(query), base),
      nir_imul_imm(b, query->arr.index.ssa, continuation_size(query->type)));
}

bool
vp_nir_lower_ray_tracing_to_rtu(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder b = nir_builder_create(impl);
   struct hash_table *states = _mesa_pointer_hash_table_create(NULL);
   struct hash_table *continuations = _mesa_pointer_hash_table_create(NULL);
   unsigned query_array_length = 1;
   bool progress = false;
   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         switch (in->intrinsic) {
         case nir_intrinsic_rq_initialize: case nir_intrinsic_rq_proceed:
         case nir_intrinsic_rq_terminate: case nir_intrinsic_rq_confirm_intersection:
         case nir_intrinsic_rq_generate_intersection: case nir_intrinsic_rq_load: break;
         default: continue;
         }
         b.cursor = nir_before_instr(instr);
         nir_deref_instr *query = nir_src_as_deref(in->src[0]);
         nir_variable *var = nir_deref_instr_get_variable(query);
         struct hash_entry *entry = _mesa_hash_table_search(states, var);
         if (!entry) {
            nir_variable *state = nir_local_variable_create(impl, state_type(var->type, Q_WORDS), var->name);
            entry = _mesa_hash_table_insert(states, var, state);
            unsigned offset = ALIGN(shader->scratch_size, 64);
            _mesa_hash_table_insert(continuations, var, (void *)(uintptr_t)offset);
            shader->scratch_size = offset + continuation_size(var->type);
            query_array_length = MAX2(query_array_length, continuation_size(var->type) /
               (VX_RT_QUERY_CONTEXT_BYTES + VX_RT_QUERY_SPILL_BYTES));
         }
         nir_deref_instr *state = state_deref(&b, query, entry->data);
         nir_def *result = NULL;
         switch (in->intrinsic) {
         case nir_intrinsic_rq_initialize: initialize(&b, in, state); break;
         case nir_intrinsic_rq_proceed:
            result = proceed(&b, state, continuation_address(&b, query,
               (uintptr_t)_mesa_hash_table_search(continuations, var)->data));
            break;
         case nir_intrinsic_rq_terminate: terminate(&b, state); break;
         case nir_intrinsic_rq_confirm_intersection:
            commit_candidate(&b, state);
            store(&b, state, Q_ACTION, nir_imm_int(&b, VX_RT_ACTION_ACCEPT));
            break;
         case nir_intrinsic_rq_generate_intersection: {
            nir_def *t = in->src[1].ssa;
            nir_push_if(&b, nir_iand(&b, nir_fge(&b, t, load(&b, state, Q_MIN)),
               nir_fge(&b, load(&b, state, Q_COMMITTED + VX_RT_HIT_T), t)));
            commit_candidate(&b, state);
            store(&b, state, Q_COMMITTED + VX_RT_HIT_T, t);
            store(&b, state, Q_GENERATED_T, t);
            store(&b, state, Q_ACTION, nir_imm_int(&b, VX_RT_ACTION_GENERATE));
            nir_pop_if(&b, NULL);
            break;
         }
         case nir_intrinsic_rq_load: result = query_value(&b, in, state); break;
         default: unreachable("ray query operation");
         }
         if (result) nir_def_rewrite_uses(&in->def, result);
         nir_instr_remove(instr);
         progress = true;
      }
   }
   _mesa_hash_table_destroy(states, NULL);
   _mesa_hash_table_destroy(continuations, NULL);
   if (progress) {
      nir_progress(true, impl, nir_metadata_none);
      nir_lower_indirect_derefs(shader, nir_var_function_temp, query_array_length);
      nir_lower_vars_to_ssa(shader);
      nir_opt_dce(shader);
   }
   return progress;
}
