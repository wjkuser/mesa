/* SPDX-License-Identifier: MIT */
#include "vp_scene.h"
#include "../../frontends/lavapipe/lvp_bvh.h"
#include "bvh.h"
#include "scene_layout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {
using namespace vortex::rtu;

class Scene {
public:
   std::vector<uint8_t> bytes = std::vector<uint8_t>(RTU_BVH_SCENE_HDR_BYTES);
   std::vector<std::array<uint8_t, RTU_INSTANCE_METADATA_BYTES>> instances;
   std::unordered_map<const void *, uint32_t> roots;

   template<typename T> uint32_t append(const T &value)
   {
      uint32_t offset = bytes.size();
      const auto *data = reinterpret_cast<const uint8_t *>(&value);
      bytes.insert(bytes.end(), data, data + sizeof(value));
      return offset;
   }

   uint32_t build(const void *base)
   {
      auto found = roots.find(base);
      if (found != roots.end())
         return found->second;
      uint32_t root = node(static_cast<const uint8_t *>(base), LVP_BVH_ROOT_NODE);
      roots.emplace(base, root);
      return root;
   }

   uint32_t node(const uint8_t *base, uint32_t id)
   {
      const void *source = base + (id & ~7u);
      switch (id & 7u) {
      case lvp_bvh_node_internal:
         return box(base, *static_cast<const lvp_bvh_box_node *>(source));
      case lvp_bvh_node_triangle: {
         const auto &src = *static_cast<const lvp_bvh_triangle_node *>(source);
         uint32_t opaque = (src.geometry_id_and_flags & LVP_GEOMETRY_OPAQUE) ? 1 : 0;
         uint32_t offset = append(VxBvhLeafHeader{
            RTU_BVH_KIND_LEAF_TRI | (1u << RTU_BVH_COUNT_SHIFT),
            src.geometry_id_and_flags & 0x0fffffffu, opaque, src.primitive_id});
         VxBvhTri triangle{};
         memcpy(triangle.v0, src.coords[0], sizeof(triangle.v0));
         memcpy(triangle.v1, src.coords[1], sizeof(triangle.v1));
         memcpy(triangle.v2, src.coords[2], sizeof(triangle.v2));
         triangle.flags = opaque;
         append(triangle);
         return offset;
      }
      case lvp_bvh_node_aabb: {
         const auto &src = *static_cast<const lvp_bvh_aabb_node *>(source);
         uint32_t offset = append(VxBvhLeafHeader{
            RTU_BVH_KIND_LEAF_PROC | (1u << RTU_BVH_COUNT_SHIFT),
            src.geometry_id_and_flags & 0x0fffffffu,
            (src.geometry_id_and_flags & LVP_GEOMETRY_OPAQUE) ? 1u : 0u,
            src.primitive_id});
         VxBvhProcAabb bounds{};
         memcpy(bounds.aabb_min, &src.bounds.min, sizeof(bounds.aabb_min));
         memcpy(bounds.aabb_max, &src.bounds.max, sizeof(bounds.aabb_max));
         append(bounds);
         return offset;
      }
      case lvp_bvh_node_instance: {
         const auto &src = *static_cast<const lvp_bvh_instance_node *>(source);
         VxBvhInstance instance{};
         instance.blas_root_byte_offset = build(reinterpret_cast<const void *>(src.bvh_ptr));
         memcpy(instance.xform, src.otw_matrix.values, sizeof(instance.xform));
         instance.custom_id = src.custom_instance_and_mask & 0xffffffu;
         instance.instance_id = src.instance_id;
         uint32_t flags = src.sbt_offset_and_flags;
         uint32_t native_flags =
            ((flags & LVP_INSTANCE_TRIANGLE_FACING_CULL_DISABLE) ? RTU_INST_FLAG_TRI_CULL_DIS : 0) |
            ((flags & LVP_INSTANCE_TRIANGLE_FLIP_FACING) ? RTU_INST_FLAG_TRI_FLIP : 0) |
            ((flags & LVP_INSTANCE_FORCE_OPAQUE) ? RTU_INST_FLAG_FORCE_OPAQUE : 0) |
            ((flags & LVP_INSTANCE_NO_FORCE_NOT_OPAQUE) ? 0 : RTU_INST_FLAG_FORCE_NO_OPQ);
         instance.cull_mask = (src.custom_instance_and_mask >> 24) |
                              (native_flags << RTU_INST_FLAGS_SHIFT);
         if (instances.size() <= src.instance_id)
            instances.resize(src.instance_id + 1);
         auto &metadata = instances[src.instance_id];
         memcpy(metadata.data(), src.otw_matrix.values, 48);
         memcpy(metadata.data() + 48, src.wto_matrix.values, 48);
         uint32_t sbt_offset = flags & 0xffffffu;
         memcpy(metadata.data() + RTU_INSTANCE_METADATA_SBT_OFFSET, &sbt_offset, 4);
         uint32_t offset = append(VxBvhLeafHeader{
            RTU_BVH_KIND_LEAF_INST | (1u << RTU_BVH_COUNT_SHIFT), 0, 0, 0});
         append(instance);
         return offset;
      }
      }
      std::abort();
   }

   uint32_t box(const uint8_t *base, const lvp_bvh_box_node &src)
   {
      VxBvhInternalNode out{};
      vk_aabb bounds[2];
      unsigned count = 0;
      for (unsigned i = 0; i < 2; ++i) {
         if (src.children[i] == LVP_BVH_INVALID_NODE)
            continue;
         uint32_t child = node(base, src.children[i]);
         bool leaf = (src.children[i] & 7u) != lvp_bvh_node_internal;
         out.child_offsets[count] = child | (leaf ? RTU_BVH_CHILD_LEAF_FLAG : 0);
         bounds[count++] = src.bounds[i];
      }
      if (!count)
         return append(VxBvhLeafHeader{RTU_BVH_KIND_LEAF_TRI, 0, 0, 0});
      out.kind = count << RTU_BVH_COUNT_SHIFT;
      for (unsigned axis = 0; axis < 3; ++axis) {
         float low[2], high[2];
         for (unsigned i = 0; i < count; ++i) {
            memcpy(&low[i], reinterpret_cast<const uint8_t *>(&bounds[i].min) + axis*4, 4);
            memcpy(&high[i], reinterpret_cast<const uint8_t *>(&bounds[i].max) + axis*4, 4);
         }
         float origin = low[0], upper = high[0];
         for (unsigned i = 1; i < count; ++i) {
            origin = std::min(origin, low[i]);
            upper = std::max(upper, high[i]);
         }
         int exponent = 0;
         if (upper > origin)
            std::frexp((upper - origin) / 255.0f, &exponent);
         exponent = std::clamp(exponent, -127, 127);
         float step = std::ldexp(1.0f, exponent);
         out.origin[axis] = origin;
         out.exp[axis] = exponent;
         for (unsigned i = 0; i < count; ++i) {
            out.qaabb_min[i][axis] = std::clamp(int(std::floor((low[i] - origin)/step)), 0, 255);
            out.qaabb_max[i][axis] = std::clamp(int(std::ceil((high[i] - origin)/step)), 0, 255);
         }
      }
      return append(out);
   }
};
}

extern "C" uint8_t *
vp_build_scene(const void *acceleration_structure, uint32_t *size)
{
   Scene scene;
   uint32_t root = scene.build(acceleration_structure);
   uint32_t table = scene.bytes.size();
   for (const auto &instance : scene.instances)
      scene.append(instance);
   uint32_t header[] = {root, RTU_SCENE_KIND_BVH4, uint32_t(scene.bytes.size()), table};
   memcpy(scene.bytes.data(), header, sizeof(header));
   *size = scene.bytes.size();
   auto *result = static_cast<uint8_t *>(malloc(*size));
   if (result)
      memcpy(result, scene.bytes.data(), *size);
   return result;
}
