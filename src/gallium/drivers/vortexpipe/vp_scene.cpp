/* SPDX-License-Identifier: MIT */
#include "vp_scene.h"
#include "../../frontends/lavapipe/lvp_bvh.h"
#include <hackrtcore/bvh.h>
#include <hackrtcore/scene_layout.h>
#include <hackrtcore/scene_builder.h>

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
   BvhLayout layout = BvhLayout::from_environment();
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
      std::vector<vortex::raytrace::BvhPrimitive> primitives;
      collect(static_cast<const uint8_t *>(base), LVP_BVH_ROOT_NODE, primitives);
      std::stable_sort(primitives.begin(), primitives.end(), [](const auto &a, const auto &b) {
         VxBvhLeafHeader ah, bh;
         memcpy(&ah, a.payload.data(), sizeof(ah));
         memcpy(&bh, b.payload.data(), sizeof(bh));
         return std::pair(ah.geometry_index, ah.prim_base) < std::pair(bh.geometry_index, bh.prim_base);
      });
      uint32_t root = vortex::raytrace::append_bvh(primitives, layout, bytes);
      if (root == UINT32_MAX)
         throw std::runtime_error("RTU BVH exceeds leaf address range");
      roots.emplace(base, root);
      return root;
   }

   void collect(const uint8_t *base, uint32_t id,
                std::vector<vortex::raytrace::BvhPrimitive> &primitives)
   {
      if (id == LVP_BVH_INVALID_NODE)
         return;
      if ((id & 7u) == lvp_bvh_node_internal) {
         const auto &box = *reinterpret_cast<const lvp_bvh_box_node *>(base + (id & ~7u));
         collect(base, box.children[0], primitives);
         collect(base, box.children[1], primitives);
         return;
      }
      const uint32_t offset = node(base, id);
      vortex::raytrace::BvhPrimitive primitive{};
      primitive.payload.assign(bytes.begin() + offset, bytes.end());
      bytes.resize(offset);
      const void *source = base + (id & ~7u);
      if ((id & 7u) == lvp_bvh_node_triangle) {
         const auto &tri = *static_cast<const lvp_bvh_triangle_node *>(source);
         for (unsigned a = 0; a < 3; ++a) {
            primitive.mn[a] = std::min({tri.coords[0][a], tri.coords[1][a], tri.coords[2][a]});
            primitive.mx[a] = std::max({tri.coords[0][a], tri.coords[1][a], tri.coords[2][a]});
         }
      } else if ((id & 7u) == lvp_bvh_node_aabb) {
         const auto &box = static_cast<const lvp_bvh_aabb_node *>(source)->bounds;
         memcpy(primitive.mn, &box.min, sizeof(primitive.mn));
         memcpy(primitive.mx, &box.max, sizeof(primitive.mx));
      } else {
         const auto &instance = *static_cast<const lvp_bvh_instance_node *>(source);
         const auto &box = reinterpret_cast<const lvp_bvh_header *>(instance.bvh_ptr)->bounds;
         const float *low = reinterpret_cast<const float *>(&box.min);
         const float *high = reinterpret_cast<const float *>(&box.max);
         for (unsigned a = 0; a < 3; ++a) {
            const float *row = instance.otw_matrix.values[a];
            primitive.mn[a] = primitive.mx[a] = row[3];
            for (unsigned b = 0; b < 3; ++b) {
               primitive.mn[a] += std::min(row[b] * low[b], row[b] * high[b]);
               primitive.mx[a] += std::max(row[b] * low[b], row[b] * high[b]);
            }
         }
      }
      primitives.push_back(std::move(primitive));
   }

   uint32_t node(const uint8_t *base, uint32_t id)
   {
      const void *source = base + (id & ~7u);
      switch (id & 7u) {
      case lvp_bvh_node_triangle: {
         const auto &src = *static_cast<const lvp_bvh_triangle_node *>(source);
         uint32_t opaque = (src.geometry_id_and_flags & LVP_GEOMETRY_OPAQUE) ? 1 : 0;
         bytes.resize((bytes.size() + 15) & ~size_t(15), 0);
         uint32_t offset = append(VxBvhLeafHeader{
            RTU_BVH_KIND_LEAF_TRI | (1u << RTU_BVH_COUNT_SHIFT),
            src.geometry_id_and_flags & 0x0fffffffu, opaque, src.primitive_id});
         VxBvhTri triangle{};
         memcpy(triangle.v0, src.coords[0], sizeof(triangle.v0));
         memcpy(triangle.v1, src.coords[1], sizeof(triangle.v1));
         memcpy(triangle.v2, src.coords[2], sizeof(triangle.v2));
         triangle.flags = opaque;
         triangle.primitive_id = src.primitive_id;
         append(triangle);
         return offset;
      }
      case lvp_bvh_node_aabb: {
         const auto &src = *static_cast<const lvp_bvh_aabb_node *>(source);
         bytes.resize((bytes.size() + 15) & ~size_t(15), 0);
         uint32_t offset = append(VxBvhLeafHeader{
            RTU_BVH_KIND_LEAF_PROC | (1u << RTU_BVH_COUNT_SHIFT),
            src.geometry_id_and_flags & 0x0fffffffu,
            (src.geometry_id_and_flags & LVP_GEOMETRY_OPAQUE) ? 1u : 0u,
            src.primitive_id});
         VxBvhProcAabb bounds{};
         memcpy(bounds.aabb_min, &src.bounds.min, sizeof(bounds.aabb_min));
         memcpy(bounds.aabb_max, &src.bounds.max, sizeof(bounds.aabb_max));
         bounds.primitive_id = src.primitive_id;
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
         bytes.resize((bytes.size() + 15) & ~size_t(15), 0);
         uint32_t offset = append(VxBvhLeafHeader{
            RTU_BVH_KIND_LEAF_INST | (1u << RTU_BVH_COUNT_SHIFT), 0, 0, src.instance_id});
         append(instance);
         return offset;
      }
      }
      std::abort();
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
   uint32_t header[] = {root, RTU_SCENE_KIND_BVH_WIDE, uint32_t(scene.bytes.size()), table};
   memcpy(scene.bytes.data(), header, sizeof(header));
   *size = scene.bytes.size();
   auto *result = static_cast<uint8_t *>(malloc(*size));
   if (result)
      memcpy(result, scene.bytes.data(), *size);
   return result;
}
