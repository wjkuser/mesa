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
   explicit Scene(vp_as_ctx *ctx) : ctx(ctx) {}
   vp_as_ctx *ctx;
   struct Blas { uint32_t address, size; };
   std::unordered_map<const void *, Blas> blases;
   BvhLayout layout = BvhLayout::from_environment();
   std::vector<uint8_t> bytes = std::vector<uint8_t>(RTU_BVH_SCENE_HDR_BYTES);
   std::vector<std::array<uint8_t, RTU_INSTANCE_METADATA_BYTES>> instances;
   Blas relocate_blas(const void *source)
   {
      auto found = blases.find(source);
      if (found != blases.end())
         return found->second;
      Scene child(ctx);
      child.bytes.clear();
      if (child.build(source) != 0)
         throw std::runtime_error("BLAS root must start its allocation");
      uint32_t size = child.bytes.size();
      auto *copy = static_cast<uint8_t *>(malloc(size));
      if (!copy) throw std::bad_alloc();
      memcpy(copy, child.bytes.data(), size);
      uint64_t address = vp_as_resident_address(ctx, source, copy, size);
      if (!address || address > UINT32_MAX)
         throw std::runtime_error("could not allocate addressable BLAS storage");
      Blas blas{uint32_t(address), size};
      blases.emplace(source, blas);
      return blas;
   }

   template<typename T> uint32_t append(const T &value)
   {
      uint32_t offset = bytes.size();
      const auto *data = reinterpret_cast<const uint8_t *>(&value);
      bytes.insert(bytes.end(), data, data + sizeof(value));
      return offset;
   }

   uint32_t build(const void *base)
   {
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
         // Vulkan defines an instance of an empty BLAS as a degenerate
         // instance bounded by its origin. Transforming the empty (+inf,-inf)
         // bounds instead produces NaNs, which are not valid SAH inputs.
         const bool empty = low[0] > high[0] || low[1] > high[1] || low[2] > high[2];
         for (unsigned a = 0; a < 3; ++a) {
            const float *row = instance.otw_matrix.values[a];
            primitive.mn[a] = primitive.mx[a] = row[3];
            if (empty)
               continue;
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
         append(src.primitive_id);
         return offset;
      }
      case lvp_bvh_node_instance: {
         const auto &src = *static_cast<const lvp_bvh_instance_node *>(source);
         VxBvhInstance instance{};
         const auto blas = relocate_blas(reinterpret_cast<const void *>(src.bvh_ptr));
         instance.blas_root_byte_offset = blas.address;
         memcpy(instance.xform, src.otw_matrix.values, sizeof(instance.xform));
         instance.custom_id = src.custom_instance_and_mask & 0xffffffu;
         instance.instance_id = blas.size;
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
         memcpy(metadata.data() + RTU_INSTANCE_METADATA_CUSTOM_OFFSET, &instance.custom_id, 4);
         bytes.resize((bytes.size() + 15) & ~size_t(15), 0);
         const char *record_bytes = std::getenv("HACKRTCORE_RTU_INSTANCE_RECORD_BYTES");
         if (record_bytes && !std::strcmp(record_bytes, "64") &&
             layout.width <= kVxBvhCompactInstanceSlots && (blas.address & 15u) == 0) {
            VxBvhCompactInstance record{};
            record.blas_address_and_flags = blas.address | native_flags;
            record.blas_bytes = blas.size;
            record.instance_id = src.instance_id;
            record.sbt_offset_and_mask = sbt_offset | (src.custom_instance_and_mask & 0xff000000u);
            memcpy(record.world_to_object, src.wto_matrix.values, sizeof(record.world_to_object));
            uint32_t offset = append(VxBvhLeafHeader{
               RTU_BVH_KIND_LEAF_INST | (1u << RTU_BVH_COUNT_SHIFT), 0,
               RTU_BVH_INSTANCE_EXTERNAL_BLAS | RTU_BVH_INSTANCE_COMPACT_RECORD, src.instance_id});
            append(record);
            return offset;
         }
         uint32_t offset = append(VxBvhLeafHeader{
            RTU_BVH_KIND_LEAF_INST | (1u << RTU_BVH_COUNT_SHIFT), 0, RTU_BVH_INSTANCE_EXTERNAL_BLAS, src.instance_id});
         append(instance);
         return offset;
      }
      }
      std::abort();
   }


};
}

extern "C" uint8_t *
vp_build_scene(vp_as_ctx *ctx, const void *acceleration_structure, uint32_t *size)
try {
   Scene scene(ctx);
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
} catch (const std::exception &) {
   return nullptr;
}
