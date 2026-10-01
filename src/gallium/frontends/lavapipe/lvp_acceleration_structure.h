
/*
 * Copyright © 2021 Google
 * Copyright © 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef LVP_ACCELERATION_STRUCTURE_H
#define LVP_ACCELERATION_STRUCTURE_H

#include "lvp_private.h"
#include "lvp_bvh.h"

struct lvp_accel_struct_serialization_header {
   uint8_t driver_uuid[VK_UUID_SIZE];
   uint8_t accel_struct_compat[VK_UUID_SIZE];
   uint64_t serialization_size;
   uint64_t compacted_size;
   uint64_t instance_count;
   uint64_t instances[];
};

VkResult
lvp_device_init_accel_struct_state(struct lvp_device *device);

void
lvp_device_finish_accel_struct_state(struct lvp_device *device);

#endif
