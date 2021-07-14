// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_max_suppression_inst.h"
#include "non_max_suppression/non_max_suppression_kernel_ref.h"
#include "network_impl.h"
#include "register_gpu.hpp"

namespace cldnn {
namespace gpu {
namespace detail {

extern primitive_impl* create_nms_cpu(const non_max_suppression_node& node);
extern primitive_impl* create_nms_gpu(const non_max_suppression_node& node);

static primitive_impl* create_nms(const non_max_suppression_node& node) {
    // auto params = get_default_params<kernel_selector::non_max_suppression_params>(node);
    // auto scoresTensor = convert_data_tensor(node.input_scores().get_output_layout());
    // const size_t kBatchNum = scoresTensor.Batch().v;
    // const size_t kClassNum = scoresTensor.Feature().v;
    // const size_t kNStreams = static_cast<size_t>(node.get_program().get_engine().configuration().n_streams);
    // const size_t kKeyValue = kBatchNum * std::min(kClassNum, static_cast<size_t>(8)) * kNStreams;

    // if (kKeyValue > 64) {
    //     return create_nms_gpu(node);
    // } else {
    //     return create_nms_cpu(node);
    // }
    return create_nms_gpu(node);
}

attach_non_max_suppression_gpu::attach_non_max_suppression_gpu() {
    implementation_map<non_max_suppression>::add({
        {std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), create_nms},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), create_nms},
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), create_nms}
    });
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
