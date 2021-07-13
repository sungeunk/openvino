/*
// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "non_max_suppression_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "non_max_suppression/non_max_suppression_kernel_selector.h"
#include "non_max_suppression/non_max_suppression_kernel_ref.h"

namespace cldnn {
namespace gpu {

struct non_max_suppression_gpu : typed_primitive_gpu_impl<non_max_suppression> {
    using parent = typed_primitive_gpu_impl<non_max_suppression>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<non_max_suppression_gpu>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<non_max_suppression>& instance,
                                                        int32_t) const override {
        kernel_arguments_data args;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_num_select_per_class())
            args.inputs.push_back(instance.num_select_per_class_mem());

        if (instance.has_iou_threshold())
            args.inputs.push_back(instance.iou_threshold_mem());

        if (instance.has_score_threshold())
            args.inputs.push_back(instance.score_threshold_mem());

        if (instance.has_soft_nms_sigma())
            args.inputs.push_back(instance.soft_nms_sigma_mem());

        args.output = instance.output_memory_ptr();
        if (instance.has_second_output())
            args.inputs.push_back(instance.second_output_mem());
        if (instance.has_third_output())
            args.inputs.push_back(instance.third_output_mem());

        return args;
    }

public:
    static primitive_impl* create(const non_max_suppression_node& arg) {
        auto params = get_default_params<kernel_selector::non_max_suppression_params>(arg);
        auto optional_params =
            get_default_optional_params<kernel_selector::non_max_suppression_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        params.inputs.push_back(convert_data_tensor(arg.input_scores().get_output_layout()));

        if (arg.has_num_select_per_class()) {
            params.inputs.push_back(convert_data_tensor(arg.num_select_per_class_node().get_output_layout()));
            params.has_num_select_per_class = true;
        }

        if (arg.has_iou_threshold()) {
            params.inputs.push_back(convert_data_tensor(arg.iou_threshold_node().get_output_layout()));
            params.has_iou_threshold = true;
        }

        if (arg.has_score_threshold()) {
            params.inputs.push_back(convert_data_tensor(arg.score_threshold_node().get_output_layout()));
            params.has_score_threshold = true;
        }

        if (arg.has_soft_nms_sigma()) {
            params.inputs.push_back(convert_data_tensor(arg.soft_nms_sigma_node().get_output_layout()));
            params.has_soft_nms_sigma = true;
        }

        if (arg.has_second_output()) {
            params.inputs.push_back(convert_data_tensor(arg.second_output_node().get_output_layout()));
            params.has_second_output = true;
        }

        if (arg.has_second_output()) {
            params.inputs.push_back(convert_data_tensor(arg.third_output_node().get_output_layout()));
            params.has_third_output = true;
        }

        params.sort_result_descending = primitive->sort_result_descending;
        params.box_encoding = primitive->center_point_box ? 1 : 0;

        auto& kernel_selector = kernel_selector::non_max_suppression_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto non_max_suppression_node = new non_max_suppression_gpu(arg, best_kernels[0]);

        return non_max_suppression_node;
    }
};

namespace detail {

primitive_impl* create_nms_gpu(const non_max_suppression_node& node) {
    return non_max_suppression_gpu::create(node);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
