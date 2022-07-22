// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "convert_color/convert_color_kernel_selector.h"
#include "convert_color/convert_color_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "data_inst.h"
#include <vector>

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct convert_color_impl : typed_primitive_impl_ocl<convert_color> {
    using parent = typed_primitive_impl_ocl<convert_color>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<convert_color_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<convert_color>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        return args;
    }

public:
    static primitive_impl* create(const convert_color_node& arg, const kernel_impl_params& impl_param) {
        auto primitive = arg.get_primitive();

        auto convert_color_params = get_default_params<kernel_selector::convert_color_params>(impl_param);
        auto convert_color_optional_params =
            get_default_optional_params<kernel_selector::convert_color_optional_params>(arg.get_program());

        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            convert_color_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
        }

        convert_color_params.input_color_format = static_cast<kernel_selector::color_format>(primitive->input_color_format);
        convert_color_params.output_color_format = static_cast<kernel_selector::color_format>(primitive->output_color_format);
        convert_color_params.mem_type = static_cast<kernel_selector::memory_type>(primitive->mem_type);

        auto& kernel_selector = kernel_selector::convert_color_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(convert_color_params, convert_color_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto convert_color = new convert_color_impl(arg, best_kernels[0]);

        return convert_color;
    }
};

namespace detail {

attach_convert_color_impl::attach_convert_color_impl() {
    auto types = {data_types::u8, data_types::f16, data_types::f32};
    auto formats = {
        format::byxf,
        format::nv12,
    };

    implementation_map<convert_color>::add(impl_types::ocl, convert_color_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
