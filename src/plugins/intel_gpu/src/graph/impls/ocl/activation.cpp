// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation/activation_kernel_base.h"
#include "activation/activation_kernel_selector.h"
#include "activation_inst.h"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct activation_impl : typed_primitive_impl_ocl<activation> {
    using parent = typed_primitive_impl_ocl<activation>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<activation_impl>(*this);
    }

    explicit activation_impl(const activation_impl& other) : parent(other),
        _is_parameterized(other._is_parameterized) {}

    activation_impl(const activation_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {
        set_node_params(arg);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<activation>());
        const auto& node = arg.as<activation>();
        _is_parameterized = node.is_parameterized();
    }

    kernel_arguments_data get_arguments(typed_primitive_inst<activation>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        if (_is_parameterized) {
            args.slope = instance.slope_memory();
        }

        return args;
    }
    static primitive_impl* create(const activation_node& arg, const kernel_impl_params& impl_param) {
        const auto& prim = arg.get_primitive();
        auto activation_params = get_default_params<kernel_selector::activation_params>(impl_param);
        auto activation_optional_params =
            get_default_optional_params<kernel_selector::activation_optional_params>(arg.get_program());

        convert_new_activation_func(prim, activation_params.activations);

        if (arg.is_parameterized()) {
            const auto& slope_layout = impl_param.input_layouts[1];
            const auto& output_layout = impl_param.output_layout;

            const auto params_num =
                kernel_selector::GetActivationAdditionalParamsNumber(activation_params.activations[0].function);

            CLDNN_ERROR_LESS_THAN(arg.id(),
                                  "Slope layout size count",
                                  slope_layout.count(),
                                  "output_layout.feature() * params_num",
                                  static_cast<size_t>(output_layout.feature() * params_num),
                                  "Error - not enough data inside additional params buffer");

            activation_params.inputActivationParams.push_back(convert_data_tensor(slope_layout));
        }

        auto& kernel_selector = kernel_selector::activation_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(activation_params, activation_optional_params);
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto activation = new activation_impl(arg, best_kernels[0]);

        return activation;
    }

private:
    bool _is_parameterized;
};

namespace detail {

attach_activation_impl::attach_activation_impl() {
    auto types = {data_types::u8, data_types::i8, data_types::f16, data_types::f32, data_types::i32};
    auto formats = {
        format::bfwzyx,
        format::bfyx,
        format::bfzyx,
        format::byxf,
        format::yxfb,
        format::b_fs_yx_fsv16,
        format::b_fs_zyx_fsv16,
        format::fs_b_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
    };

    implementation_map<activation>::add(impl_types::ocl, activation_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
