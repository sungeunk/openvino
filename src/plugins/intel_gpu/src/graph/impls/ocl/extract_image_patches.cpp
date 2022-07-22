// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"

#include "extract_image_patches/extract_image_patches_kernel_selector.h"
#include "extract_image_patches/extract_image_patches_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct extract_image_patches_impl : typed_primitive_impl_ocl<extract_image_patches> {
    using parent = typed_primitive_impl_ocl<extract_image_patches>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<extract_image_patches_impl>(*this);
    }

public:
    static primitive_impl* create(const extract_image_patches_node& arg, const kernel_impl_params& impl_param) {
        const auto& prim = arg.get_primitive();
        auto params = get_default_params<kernel_selector::extract_image_patches_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::extract_image_patches_optional_params>(arg.get_program());

        params.sizes = prim->sizes;
        params.strides = prim->strides;
        params.rates = prim->rates;
        params.auto_pad = prim->auto_pad;

        auto& kernel_selector = kernel_selector::extract_image_patches_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto extract_image_patches = new extract_image_patches_impl(arg, best_kernels[0]);

        return extract_image_patches;
    }
};

namespace detail {

attach_extract_image_patches_impl::attach_extract_image_patches_impl() {
    auto types = {data_types::u8, data_types::i8, data_types::f16, data_types::f32, data_types::i32, data_types::i64};
    auto formats = {
        format::bfyx,
    };

    implementation_map<extract_image_patches>::add(impl_types::ocl, extract_image_patches_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
