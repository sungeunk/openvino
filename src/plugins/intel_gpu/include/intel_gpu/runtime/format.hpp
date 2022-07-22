// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <list>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <stdexcept>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_memory Memory description and management
/// @{

/// @brief Format information helper class.
struct format_traits {
    /// @brief String representation of a format.
    std::string str;
    /// @brief Number of batch dimensions in a format.
    size_t batch_num;
    /// @brief Number of feature map/channel dimensions in a format.
    size_t feature_num;
    /// @brief Number of spatial (x,y) dimensions in a format.
    size_t spatial_num;
    /// @brief Number of groups in a format.
    size_t group_num;
    /// @brief Dimensions order. Default {0, 1, 2, ... rank }
    std::vector<size_t> _order;
    /// @brief Dimensions changing order from rare to often.
    std::string order;
    /// @brief Dimensions order for internal storage.
    std::string internal_order;
    /// @brief Block sizes as a vector of pairs of dimension number and block size ordered from rare to often.
    std::vector<std::pair<size_t, int>> block_sizes;
    /// @brief Characters representing batch dimensions in an order.
    static const char* batch_chars() { return "bno"; }
    /// @brief Characters representing feature map/channel dimensions in an order.
    static const char* feature_chars() { return "fic"; }
    /// @brief Characters representing spatial dimensions in an order.
    static const char* spatial_chars() { return "xyzhsw"; }
    /// @brief Characters representing group dimensions in an order.
    static const char* group_chars() { return "g"; }
    /// @brief Checks if @p c represents batch dimension.
    static bool is_batch_char(char c) { return std::string(batch_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents feature map/channel dimension.
    static bool is_feature_char(char c) { return std::string(feature_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents spatial dimension.
    static bool is_spatial_char(char c) { return std::string(spatial_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents group dimensions.
    static bool is_group_char(char c) { return std::string(group_chars()).find_first_of(c) != std::string::npos; }
};

/// @brief Represents memory formats (orders).
/// @n In CNN most of data is described as 4 dimensional blocks. In GPU plugin we describe memory with 4 letters
/// - b - number of blocks in batch. For weights formats: output features - conv, neurons - inner product
/// - f - number of feature maps, features or channels. For weights formats: input features - conv, inputs, inner product
/// - x - spatial, width
/// - y - spatial, height
/// /n
/// For explanation how each format type is implemented in memory we will use naming shown bellow:
struct format {
    enum type : int32_t {
        // Data formats
        bfyx,                                   ///< the most common format for activations in clDNN.
        bfzyx,                                  ///< format for 5d data tensors
        bfwzyx,                                 ///< batch, feature, 4D spatial
        yxfb,                                   ///< batch first, feature and than spatials
        byxf,                                   ///< used in bitmaps, input from user i.e b images of RGB format
        fyxb,                                   ///< format not used inside clDNN, but supported in reorder as extension
                                                ///< for user provided formats.
        b_fs_yx_fsv2,
        b_fs_zyx_fsv2,
        b_fs_yx_fsv4,                           ///< format for input for IMAD convolutions
        b_fs_zyx_fsv4,                          ///< format for input for IMAD 3D convolutions
        b_fs_yx_fsv16,                          ///< format used for blocked convolution
        b_fs_yx_fsv32,                          ///< format used for blocked int8 convolution
        b_fs_zyx_fsv16,                         ///< format used for 3D blocked convolution (features blocked by 16)
        b_fs_zyx_fsv32,                         ///< format used for blocked int8 3d convolution
        bs_fs_yx_bsv16_fsv32,                   ///< format used for 2D blocked convolution (batch and features blocked by 16 and 32)
        bs_fs_zyx_bsv16_fsv32,                  ///< format used for 3D blocked convolution (batch and features blocked by 16 and 32)
        bs_fs_zyx_bsv16_fsv16,                  ///< format used for 3D blocked convolution (batch and features blocked by 16)
        bs_fs_yx_bsv16_fsv16,                   ///< format used for 2D blocked convolution (batch and features blocked by 16)
        bs_fs_yx_bsv4_fsv4,                     ///< format used for 2D blocked convolution (batch and features blocked by 4)
        bs_fs_yx_bsv8_fsv4,                     ///< format used for 2D blocked convolution (batch and features blocked by 8 and 4)
        bs_fs_zyx_bsv8_fsv4,                    ///< format used for 3D blocked convolution (batch and features blocked by 8 and 4)
        bs_fs_yx_bsv8_fsv2,                     ///< format used for 2D blocked convolution (batch and features blocked by 8 and 2)
        bs_fs_zyx_bsv8_fsv2,                    ///< format used for 3D blocked convolution (batch and features blocked by 8 and 2)
        bs_fs_yx_bsv4_fsv2,                     ///< format used for 2D blocked convolution (batch blocked by 4, features blocked by 2)
        bs_fs_zyx_bsv4_fsv4,                    ///< format used for 3D blocked convolution (batch and features blocked by 4)
        bs_fs_zyx_bsv4_fsv2,                    ///< format used for 3D blocked convolution (batch blocked by 4, features blocked by 2)
        bs_fs_yx_bsv32_fsv32,                   ///< format used for big batches (batch and features blocked by 32)
        bs_fs_yx_bsv32_fsv16,                   ///< format used for big batches (batch blocked by 32, features blocked by 16)
        bs_fs_zyx_bsv32_fsv32,                  ///< format used for big batches (batch and features blocked by 32)
        bs_fs_zyx_bsv32_fsv16,                  ///< format used for big batches (batch blocked by 32, features blocked by 16)
        fs_b_yx_fsv32,                          ///< format for input for fp16 primitives
        bs_xs_xsv8_bsv8,                        ///< format used only for fully connected
        bs_xs_xsv8_bsv16,                       ///< format used only for fully connected
        bs_x_bsv16,                             ///< format used only for fully connected weights fp16 batch=1 : bs - batch slice
                                                ///< (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx)
        b_fs_yx_32fp,                           ///< format for data for binary convolutions
        winograd_2x3_s1_data,                   ///< format used for input for winograd convolution, F(2,3) -- filter 3x3 with stride 1
        nv12,                                   ///< format for media nv12 input
        image_2d_rgba,                          ///< format for image2d RGBA, always allocates memory for 4 feature maps (even when only 3 are used)

        // Weights formats
        oiyx,                                         ///< the most common format for 2D weights
        ioyx,                                         ///< 2D weights format for deconvolutions
        yxio,                                         ///< format used 2D weights
        oizyx,                                        ///< the most common format for 3D convolution
        iozyx,                                        ///< 3D weights format for deconvolutions
        iyxo,
        oyxi,
        os_iyx_osv16,                                 ///< format used only for convolution weights
        o_is_yx_isv16,                                ///< format used only for convolution weights
        os_yxi_osv16,                                 ///< format used only for convolution weights
        os_is_yx_osv16_isv16,                         ///< format used for convolution i8 weights
        os_is_zyx_osv32_isv16,
        os_is_zyx_osv64_isv16,
        os_zyxi_osv16,                                ///< format used for weights for 3D convolution
        os_is_yx_isv16_osv16,                         ///< format used for blocked convolution
        os_is_zyx_isv16_osv16,                        ///< format used for weights for blocked 3D convolution
        is_os_zyx_isv16_osv16,                        ///< format used for weights for blocked 3D deconvolution
        is_os_yx_isv16_osv16,                         ///< format used for weights for blocked deconvolution
        os_is_yx_isv8_osv16_isv2,                     ///< format used for weights for blocked 2D convolution
        os_is_zyx_isv8_osv16_isv2,                    ///< format used for weights for blocked 3D convolution
                                                      ///< os - output feature maps slice, i - input feature maps,
                                                      ///< yx - spatials, sv16 - 16 values of single slice.
        os_iyx_osv32,                                 ///< format used only for convolution weights:
                                                      ///< os - output feature maps slice, i - input feature maps,
                                                      ///< yx - spatials, sv32 - 32 values of single slice.
        os_iyx_osv64,                                 ///< format used only for convolution weights:
                                                      ///< os - output feature maps slice, i - input feature maps,
                                                      ///< yx - spatials, sv64 - 64 values of single slice.
        image_2d_weights_c4_fyx_b,                    ///< image format for weights, width size is f*y*x/4
                                                      ///< (4-channels filled with fyx data), height is b
        image_2d_weights_c1_b_fyx,                    ///< image format for weights, width size is b,
                                                      ///< height is f*y*x, single channel
        winograd_2x3_s1_weights,                      ///< format used for weights for winograd non-fused
                                                      ///< convolution, F(2,3) -- filter 3x3 with stride 1
        winograd_2x3_s1_fused_weights,                ///< format used for weights for winograd fused
                                                      ///< convolution, F(2,3) -- filter 3x3 with stride 1
        winograd_6x3_s1_fused_weights,                ///< format used for weights for winograd fused
                                                      ///< convolution, F(6,3) -- filter 3x3 with stride 1
        image_2d_weights_winograd_6x3_s1_fbxyb,       ///< image format used for weights for winograd fused
                                                      ///< convolution, F(6,3) -- filter 3x3 with stride 1
        image_2d_weights_winograd_6x3_s1_xfbyb,       ///< image format used for weights for winograd fused
                                                      ///< convolution, F(6,3) -- filter 3x3 with stride 1
        os_is_yx_isa8_osv8_isv4,                      ///< format for weights for MMAD convolution
        os_is_zyx_isa8_osv8_isv4,                     ///< format for weights for MMAD convolution
        os_is_yx_isa8_osv16_isv4,                     ///< format for weights for fully connected MMAD
        os_is_zyx_isa8_osv16_isv4,                    ///< format for weights for fully connected MMAD
        os_is_yx_isa8_osv8_isv4_swizzled_by_4,        ///< format for weights for MMAD convolution
        os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,   ///< format for weights for MMAD fsv32 convolution
        os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4,  ///< format for weights for MMAD fsv32 convolution
        os_is_yx_osa4_isa8_osv8_isv2,                 ///< format for weights for MMAD fsv32 convolution
        os_is_zyx_osa4_isa8_osv8_isv2,                ///< format for weights for MMAD fsv32 convolution
        os_is_zyx_osa4_isa8_osv8_isv4,                ///< format for weights for MMAD fsv32 convolution
        os_is_yx_osa4_isa8_osv8_isv4,                 ///< format for weights for MMAD fsv32 convolution
        os_is_yx_osa2_isa8_osv8_isv2,
        os_is_zyx_osa2_isa8_osv8_isv2,
        os_is_yx_osa2_isa8_osv16_isv2,
        os_is_yx_osa2_isa8_osv16_isv4,
        os_is_yx_isa8_osv8_isv2,
        is_os_yx_isa8_osv8_isv2,
        os_is_zyx_isa8_osv8_isv2,
        is_os_zyx_isa8_osv8_isv2,
        is_os_yx_isa2_osa8_isv8_osv2,
        is_os_yx_isa4_osa8_isv8_osv4,
        is_os_yx_osa4_isa8_osv8_isv4,
        is_o_yx_isv32,                                ///< format for weights for 1x1 MMAD convolutions
        is_o32_yx_isv32_swizzled_by_4,                ///< format for weights for 1x1 MMAD convolutions
        os_is_y_x8_osv8_isv4,                         ///< format for weights for 1x1 MMAD convolutions
        os_is_y_x8_osv8_isv4_swizzled_by_4,           ///< format for weights for 1x1 MMAD convolutions
        os_is_yx_osv16_isv4,                          ///< format for weights for IMAD convolutions
        os_is_yx_osv8_isv4,                           ///< format used for convolution i8 weights
        os_is_zyx_osv8_isv4,                          ///< format used for convolution i8 weights
        os_is_yx_osv8_isv2,                           ///< format used for convolution fp16 weights
        os_is_zyx_osv8_isv2,                          ///< format used for convolution fp16 weights
        os_is_zyx_osv16_isv16,                        ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv4_swizzled_by_2,            ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv4,                          ///< format for weights for IMAD convolutions
        os_is_zyx_osv32_isv4,                         ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv32p,                        ///< format for weights for binary convolutions
        lstm_weights_dio,                             ///< dynamic_lstm, direction,
                                                      ///< than IO (I - input size, O - 4 * hidden_size)
        os_is_osv32_isv32_swizzled_by_4,              ///< format for weights for 1x1 IMAD convolution
        os_iyx_osv32__ai32,
        iy_xs_os_xsv2_osv8__ao32,
        iy_xs_os_xsv2_osv16__ao32,
        i_yxs_os_yxsv2_osv16,
        os_i_yxs_osv4_yxsv4,
        os_i_osv16__ai8,                              ///< format used only for fully connected weights
        os_i_osv8__ai8,                               ///< format used only for fully connected weights

        goiyx,                                        ///< format used for weights for 2D convolution
        gioyx,                                        ///< format used for weights for 2D deconvolution
        gyxio,                                        ///< format used for weights for 2D convolution
        goizyx,                                       ///< format used for weights for 3D convolution
        giozyx,                                       ///< format used for weights for 3D deconvolution
        g_os_iyx_osv16,                               ///< format used for weights for 2D convolution
        g_os_iyx_osv32,                               ///< format used for weights for 2D convolution
        gs_oiyx_gsv16,                                ///< format used for weights for 2D convolution
        gs_oizyx_gsv16,                               ///< format used for weights for 3D convolution
        gs_oiyx_gsv32,                                ///< format used for weights for 2D convolution
        gs_oizyx_gsv32,                               ///< format used for weights for 3D convolution
        g_is_os_zyx_isv16_osv16,                      ///< format used for grouped weights for blocked 3D deconvolution
        g_os_is_yx_osv16_isv4,
        g_os_is_zyx_osv16_isv16,
        g_is_os_yx_isv16_osv16,
        g_os_is_yx_isa8_osv8_isv2,
        g_os_is_zyx_isv8_osv16_isv2,
        g_os_is_yx_isv8_osv16_isv2,
        g_os_is_zyx_isv16_osv16,
        g_os_zyx_is_osv16_isv4,                       ///< format for imad deconvolution
        g_os_zyx_is_osv16_isv16,                      ///< format for imad deconvolution
        g_os_zyx_is_osv16_isv32,                      ///< format for imad deconvolution
        g_os_zyx_is_osv32_isv4,                       ///< format for imad deconvolution
        g_os_zyx_is_osv32_isv16,                      ///< format for imad deconvolution
        g_os_zyx_is_osv32_isv32,                      ///< format for imad deconvolution
        g_os_is_yx_isv16_osv16,
        g_os_is_yx_osv8_isv2,
        g_os_is_yx_osv8_isv4,
        gs_oi_yxs_gsv4_yxsv4,
        gs_oi_yxs_gsv16_yxsv4,
        gs_oi_yxs_gsv32_yxsv4,
        gi_yxs_os_yxsv2_osv16,
        giy_xs_os_xsv2_osv8__ao32,
        giy_xs_os_xsv2_osv16__ao32,
        g_os_is_yx_osa2_isa8_osv8_isv2,
        g_os_is_yx_osa4_isa8_osv8_isv4,
        g_os_is_yx_osa4_isa8_osv8_isv2,
        g_os_is_yx_osa2_isa8_osv16_isv2,
        g_os_is_yx_osa2_isa8_osv16_isv4,
        g_os_is_zyx_osa4_isa8_osv8_isv2,
        g_os_is_zyx_osa4_isa8_osv8_isv4,

        format_num,  ///< number of format types
        any        = -1
    };

    /// @brief Get format traits for particular @p format::type
    static const format_traits& traits(type fmt);
    /// @brief Returns number of batch dimensions for a @p format.
    static size_t batch_num(type fmt) { return traits(fmt).batch_num; }
    /// @brief Returns number of feature dimensions for a @p format.
    static size_t feature_num(type fmt) { return traits(fmt).feature_num; }
    /// @brief Returns number of spatial dimensions for a @p format.
    static size_t spatial_num(type fmt) { return traits(fmt).spatial_num; }
    /// @brief Returns number of group dimensions for a @p format.
    static size_t group_num(type fmt) { return traits(fmt).group_num; }
    /// @brief Returns an order of dimensions for a @ format.
    static const std::string& order(type fmt) { return traits(fmt).order; }
    /// @brief Returns an internal orders of dimensions for a @p format.
    static const std::string& internal_order(type fmt) { return traits(fmt).internal_order; }
    /// @brief Returns block sizes for @p format.
    static const std::vector<std::pair<size_t, int>>& block_sizes(type fmt) { return traits(fmt).block_sizes; }
    /// @brief Returns number of dimensions contained within a @p format
    static size_t dimension(type fmt) { return order(fmt).size(); }
    /// @brief Checks if @p format is a winograd format
    static bool is_winograd(type fmt) {
        return (fmt == winograd_2x3_s1_data ||
                fmt == winograd_2x3_s1_weights ||
                fmt == winograd_2x3_s1_fused_weights ||
                fmt == winograd_6x3_s1_fused_weights ||
                fmt == image_2d_weights_winograd_6x3_s1_fbxyb ||
                fmt == image_2d_weights_winograd_6x3_s1_xfbyb); }
    /// @brief Checks if @p format is of image2d type
    static bool is_image_2d(type fmt) {
        return (fmt == image_2d_weights_c4_fyx_b ||
                fmt == image_2d_weights_c1_b_fyx ||
                fmt == image_2d_weights_winograd_6x3_s1_fbxyb ||
                fmt == image_2d_weights_winograd_6x3_s1_xfbyb ||
                fmt == nv12 ||
                fmt == image_2d_rgba);
    }
    /// @brief Checks if @p format is weights format
    static bool is_weights_format(type fmt) {
        const auto internal_order = traits(fmt).internal_order;
        const auto weights_chars = { "o", "i" };
        for (const auto& c : weights_chars) {
            if (internal_order.find_first_of(c) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    /// @brief Checks if @p format is simple data format
    static bool is_simple_data_format(type fmt) {
        return (fmt == yxfb || fmt == byxf ||
                fmt == bfyx || fmt == fyxb ||
                fmt == bfzyx || fmt == bfwzyx);
    }

    static format get_default_format(size_t rank, bool is_weights = false, bool is_grouped = false) {
        auto default_fmt = cldnn::format::bfyx;
        if (is_weights) {
            if (is_grouped) {
                if (rank == 5) {
                    default_fmt = cldnn::format::goiyx;
                } else if (rank == 6) {
                    default_fmt = cldnn::format::goizyx;
                }
            } else {
                if (rank == 4) {
                    default_fmt = cldnn::format::oiyx;
                } else if (rank == 5) {
                    default_fmt = cldnn::format::oizyx;
                }
            }
        } else {
            if (rank == 5) {
                default_fmt = cldnn::format::bfzyx;
            } else if (rank == 6) {
                default_fmt = cldnn::format::bfwzyx;
            }
        }
       return default_fmt;
    }

    /// @brief Checks if @p format is of grouped type
    static bool is_grouped(type fmt) { return group_num(fmt) != 0; }
    /// @brief Checks if @p format is of image type
    static bool is_image(type fmt) { return (is_image_2d(fmt)); }
    /// @brief Checks if @p format is blocked format
    static bool is_blocked(type fmt) { return !(block_sizes(fmt).empty()); }
    /// @brief Checks if @p format is nv12 format
    static bool is_nv12(type fmt) { return (fmt == nv12); }

    /// @brief Returns number of batch dimensions.
    size_t batch_num() const { return traits(value).batch_num; }
    /// @brief Returns number of feature dimensions.
    size_t feature_num() const { return traits(value).feature_num; }
    /// @brief Returns number of spatial dimensions.
    size_t spatial_num() const { return traits(value).spatial_num; }
    /// @brief Returns number of group dimensions.
    size_t group_num() const { return traits(value).group_num; }
    /// @brief Returns an order of dimensions in form of string.
    const std::string& order() const { return traits(value).order; }
    /// @brief Returns an internal orders of dimensions form of string.
    const std::string& internal_order() const { return traits(value).internal_order; }
    /// @brief Returns block sizes as vector of pairs of dimension and block size for that dimension.
    const std::vector<std::pair<size_t, int>>& block_sizes() const { return traits(value).block_sizes; }
    /// @brief Returns number of dimensions contained within this format
    size_t dimension() const { return order(value).size(); }
    /// @brief Checks if @p format is a winograd format
    bool is_winograd() const { return is_winograd(value); }
    /// @brief Checks if @p format is of image 2d type
    bool is_image_2d() const { return is_image_2d(value); }
    /// @brief Checks if @p format is of image type
    bool is_image() const { return is_image(value); }
    /// @brief Checks if @p format is blocked format
    bool is_blocked() { return is_blocked(value); }
    /// @brief Checks if @p format is a nv12 format
    bool is_nv12() const { return is_nv12(value); }

    /// @brief Transforms dimension from internal order to external order
    size_t internal_to_external(size_t idx) const {
        auto index = order().find_first_of(internal_order()[idx]);
        if (index == std::string::npos)
            throw std::invalid_argument("Internal dimension index does not map to external index.");
        return index;
    }

    type value;
    /// @brief Implicit conversion from format::type.
    constexpr format(type t) : value(t) {}
    /// @brief Implicit conversion to format::type.
    constexpr operator type() const { return value; }

    std::string to_string() const;
};

/// @}
/// @}
}  // namespace cldnn
