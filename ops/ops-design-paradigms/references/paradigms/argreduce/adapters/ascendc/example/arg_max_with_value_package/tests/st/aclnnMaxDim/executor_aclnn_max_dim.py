#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr


# 注：ATK 测试框架按 executor_<ApiName>.py 命名规则发现执行器模块；
# 本镜像为满足 CodeCheck G.NAM.01 改为 snake_case 命名，如需接入测试框架运行，
# 请改回 executor_aclnnMaxDim.py 并将类名对齐框架注册约定
@register("aclnn_max_dim")
class ExecMax(AclnnBaseApi):
    def get_format(self, input_data: InputDataset, index=None, name=None):
        if input_data.kwargs["format"] == "NCL":
            return AclFormat.ACL_FORMAT_NCL
        if input_data.kwargs["format"] == "NCHW":
            return AclFormat.ACL_FORMAT_NCHW
        if input_data.kwargs["format"] == "NCDHW":
            return AclFormat.ACL_FORMAT_NCDHW
        if input_data.kwargs["format"] == "ND":
            return AclFormat.ACL_FORMAT_ND
        return AclFormat.ACL_FORMAT_ND