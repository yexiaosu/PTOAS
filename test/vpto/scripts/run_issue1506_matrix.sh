#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
: "${RESULT_ROOT:?Set RESULT_ROOT to an isolated result directory}"
: "${PTOAS_BIN:?Set PTOAS_BIN to this worktree virtual environment entry point}"
mkdir -p "${RESULT_ROOT}"
export DEVICE=SIM COMPILE_ONLY=0 COMPARE_STRICT=1
export PTO_ISA_PATH="${PTO_ISA_PATH:?Set PTO_ISA_PATH}"
export PTOAS_FLAGS='--pto-arch=a5 --pto-backend=vpto --vpto-scheduler=off --enable-bisheng-vec-misched=false'
WORK_SPACE="${RESULT_ROOT}/smoke" CASE_NAME=micro-op/binary-vector/vadd \
    bash "${SCRIPT_DIR}/run_host_vpto_validation.sh" > "${RESULT_ROOT}/smoke.log" 2>&1

result=0
for spec in off:false on:false off:true on:true; do
    scheduler="${spec%%:*}"
    bisheng="${spec##*:}"
    group="ptoas-${scheduler}_bisheng-${bisheng}"
    export PTOAS_FLAGS="--pto-arch=a5 --pto-backend=vpto --vpto-scheduler=${scheduler}"
    PTOAS_FLAGS+=" --enable-bisheng-vec-misched=${bisheng}"
    for repeat in 1 2 3; do
        output="${RESULT_ROOT}/${group}/run${repeat}"
        mkdir -p "${output}"
        echo "START ${group} run${repeat}"
        if WORK_SPACE="${output}" CASE_NAME=kernels/issue-1506-vec-misched \
            timeout 900 bash "${SCRIPT_DIR}/run_host_vpto_validation.sh" > "${output}/runner.log" 2>&1; then
            echo "PASS ${group} run${repeat}"
        else
            echo "FAIL ${group} run${repeat}"
            result=1
            break
        fi
    done
done
exit "${result}"
