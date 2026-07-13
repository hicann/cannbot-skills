// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect } from "vitest";

describe("i18n", () => {
  it("returns value for known key", async () => {
    const { t, setLanguage } = await import("../src/utils/i18n.js");
    setLanguage("zh_CN");
    const value = t("wizard_title");
    expect(value).not.toBe("wizard_title");
    expect(typeof value).toBe("string");
    expect(value.length).toBeGreaterThan(0);
  });

  it("returns key itself for unknown key", async () => {
    const { t } = await import("../src/utils/i18n.js");
    const value = t("nonexistent_key_12345");
    expect(value).toBe("nonexistent_key_12345");
  });

  it("switches language with setLanguage", async () => {
    const { t, setLanguage } = await import("../src/utils/i18n.js");
    setLanguage("en_US");
    const enValue = t("wizard_cancel");
    setLanguage("zh_CN");
    const zhValue = t("wizard_cancel");
    expect(enValue).not.toBe(zhValue);
  });

  it("returns string type for all common keys", async () => {
    const { t, setLanguage } = await import("../src/utils/i18n.js");
    setLanguage("zh_CN");
    const keys = ["wizard_title", "list_title", "install_done", "doctor_title", "error_plugin_not_found"];
    for (const key of keys) {
      const value = t(key);
      expect(typeof value).toBe("string");
    }
  });

  it("handles en_US locale correctly", async () => {
    const { t, setLanguage } = await import("../src/utils/i18n.js");
    setLanguage("en_US");
    const value = t("doctor_title");
    expect(value).not.toBe("doctor_title");
    expect(typeof value).toBe("string");
  });
});

describe("constants", () => {
  it("BACK is __back__", async () => {
    const { BACK } = await import("../src/utils/constants.js");
    expect(BACK).toBe("__back__");
  });

  it("CANCEL is __cancel__", async () => {
    const { CANCEL } = await import("../src/utils/constants.js");
    expect(CANCEL).toBe("__cancel__");
  });
});
