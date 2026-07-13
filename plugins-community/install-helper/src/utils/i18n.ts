// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import zhCN from "../locales/zh_CN.json" with { type: "json" };
import enUS from "../locales/en_US.json" with { type: "json" };

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

type Messages = Record<string, string>;

const EMBEDDED_LOCALES: Record<string, Messages> = {
  zh_CN: zhCN as Messages,
  en_US: enUS as Messages,
};

function loadLocale(locale: string): Messages {
  const embedded = EMBEDDED_LOCALES[locale];
  if (embedded) return embedded;

  const localePath = join(__dirname, "locales", `${locale}.json`);
  if (existsSync(localePath)) {
    try {
      const content = readFileSync(localePath, "utf-8");
      return JSON.parse(content);
    } catch {
    }
  }

  const fallbackPath = join(__dirname, "locales", "zh_CN.json");
  if (existsSync(fallbackPath)) {
    try {
      const content = readFileSync(fallbackPath, "utf-8");
      return JSON.parse(content);
    } catch {
    }
  }

  return EMBEDDED_LOCALES.zh_CN || {};
}

let currentLang: "zh_CN" | "en_US" = "zh_CN";
let messages: Messages = loadLocale("zh_CN");

export function setLanguage(lang: "zh_CN" | "en_US"): void {
  currentLang = lang;
  messages = loadLocale(lang);
}

export function t(key: string): string {
  return messages[key] !== undefined ? messages[key] : key;
}
