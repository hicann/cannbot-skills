// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Minimal pure-JS zip writer (STORE method, no compression).
// Recreates a folder tree from { path, data } entries; valid for unzip on any platform.

export interface ZipEntry {
  path: string;
  data: Uint8Array;
}

const CRC_TABLE: Uint32Array = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes: Uint8Array): number {
  let c = ~0;
  for (const byte of bytes) {
    const idx = (c ^ byte) & 0xff;
    c = (c >>> 8) ^ CRC_TABLE[idx];
  }
  return (c ^ 0xffffffff) >>> 0;
}

function u16(n: number): Uint8Array {
  return new Uint8Array([n & 0xff, (n >>> 8) & 0xff]);
}

function u32(n: number): Uint8Array {
  const v = n >>> 0;
  return new Uint8Array([v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff]);
}

function concat(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

function textToBytes(s: string): Uint8Array {
  return new TextEncoder().encode(s);
}

export function normalizeZipPath(path: string): string {
  let p = path.replace(/\\/g, "/");
  while (p.startsWith("/")) p = p.slice(1);
  // zip entries must be relative; drop any drive prefix like "C:/"
  if (/^[a-zA-Z]:\//.test(p)) p = p.slice(p.indexOf("/") + 1);
  return p;
}

const DOS_DATE = 0x5c21; // 2026-01-01 in DOS date format ((2026-1980)<<9 | 1<<5 | 1)
const UTF8_FLAG = 0x0800; // general-purpose bit 11: filenames are UTF-8 (required by Windows Explorer for non-ASCII names)

export function buildZip(entries: ZipEntry[]): Blob {
  const localParts: Uint8Array[] = [];
  const centralParts: Uint8Array[] = [];

  let offset = 0;
  for (const entry of entries) {
    const name = textToBytes(normalizeZipPath(entry.path));
    const data = entry.data;
    const crc = crc32(data);
    const localOffset = offset;

    const localHeader = concat([
      u32(0x04034b50), // local file header signature
      u16(20),         // version needed
      u16(UTF8_FLAG),  // flags (UTF-8 filenames)
      u16(0),          // method (STORE)
      u16(0),          // mod time
      u16(DOS_DATE),   // mod date
      u32(crc),
      u32(data.length),  // compressed size
      u32(data.length),  // uncompressed size
      u16(name.length),
      u16(0),          // extra length
    ]);

    localParts.push(localHeader, name, data);
    offset += localHeader.length + name.length + data.length;

    centralParts.push(concat([
      u32(0x02014b50), // central dir header signature
      u16(20),         // version made by
      u16(20),         // version needed
      u16(UTF8_FLAG),  // flags (UTF-8 filenames)
      u16(0),          // method
      u16(0),          // mod time
      u16(DOS_DATE),   // mod date
      u32(crc),
      u32(data.length),
      u32(data.length),
      u16(name.length),
      u16(0),          // extra
      u16(0),          // comment
      u16(0),          // disk start
      u16(0),          // internal attrs
      u32(0),          // external attrs
      u32(localOffset),
    ]), name);
  }

  const localBytes = concat(localParts);
  const centralBytes = concat(centralParts);
  const end = concat([
    u32(0x06054b50), // end of central dir signature
    u16(0),
    u16(0),
    u16(entries.length),
    u16(entries.length),
    u32(centralBytes.length),
    u32(localBytes.length),
    u16(0),          // comment length
  ]);

  return new Blob([localBytes, centralBytes, end], { type: "application/zip" });
}
