// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { crc32, normalizeZipPath, buildZip, type ZipEntry } from '../src/lib/zip-store.ts';

function bytes(s: string): Uint8Array {
  return new TextEncoder().encode(s);
}

function findBytes(haystack: Uint8Array, needle: number[]): number {
  outer: for (let i = 0; i <= haystack.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j] !== needle[j]) continue outer;
    }
    return i;
  }
  return -1;
}

const PK_LOCAL = [0x50, 0x4b, 0x03, 0x04];
const PK_CENTRAL = [0x50, 0x4b, 0x01, 0x02];
const PK_END = [0x50, 0x4b, 0x05, 0x06];

describe('zip-store', () => {
  describe('crc32', () => {
    it('known vector: crc32("123456789") == 0xCBF43926', () => {
      expect(crc32(bytes('123456789'))).toBe(0xCBF43926);
    });
    it('crc32 of empty is 0', () => {
      expect(crc32(new Uint8Array(0))).toBe(0);
    });
  });

  describe('normalizeZipPath', () => {
    it('strips leading slashes', () => {
      expect(normalizeZipPath('/a/b/c.ts')).toBe('a/b/c.ts');
    });
    it('converts backslashes', () => {
      expect(normalizeZipPath('C:\\Users\\me\\foo.ts')).toBe('Users/me/foo.ts');
    });
    it('strips drive prefix when followed by slash', () => {
      expect(normalizeZipPath('C:/Users/me/foo.ts')).toBe('Users/me/foo.ts');
    });
  });

  describe('buildZip', () => {
    it('produces valid zip signatures and structure', async () => {
      const entries: ZipEntry[] = [
        { path: 'docs/a.md', data: bytes('# A\nhello') },
        { path: 'docs/b.md', data: bytes('# B\nworld') },
      ];
      const blob = buildZip(entries);
      expect(blob.type).toBe('application/zip');
      const ab = await blob.arrayBuffer();
      const arr = new Uint8Array(ab);

      // local file header at start
      expect(findBytes(arr, PK_LOCAL)).toBe(0);
      // central directory header present
      const cd = findBytes(arr, PK_CENTRAL);
      expect(cd).toBeGreaterThan(0);
      // end of central directory present
      const eocd = findBytes(arr, PK_END);
      expect(eocd).toBeGreaterThan(cd);

      // file paths appear
      expect(findBytes(arr, [...bytes('docs/a.md')])).toBeGreaterThan(0);
      expect(findBytes(arr, [...bytes('# A')])).toBeGreaterThan(0);
    });

    it('writes correct CRC for each entry in the local header', async () => {
      const data = bytes('hello zip\n');
      const blob = buildZip([{ path: 'x.txt', data }]);
      const arr = new Uint8Array(await blob.arrayBuffer());
      // local header layout: sig(4) ver(2) flags(2) method(2) time(2) date(2) crc(4) ...
      // crc is at offset 14..18
      const dv = new DataView(arr.buffer, arr.byteOffset + 14, 4);
      expect(dv.getUint32(0, true)).toBe(crc32(data));
    });

    it('sets UTF-8 flag (bit 11) in local + central headers for Windows Explorer compatibility', async () => {
      const blob = buildZip([{ path: 'docs/状态.md', data: bytes('x') }]);
      const arr = new Uint8Array(await blob.arrayBuffer());
      const localFlags = new DataView(arr.buffer, arr.byteOffset + 6, 2).getUint16(0, true);
      expect(localFlags & 0x0800).toBe(0x0800);
      const cd = findBytes(arr, PK_CENTRAL);
      const cdFlags = new DataView(arr.buffer, arr.byteOffset + cd + 8, 2).getUint16(0, true);
      expect(cdFlags & 0x0800).toBe(0x0800);
    });

    it('encodes a valid DOS date (2026-01-01) decodeable by zipfile', async () => {
      const blob = buildZip([{ path: 'a.txt', data: bytes('1') }]);
      const arr = new Uint8Array(await blob.arrayBuffer());
      const date = new DataView(arr.buffer, arr.byteOffset + 12, 2).getUint16(0, true);
      expect(date).toBe(0x5c21);
    });

    it('end-of-central-directory records the entry count', async () => {
      const blob = buildZip([
        { path: 'a.txt', data: bytes('1') },
        { path: 'b.txt', data: bytes('2') },
        { path: 'c.txt', data: bytes('3') },
      ]);
      const arr = new Uint8Array(await blob.arrayBuffer());
      const eocd = findBytes(arr, PK_END);
      const dv = new DataView(arr.buffer, arr.byteOffset + eocd + 10, 2);
      expect(dv.getUint16(0, true)).toBe(3);
    });

    it('handles empty entry list (valid empty zip)', async () => {
      const blob = buildZip([]);
      const arr = new Uint8Array(await blob.arrayBuffer());
      expect(findBytes(arr, PK_END)).toBeGreaterThan(-1);
    });
  });
});
