/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#include "cann_compat_library.h"
#include <elf.h>


static void cudaLibraryUpdateFileSize(uint64_t *file_size, uint64_t candidate)
{
    if (candidate > *file_size)
    {
        *file_size = candidate;
    }
}


static uint64_t cudaLibraryGetElf64Size(const uint8_t *buf)
{
    const Elf64_Ehdr *ehdr = (const Elf64_Ehdr *)buf;
    uint64_t fileSize = sizeof(Elf64_Ehdr);
    cudaLibraryUpdateFileSize(&fileSize, ehdr->e_phoff + (uint64_t)ehdr->e_phnum * ehdr->e_phentsize);
    for (int i = 0; i < ehdr->e_phnum; i++)
    {
        const Elf64_Phdr *phdr = (const Elf64_Phdr *)(buf + ehdr->e_phoff + i * ehdr->e_phentsize);
        cudaLibraryUpdateFileSize(&fileSize, phdr->p_offset + phdr->p_filesz);
    }
    if (ehdr->e_shoff == 0 || ehdr->e_shnum == 0)
    {
        return fileSize;
    }
    cudaLibraryUpdateFileSize(&fileSize, ehdr->e_shoff + (uint64_t)ehdr->e_shnum * ehdr->e_shentsize);
    for (int i = 0; i < ehdr->e_shnum; i++)
    {
        const Elf64_Shdr *shdr = (const Elf64_Shdr *)(buf + ehdr->e_shoff + i * ehdr->e_shentsize);
        if (shdr->sh_type != SHT_NOBITS)
        {
            cudaLibraryUpdateFileSize(&fileSize, shdr->sh_offset + shdr->sh_size);
        }
    }
    return fileSize;
}


static uint64_t cudaLibraryGetElf32Size(const uint8_t *buf)
{
    const Elf32_Ehdr *ehdr = (const Elf32_Ehdr *)buf;
    uint64_t fileSize = sizeof(Elf32_Ehdr);
    cudaLibraryUpdateFileSize(&fileSize, ehdr->e_phoff + (uint64_t)ehdr->e_phnum * ehdr->e_phentsize);
    for (int i = 0; i < ehdr->e_phnum; i++)
    {
        const Elf32_Phdr *phdr = (const Elf32_Phdr *)(buf + ehdr->e_phoff + i * ehdr->e_phentsize);
        cudaLibraryUpdateFileSize(&fileSize, (uint64_t)phdr->p_offset + phdr->p_filesz);
    }
    if (ehdr->e_shoff == 0 || ehdr->e_shnum == 0)
    {
        return fileSize;
    }
    cudaLibraryUpdateFileSize(&fileSize, ehdr->e_shoff + (uint64_t)ehdr->e_shnum * ehdr->e_shentsize);
    for (int i = 0; i < ehdr->e_shnum; i++)
    {
        const Elf32_Shdr *shdr = (const Elf32_Shdr *)(buf + ehdr->e_shoff + i * ehdr->e_shentsize);
        if (shdr->sh_type != SHT_NOBITS)
        {
            cudaLibraryUpdateFileSize(&fileSize, (uint64_t)shdr->sh_offset + shdr->sh_size);
        }
    }
    return fileSize;
}


static int cudaLibraryHasElfMagic(const uint8_t *buf)
{
    if (buf[EI_MAG0] != ELFMAG0)
    {
        return 0;
    }
    if (buf[EI_MAG1] != ELFMAG1)
    {
        return 0;
    }
    if (buf[EI_MAG2] != ELFMAG2)
    {
        return 0;
    }
    if (buf[EI_MAG3] != ELFMAG3)
    {
        return 0;
    }
    return 1;
}


static uint64_t cudaLibraryGetElfSizeByClass(const uint8_t *buf)
{
    uint8_t elf_class = buf[EI_CLASS];

    if (elf_class == ELFCLASS64)
    {
        return cudaLibraryGetElf64Size(buf);
    }
    if (elf_class == ELFCLASS32)
    {
        return cudaLibraryGetElf32Size(buf);
    }

    return 0;
}


uint64_t get_elf_file_size(const void *elf_buf)
{
    if (!elf_buf)
    {
        return 0;
    }

    const uint8_t *buf = (const uint8_t *)elf_buf;

    if (!cudaLibraryHasElfMagic(buf))
    {
        return 0;
    }

    return cudaLibraryGetElfSizeByClass(buf);
}

cudaError_t cudaLibraryLoadData(cudaLibrary_t *library,
                                const void *code,
                                cudaJitOption *jitOptions,
                                void **jitOptionsValues,
                                unsigned int numJitOptions,
                                cudaLibraryOption *libraryOptions,
                                void **libraryOptionValues,
                                unsigned int numLibraryOptions)
{
    cudaLibraryIgnoreOptions(jitOptions, jitOptionsValues, numJitOptions,
                             libraryOptions, libraryOptionValues, numLibraryOptions);

    if (!library || !code)
    {
        return cudaErrorInvalidValue;
    }

    /* Calculate ELF file size from memory buffer */
    uint64_t elf_size = get_elf_file_size(code);
    if (elf_size == 0)
    {
        return cudaErrorInvalidKernelImage;
    }

    /* Load binary from memory using calculated size */
    aclError ret = aclrtBinaryLoadFromData(code, (size_t)elf_size, NULL, library);
    return acl2cudaError(ret);
}
