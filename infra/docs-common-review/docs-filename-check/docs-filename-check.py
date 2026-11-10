#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CANN-2.0
"""
Markdown文件名称质量检查python脚本
用于检查Markdown文档名称的规范性和正确性
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


class MarkdownFilenameChecker:
    """Markdown文件名称检查器"""
    
    def __init__(self, input_path: str):
        """
        初始化检查器
        
        Args:
            input_path: 输入路径，可以是文件夹或markdown文档的绝对路径
        """
        self.input_path = input_path
        self.result: List[Dict[str, Any]] = []
        self.rule_info = {
            "rule_no": "1",
            "rule_item_name": "filename-check",
            "rule_name": "Filename Check",
            "rule_type": "规范性",
            "rule_level": "提示"
        }
    
    def is_markdown_file(self, file_path: str) -> bool:
        """
        检查是否为markdown文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否为markdown文件
        """
        return file_path.lower().endswith('.md')
    
    def get_filename_without_extension(self, file_path: str) -> str:
        """
        获取不带扩展名的文件名
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 不带扩展名的文件名
        """
        filename = os.path.basename(file_path)
        if filename.lower().endswith('.md'):
            return filename[:-3]
        return filename
    
    def check_chinese_characters(self, filename_without_ext: str, full_path: str) -> Optional[Dict[str, Any]]:
        """
        检查文件名是否包含中文字符
        
        Args:
            filename_without_ext: 文件名（不含扩展名）
            full_path: 文件完整路径
            
        Returns:
            Optional[Dict]: 如果发现问题，返回问题字典；否则返回None
        """
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        chinese_chars = chinese_pattern.findall(filename_without_ext)
        
        if chinese_chars:
            chinese_text = ''.join(chinese_chars)
            basename = filename_without_ext + '.md'
            location_path = full_path.replace('\\', '/')
            
            return {
                "rule_group": "通用规则",
                "rule_item": self.rule_info["rule_item_name"],
                "rule_type": self.rule_info["rule_type"],
                "rule_name": self.rule_info["rule_name"],
                "snippet": basename,
                "description": f"文件名称中包含\"{chinese_text}\"中文字符",
                "rule_level": self.rule_info["rule_level"],
                "file_name": basename,
                "line_number": "",
                "location": f"vscode://file/{location_path}",
                "suggestion": "建议将文件名称修改为纯英文格式"
            }
        return None
    
    def check_special_characters(self, filename_without_ext: str, full_path: str) -> Optional[Dict[str, Any]]:
        """
        检查文件名是否包含不允许的特殊字符
        
        Args:
            filename_without_ext: 文件名（不含扩展名）
            full_path: 文件完整路径
            
        Returns:
            Optional[Dict]: 如果发现问题，返回问题字典；否则返回None
        """
        allowed_pattern = re.compile(r'^[a-zA-Z0-9\-_&]+$')
        
        if not allowed_pattern.match(filename_without_ext):
            invalid_chars = []
            for char in filename_without_ext:
                if not re.match(r'[a-zA-Z0-9\-_&]', char):
                    if not re.match(r'[\u4e00-\u9fff]', char):
                        invalid_chars.append(char)
            
            if invalid_chars:
                invalid_text = ''.join(invalid_chars)
                basename = filename_without_ext + '.md'
                location_path = full_path.replace('\\', '/')
                
                return {
                    "rule_group": "通用规则",
                    "rule_item": self.rule_info["rule_item_name"],
                    "rule_type": self.rule_info["rule_type"],
                    "rule_name": self.rule_info["rule_name"],
                    "snippet": basename,
                    "description": f"文件名称中包含不允许的特殊字符\"{invalid_text}\"",
                    "rule_level": self.rule_info["rule_level"],
                    "file_name": basename,
                    "line_number": "",
                    "location": f"vscode://file/{location_path}",
                    "suggestion": "建议将文件名称中的特殊字符替换为英文字母、数字、英文中横线（-）、英文下划线（_）或&符号"
                }
        return None
    
    def check_filename_length(self, filename_without_ext: str, full_path: str) -> Optional[Dict[str, Any]]:
        """
        检查文件名长度是否超过70个ASCII字符
        
        Args:
            filename_without_ext: 文件名（不含扩展名）
            full_path: 文件完整路径
            
        Returns:
            Optional[Dict]: 如果发现问题，返回问题字典；否则返回None
        """
        ascii_count = sum(1 for char in filename_without_ext if ord(char) < 128)
        
        if ascii_count > 70:
            basename = filename_without_ext + '.md'
            location_path = full_path.replace('\\', '/')
            
            return {
                "rule_group": "通用规则",
                "rule_item": self.rule_info["rule_item_name"],
                "rule_type": self.rule_info["rule_type"],
                "rule_name": self.rule_info["rule_name"],
                "snippet": basename,
                "description": f"文件名称超过了70个ASCII字母/数字/符号，当前长度为{ascii_count}",
                "rule_level": self.rule_info["rule_level"],
                "file_name": basename,
                "line_number": "",
                "location": f"vscode://file/{location_path}",
                "suggestion": "建议缩短文件名称，使其不超过70个ASCII字母/数字/符号"
            }
        return None
    
    def check_duplicate_filenames(self, directory: str) -> List[Dict[str, Any]]:
        """
        检查目录下是否存在重名的markdown文件
        
        Args:
            directory: 目录路径
            
        Returns:
            List[Dict]: 问题列表
        """
        issues = []
        
        md_files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and self.is_markdown_file(file_path):
                filename_without_ext = self.get_filename_without_extension(file_path)
                md_files.append({
                    "full_path": file_path,
                    "filename": filename_without_ext,
                    "basename": file
                })
        
        filename_count = {}
        for md_file in md_files:
            filename = md_file["filename"]
            if filename in filename_count:
                filename_count[filename].append(md_file)
            else:
                filename_count[filename] = [md_file]
        
        for filename, files in filename_count.items():
            if len(files) > 1:
                for file_info in files:
                    full_path = file_info["full_path"]
                    basename = file_info["basename"]
                    location_path = full_path.replace('\\', '/')
                    
                    issues.append({
                        "rule_group": "通用规则",
                        "rule_item": self.rule_info["rule_item_name"],
                        "rule_type": self.rule_info["rule_type"],
                        "rule_name": self.rule_info["rule_name"],
                        "snippet": basename,
                        "description": f"目录下存在重名文件\"{filename}\"",
                        "rule_level": self.rule_info["rule_level"],
                        "file_name": basename,
                        "line_number": "",
                        "location": f"vscode://file/{location_path}",
                        "suggestion": "建议修改文件名称，确保同一目录下不存在重名的markdown文件"
                    })
        
        return issues
    
    def check_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        检查单个markdown文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Dict]: 问题列表
        """
        issues = []
        
        if not self.is_markdown_file(file_path):
            return issues
        
        filename_without_ext = self.get_filename_without_extension(file_path)
        
        chinese_issue = self.check_chinese_characters(filename_without_ext, file_path)
        if chinese_issue:
            issues.append(chinese_issue)
        
        if not chinese_issue:
            special_issue = self.check_special_characters(filename_without_ext, file_path)
            if special_issue:
                issues.append(special_issue)
        
        length_issue = self.check_filename_length(filename_without_ext, file_path)
        if length_issue:
            issues.append(length_issue)
        
        directory = os.path.dirname(file_path)
        duplicate_issues = self.check_duplicate_filenames(directory)
        for issue in duplicate_issues:
            if issue["file_name"] == filename_without_ext + '.md':
                issues.append(issue)
        
        return issues
    
    def check_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        检查目录下的所有markdown文件
        
        Args:
            directory: 目录路径
            
        Returns:
            List[Dict]: 问题列表
        """
        issues = []
        
        md_files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and self.is_markdown_file(file_path):
                md_files.append(file_path)
        
        for md_file in md_files:
            file_issues = self.check_single_file(md_file)
            issues.extend(file_issues)
        
        duplicate_issues = self.check_duplicate_filenames(directory)
        existing_files = set()
        for issue in issues:
            existing_files.add(issue["file_name"])
        
        for duplicate_issue in duplicate_issues:
            if duplicate_issue["file_name"] not in existing_files:
                issues.append(duplicate_issue)
        
        return issues
    
    def run(self) -> List[Dict[str, Any]]:
        """
        执行检查
        
        Returns:
            List[Dict]: 检查结果列表
        """
        if os.path.isfile(self.input_path):
            self.result = self.check_single_file(self.input_path)
        elif os.path.isdir(self.input_path):
            self.result = self.check_directory(self.input_path)
        else:
            print(f"错误：输入路径不存在 - {self.input_path}")
            return []
        
        return self.result
    
    def save_result(self, output_path: Optional[str] = None):
        """
        保存检查结果
        
        Args:
            output_path: 输出文件路径，默认为output/filename-markdown-check-result.json
        """
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            output_path = os.path.join(project_root, "output", "filename-markdown-check-result.json")
        
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.result, f, ensure_ascii=False, indent=2)
        
        print(f"检查结果已保存到: {output_path}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python markdown_filename_checker.py <输入路径> [输出路径]")
        print("输入路径可以是:")
        print("  - Markdown文档的绝对路径")
        print("  - 包含Markdown文档的文件夹绝对路径")
        print("输出路径（可选）:")
        print("  - 检查结果存放的绝对路径（文件夹）")
        print("  - 默认为项目根目录下的output文件夹")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        output_dir = os.path.join(project_root, "output")
    
    result_file_path = os.path.join(output_dir, "filename-markdown-check-result.json")
    
    if os.path.exists(result_file_path):
        os.remove(result_file_path)
        print(f"已删除旧的检查结果文件: {result_file_path}")
    
    checker = MarkdownFilenameChecker(input_path)
    
    result = checker.run()
    
    if result:
        print(f"发现 {len(result)} 个问题:")
        for issue in result:
            print(f"  - {issue['file_name']}: {issue['description']}")
    else:
        print("未发现问题")
    
    checker.save_result(result_file_path)


if __name__ == "__main__":
    main()