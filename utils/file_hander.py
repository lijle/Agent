import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def get_file_md5_hex(filepath: str):    #读取一个文件，计算它的 MD5 哈希值，并返回 32 位十六进制字符串

    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件不存在: {filepath}")
        return

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]不是文件: {filepath}")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096    # 4k分片，避免文件过大爆内存

    try:
        with open(filepath, 'rb') as f:
            while chunk:= f.read(chunk_size):
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:
                 md5_obj.update(chunk)
                 chunk = f.read(chunk_size)
            """
            # hexdigest() 会把 MD5 结果转成十六进制字符串。
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[md5计算]文件{filepath}计算失败: {str( e)}")
        return None

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):    #返回文件夹内的文件列表(允许的文件)
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]路径不是文件夹: {path}")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)

def pdf_loader(filepath: str, password=None) -> list[Document]:
    return PyPDFLoader(filepath, password).load()

def txt_loader(filepath: str) -> list[Document]:
    encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk")

    for encoding in encodings:
        try:
            return TextLoader(filepath, encoding=encoding).load()
        except UnicodeDecodeError:
            continue
        except RuntimeError as e:
            if "Error loading" in str(e):
                continue
            raise

    raise RuntimeError(f"无法识别文本文件编码: {filepath}")
