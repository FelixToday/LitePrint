import os
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import time


class BaseLogger:
    def __init__(self, json_save_path: str, log_save_path: str, log_to_console: bool = True):
        """
        初始化训练日志记录器

        参数:
            save_path: 日志文件保存路径(.json)
            log_to_console: 是否在控制台输出日志信息
        """
        self.json_save_path = json_save_path
        self.log_save_path = log_save_path
        self.time_buf = {}
        self.outputs: Dict[str, List[Any]] = {}
        self.outputs["metadata"] = {
            'create_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'update_at': None,
            'end_at': None
        }
        self.log_to_console = log_to_console

        # 配置日志格式
        self._configure_logging()

    def time_start(self, tag: str) -> None:
        """
        开始计时

        参数:
            tag: 计时标识符
        """
        self.time_buf[tag] = time.time()

    def time_end(self, tag: str) -> float:
        """
        结束计时并返回耗时(秒)

        参数:
            tag: 计时标识符

        返回:
            耗时(秒)
        """
        if tag not in self.time_buf:
            print(f"未找到计时标签: {tag}")

        duration = time.time() - self.time_buf[tag]
        del self.time_buf[tag]

        # 自动记录耗时到日志
        self.add_field(f"timing.{tag}", duration)
        self.info(f"{tag}:{duration:.2f} s")
        return duration

    def _configure_logging(self):
        """配置日志格式"""
        self.log_format = '%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S,%f'

    def info(self, message: str, *args, is_logfile: bool = False, log_to_console: bool = True, **kwargs):
        """
        记录信息级别日志

        参数:
            message: 要记录的信息(可以使用%s等格式化占位符)
            *args: 格式化参数
            log_to_file: 是否写入日志
            **kwargs:
                - extra: 额外信息字典
        """
        # 获取调用信息
        frame = sys._getframe(1)
        filename = os.path.basename(frame.f_code.co_filename)
        func_name = frame.f_code.co_name
        lineno = frame.f_lineno

        # 格式化消息
        formatted_msg = message % args if args else message

        # 添加额外信息
        extra = kwargs.get('extra', {})
        if extra:
            formatted_msg += " " + " ".join(f"{k}={v}" for k, v in extra.items())

        # 格式化日志行
        log_entry = self.log_format % {
            'asctime': datetime.now().strftime(self.date_format)[:-3],  # 去掉最后3位微秒
            'filename': filename,
            'funcName': func_name,
            'lineno': lineno,
            'levelname': 'INFO',
            'message': formatted_msg
        }

        # 控制台输出
        if self.log_to_console and log_to_console:
            print(log_entry)

        # 文件输出
        if is_logfile:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(os.path.abspath(self.log_save_path)), exist_ok=True)
                mode = 'a' if os.path.exists(self.log_save_path) else 'w'
                with open(self.log_save_path, mode, encoding='utf-8') as f:
                    f.write(log_entry + '\n')
            except Exception as e:
                print(f"写入日志文件失败: {e}", file=sys.stderr)

    def add_field(self, field_name: str, value: Optional[Any] = None, replace: bool = False) -> None:
        """
        添加或更新日志字段

        参数:
            field_name: 字段名称(支持点号分隔的嵌套字段，如'metrics.accuracy')
            value: 要记录的值(可选)
        """
        keys = field_name.split('.')

        local_outputs=self.outputs
        for key in keys[:-1]:
            if key not in local_outputs:
                local_outputs[key] = {}
            local_outputs=local_outputs[key]
        final_key = keys[-1]
        if final_key not in local_outputs or replace:
            local_outputs[final_key] = value
        else:
            if isinstance(local_outputs[final_key], list):
                local_outputs[final_key].append(value)
            else:
                local_outputs[final_key] = [local_outputs[final_key], value]

    def clear_field(self, field_name: str) -> None:
        """
        清除指定字段(支持点号分隔的嵌套字段，如'metrics.accuracy')

        参数:
            field_name: 要清除的字段名称
        """

        keys = field_name.split('.')
        assert keys[0] != 'metadata', '不能删除 metadata'
        current = self.outputs

        try:
            # 遍历到倒数第二个key
            for key in keys[:-1]:
                current = current[key]

            # 删除最后一个key
            final_key = keys[-1]
            if final_key in current:
                del current[final_key]
        except (KeyError, TypeError):
            # 如果路径不存在，则忽略
            pass

    def log(self, field_name: str, value: Any, replace: bool = False) -> None:
        """
        记录值到指定字段

        参数:
            field_name: 字段名称
            value: 要记录的值
        """
        self.add_field(field_name, value, replace)
        self.outputs['metadata']['update_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_to_json()

    def save_to_json(self) -> None:
        """
        保存日志到JSON文件(包含元数据)
        不使用缩进，生成紧凑的JSON格式
        """
        os.makedirs(os.path.dirname(os.path.abspath(self.json_save_path)), exist_ok=True)
        self.outputs['metadata']["end_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration_seconds = (
                datetime.strptime(self.outputs['metadata']["end_at"], '%Y-%m-%d %H:%M:%S') -
                datetime.strptime(self.outputs['metadata']["create_at"], '%Y-%m-%d %H:%M:%S')
        ).total_seconds()

        # 转换为 时:分:秒 格式
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        self.outputs['metadata']["duration"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        with open(self.json_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.outputs, f, indent=4, ensure_ascii=False, separators=(',', ':'))

    def load_from_json(self) -> None:
        """
        从JSON文件加载日志数据
        """
        if os.path.exists(self.json_save_path):
            with open(self.json_save_path, 'r', encoding='utf-8') as f:
                self.outputs = json.load(f)


    def print_logs(self) -> None:
        """
        以美观格式打印当前日志内容
        """
        print(json.dumps(self.outputs, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    logger = BaseLogger("../cache/test.json", "../cache/test.txt")

    # 基本用法
    logger.info("这是一个信息日志")

    # 带格式化的用法
    logger.info("训练进度: epoch=%d/%d, loss=%.4f", 10, 100, 0.1234)

    # 带额外信息的用法
    logger.info("模型评估结果", extra={"accuracy": 0.95, "f1": 0.92})

    logger.info("这条日志会写入文件", is_logfile=False)

    # 耗时统计
    logger.time_start("训练流程")

    # 信息日志
    logger.info("实验配置", extra={"batch_size": 32, "epochs": 100}, is_logfile=True)

    # 结构化记录
    logger.log("config", {"lr": 0.01, "optimizer": "Adam"})

    for epoch in range(100):
        #logger.add_field(f"epochs.{epoch}.loss", 0.1 / (epoch + 1))
        logger.log(f"epochs.loss", 0.1 / (epoch + 1))
        logger.info("进度: %d%%", (epoch + 1), is_logfile=True)

    # 结束统计
    logger.time_end("训练流程")
    logger.save_to_json()
