def print_colored(text, color="", is_print=True):
    """
    输出带颜色的文字，并返回该字符串（带颜色）
    """
    colors = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    if color == "" or color.lower() not in colors:
        str_info = text
    else:
        color_code = colors.get(color.lower(), "37")  # 默认白色
        str_info = f"\033[{color_code}m{text}\033[0m"
    if is_print:
        print(str_info)
    return str_info

def print_title(text, color="red", is_print=True):
    print_colored("="*20+" "+text+" "+"="*20, color, is_print)

def sort_lists(*lists):
    """
    将输入的列表按照第一个列表进行排序
    """
    # 获取第一个列表
    first_list = lists[0]
    # 对第一个列表的索引进行排序
    sorted_indices = sorted(range(len(first_list)), key=lambda i: first_list[i])
    # 根据排序后的索引重新排序所有列表
    sorted_lists = [[lst[i] for i in sorted_indices] for lst in lists]
    return sorted_lists

def str_to_bool(value):
    """
    将字符串值转换为布尔值。

    该函数接受一个字符串或布尔值输入，并将其转换为对应的布尔值。
    如果输入已经是布尔值，则直接返回。
    对于字符串输入，不区分大小写，支持多种常见布尔表示形式。

    Args:
        value: 待转换的值，可以是布尔值或字符串。
            如果是字符串，支持的真值包括：'yes', 'true', 't', 'y', '1'
            支持的假值包括：'no', 'false', 'f', 'n', '0'

    Returns:
        bool: 转换后的布尔值

    Raises:
        argparse.ArgumentTypeError: 当输入无法识别为有效的布尔值时抛出异常

    Examples:
        >>> str_to_bool('True')
        True
        >>> str_to_bool('0')
        False
        >>> str_to_bool(True)
        True

    Note:
        - 字符串比较不区分大小写
        - 如果输入值不在支持的范围内，会抛出ArgumentTypeError异常
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')
def same_seed(fix_seed):
    """
    设置随机种子
    """
    import torch.backends.cudnn as cudnn
    import random
    import torch
    import numpy as np

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    rng = np.random.RandomState(fix_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def print_dict(d):
    import json
    print(json.dumps(d, indent=4, ensure_ascii=False))
if __name__ == '__main__':
    print_title("你好")