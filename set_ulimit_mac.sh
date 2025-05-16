#!/bin/bash

# 检测当前 Shell 类型
SHELL_NAME=$(basename "$SHELL")

# 映射配置文件
if [ "$SHELL_NAME" = "zsh" ]; then
    CONFIG_FILE="$HOME/.zshrc"
elif [ "$SHELL_NAME" = "bash" ]; then
    CONFIG_FILE="$HOME/.bash_profile"
else
    echo "⚠️ 未知的 Shell 类型：$SHELL_NAME"
    echo "请手动将 'ulimit -n 4096' 添加到你的 shell 配置文件中。"
    exit 1
fi

# 备份配置文件
BACKUP_FILE="${CONFIG_FILE}.bak_$(date +%Y%m%d_%H%M%S)"
cp "$CONFIG_FILE" "$BACKUP_FILE"
echo "📝 已备份原始配置文件到: $BACKUP_FILE"

# 检查是否已设置 ulimit
if grep -q "ulimit -n 4096" "$CONFIG_FILE"; then
    echo "✅ ulimit 已经设置，无需重复添加。"
else
    echo -e "\n# 设置最大打开文件数限制\nulimit -n 4096" >> "$CONFIG_FILE"
    echo "✅ 已添加 'ulimit -n 4096' 到 $CONFIG_FILE"
fi

# 提示下一步
echo "🔄 请运行以下命令使设置生效，或重启终端："
echo "    source $CONFIG_FILE"
