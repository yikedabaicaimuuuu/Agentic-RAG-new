#!/bin/bash

echo "🚀 开始自动清理和推送流程"

# 检查是否在 Git 仓库
if [ ! -d ".git" ]; then
    echo "❌ 当前目录不是 Git 仓库"
    exit 1
fi

git status

read -p "⚠ 确认你的代码和修改已提交（否则会丢失）。是否继续？(y/n): " choice
if [[ "$choice" != "y" ]]; then
    echo "退出。"
    exit 0
fi

# 4. 清理历史中不需要的目录和文件
echo "🧹 使用 git filter-repo 清理历史中不需要的路径..."
git filter-repo --path arag --path vectorstore-hotpot --path vectorstore
--path data --path data-hotpot --invert-paths

# 5. 更新 .gitignore 防止将来误加
echo "🛡 更新 .gitignore..."
cat <<EOL > .gitignore
arag/
vectorstore/
vectorstore-hotpot/
data/
data-hotpot/
__pycache__/
.DS_Store
*.faiss
*.dylib
*.pt
*.h5
EOL

git add .gitignore
git commit -m "Add clean .gitignore to prevent future accidental adds"

# 6. 强制推送
echo "🚀 强制推送到 origin main"
git push -u origin main --force

# 7. 查看仓库大小
echo "✅ 当前 .git 目录大小："
du -sh .git

echo "🎉 自动清理、重写历史、推送完成！"

