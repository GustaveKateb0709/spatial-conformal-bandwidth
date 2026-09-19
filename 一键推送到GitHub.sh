#!/bin/bash
# 把本目录内容推送到 GitHub 空仓库。需要能访问 github.com（若走代理请确保代理可用）。
set -e
REPO="https://github.com/GustaveKateb0709/spatial-conformal-bandwidth.git"
cd "$(dirname "$0")"
rm -rf .git
git init -b main
git add -A
git -c user.name="Tian-Yu Wang" -c user.email="wangty0709@hotmail.com" \
    commit -m "Analysis pipeline for spatially weighted conformal prediction"
git remote add origin "$REPO"
git push -u origin main
echo "完成。仓库地址: https://github.com/GustaveKateb0709/spatial-conformal-bandwidth"
