FROM nvcr.io/nvidia/pytorch:24.03-py3

# 等号の前後にスペースは不要です
ARG PROJECT_NAME=CenterNet_pytorch

# システム更新とクリーンアップ
RUN apt update && \
    apt upgrade -y && \
    apt-get clean

# 必要なディレクトリの作成
RUN mkdir -p /workspaces

ARG USERNAME=miyata
ARG GROUPNAME=miyata
ARG UID=1000
ARG GID=1000

# ユーザーとグループの作成、そしてディレクトリの所有権を設定
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME && \
    chown -R $USERNAME:$GROUPNAME /workspaces

# プロジェクトのクローン
RUN git clone https://github.com/MiyataYuya/CenterNet_Pytorch.git /workspaces/${PROJECT_NAME}

# 作業ディレクトリの設定
WORKDIR /workspaces/${PROJECT_NAME}

# 環境変数の設定、パスの結合には引数を使用
ENV PYTHONPATH="${PYTHONPATH}:/workspaces/${PROJECT_NAME}/"

# ユーザーの切り替え
USER $USERNAME
