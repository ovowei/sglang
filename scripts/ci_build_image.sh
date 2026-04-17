#!/bin/bash
set -e

REGISTRY=10.12.11.7:16622
REPO_PATH=/mnt/data/wjh/sglang-taas
CACHE_TARBALL=/mnt/data/wjh/image-cache/root_cache.tar.gz
INITIAL_BASE=$REGISTRY/sglang:v0.5.10_glm51_0415_cudaMemCopy_content_list
FETCHCONTENT_CACHE=/mnt/afs/sglang-taas/fetchcontent-cache
FLASHMLA_DIR=/mnt/afs/sglang-taas/3rdparty/FlashMLA
FLASHMLA_COMMIT=e1c0028077c83bf9561ce2b1eafdb8165a596351

# 检测 kernel 是否变动
if git -C $REPO_PATH diff --name-only HEAD~1 HEAD -- sgl-kernel/ | grep -q .; then
  REBUILD_KERNEL=true
  if docker pull $REGISTRY/sglang:kernel-base 2>/dev/null; then
    BASE_IMAGE=$REGISTRY/sglang:kernel-base
  else
    BASE_IMAGE=$INITIAL_BASE
  fi
  echo "[CI] Kernel changed, slow path: base=$BASE_IMAGE"
else
  REBUILD_KERNEL=false
  if docker pull $REGISTRY/sglang:dev 2>/dev/null; then
    BASE_IMAGE=$REGISTRY/sglang:dev
  else
    BASE_IMAGE=$INITIAL_BASE
  fi
  echo "[CI] Python only, fast path: base=$BASE_IMAGE"
fi

# 确保 FetchContent cache 目录存在
mkdir -p $FETCHCONTENT_CACHE

# 确保 FlashMLA 本地 clone 存在且在正确 commit
if [ ! -d "$FLASHMLA_DIR/.git" ]; then
  echo "[CI] Cloning FlashMLA..."
  mkdir -p $(dirname $FLASHMLA_DIR)
  https_proxy=http://127.0.0.1:21683 git clone https://github.com/deepseek-ai/FlashMLA.git $FLASHMLA_DIR
fi
CURRENT=$(git -C $FLASHMLA_DIR rev-parse HEAD 2>/dev/null || echo "none")
if [ "$CURRENT" != "$FLASHMLA_COMMIT" ]; then
  echo "[CI] Updating FlashMLA to $FLASHMLA_COMMIT..."
  https_proxy=http://127.0.0.1:21683 git -C $FLASHMLA_DIR fetch origin
  git -C $FLASHMLA_DIR checkout $FLASHMLA_COMMIT
fi
echo "[CI] FlashMLA at $(git -C $FLASHMLA_DIR rev-parse --short HEAD)"

# 启动临时容器，挂载 FetchContent cache 和 FlashMLA
CID=$(docker run -d --network host --entrypoint "" \
  -v $FETCHCONTENT_CACHE:/mnt/afs/fetchcontent-cache \
  -v $FLASHMLA_DIR:/sgl-workspace/sglang/3rdparty/FlashMLA:ro \
  $BASE_IMAGE sleep infinity)
trap "docker stop $CID 2>/dev/null || true" EXIT

# 替换代码
docker exec $CID rm -rf /sgl-workspace/sglang
docker cp $REPO_PATH/. $CID:/sgl-workspace/sglang

# 重编 kernel（慢速路径）
if [ "$REBUILD_KERNEL" = true ]; then
  echo "[CI] Recompiling sgl-kernel..."
  docker exec $CID bash -c "
    export https_proxy=http://127.0.0.1:21683
    export http_proxy=http://127.0.0.1:21683
    cd /sgl-workspace/sglang/sgl-kernel &&
    TORCH_CUDA_ARCH_LIST='10.0' \
    make build \
      MAX_JOBS=2 \
      CMAKE_BUILD_PARALLEL_LEVEL=2 \
      'CMAKE_ARGS=-DSGL_KERNEL_COMPILE_THREADS=1 -DFETCHCONTENT_BASE_DIR=/mnt/afs/fetchcontent-cache'
  "
fi

# 可选：注入 cache
if [ -f "$CACHE_TARBALL" ]; then
  echo "[CI] Injecting kernel cache..."
  docker cp $CACHE_TARBALL $CID:/tmp/root_cache.tar.gz
  docker exec $CID tar xzf /tmp/root_cache.tar.gz -C /root
else
  echo "[CI] No cache found, skipping"
fi

# Commit & push
COMMIT_SHA=$(git -C $REPO_PATH rev-parse --short HEAD)
docker commit $CID $REGISTRY/sglang:dev
docker commit $CID $REGISTRY/sglang:dev-$COMMIT_SHA
docker push $REGISTRY/sglang:dev
docker push $REGISTRY/sglang:dev-$COMMIT_SHA

if [ "$REBUILD_KERNEL" = true ]; then
  docker commit $CID $REGISTRY/sglang:kernel-base
  docker push $REGISTRY/sglang:kernel-base
fi

echo "[CI] Done: $REGISTRY/sglang:dev-$COMMIT_SHA"
