#!/bin/bash
set -e

REGISTRY=10.12.11.7:16622
REPO_PATH=/mnt/data/wjh/sglang-taas
CACHE_TARBALL=/mnt/data/wjh/image-cache/root_cache.tar.gz
INITIAL_BASE=$REGISTRY/sglang:v0.5.10_glm51_0415_cudaMemCopy_content_list

# 检测 kernel 是否变动
if git -C $REPO_PATH diff HEAD~1 --name-only | grep -q "^sgl-kernel/"; then
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

# 启动临时容器
CID=$(docker run -d $BASE_IMAGE sleep infinity)
trap "docker stop $CID 2>/dev/null || true" EXIT

# 替换代码
docker exec $CID rm -rf /sgl-workspace/sglang
docker cp $REPO_PATH/. $CID:/sgl-workspace/sglang

# 重编 kernel（慢速路径）
if [ "$REBUILD_KERNEL" = true ]; then
  echo "[CI] Recompiling sgl-kernel..."
  docker exec $CID bash -c "
    cd /sgl-workspace/sglang/sgl-kernel &&
    TORCH_CUDA_ARCH_LIST='10.0' pip install -e . --no-build-isolation
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
