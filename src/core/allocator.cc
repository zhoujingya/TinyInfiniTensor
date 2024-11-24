#include "core/allocator.h"
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
  used = 0;
  peak = 0;
  ptr = nullptr;

  // 'alignment' defaults to sizeof(uint64_t), because it is the length of
  // the longest data type currently supported by the DataType field of
  // the tensor
  alignment = sizeof(uint64_t);
}

Allocator::~Allocator() {
  if (this->ptr != nullptr) {
    runtime->dealloc(this->ptr);
  }
}

size_t Allocator::alloc(size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  // pad the size to the multiple of alignment
  size = this->getAlignedSize(size);

  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来分配内存，返回起始地址偏移量
  // =================================== 作业
  // ===================================
  // 1. 如果free_blocks为空，则直接分配内存
  if (free_blocks.empty()) {
    size_t offset = used;
    used += size;
    peak = std::max(peak, used);
    return offset;
  }

  // 2. 从free_blocks中找到一个合适的block
  for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
    if (it->second >= size) {
      // 找到大小足够的空闲块
      size_t offset = it->first;
      size_t blockSize = it->second;

      // 从free_blocks中移除这个块
      free_blocks.erase(it);

      // 如果剩余空间足够大,将其加入free_blocks
      if (blockSize > size) {
        free_blocks[offset + size] = blockSize - size;
      }

      used += size;
      peak = std::max(peak, used);
      return offset;
    }
  }

  // 3. 找不到合适的block,分配新内存
  size_t offset = used;
  used += size;
  peak = std::max(peak, used);
  return offset;
  if (free_blocks.empty()) {
    size_t offset = used;
    used += size;
    peak = std::max(peak, used);
    return offset;
  }
  // 2. 如果free_blocks不为空，则从free_blocks中找到一个合适的block进行分配
  for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
    // 找到第一个大小大于等于size的block
    // 判断这个地址的空间有没有被回收
    if (it->second >= size) {
      ptr = (char *)ptr + it->first;
      used += size;
      peak = std::max(peak, used);
      return it->first;
    }
  }

  return 0;
}

void Allocator::free(size_t addr, size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  size = getAlignedSize(size);

  // =================================== 作业
  // ===================================

  // =================================== 作业
  // ===================================
  // 1. 减少已使用的内存大小
  used -= size;

  // 2. 将要释放的内存块加入free_blocks
  free_blocks[addr] = size;

  // 3. 尝试合并相邻的空闲块
  auto it = free_blocks.find(addr);

  // 向前合并
  if (it != free_blocks.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second == addr) {
      // 前一个块的结尾地址等于当前块的起始地址,可以合并
      prev->second += it->second;
      free_blocks.erase(it);
      it = prev;
    }
  }

  // 向后合并
  auto next = std::next(it);
  if (next != free_blocks.end()) {
    if (it->first + it->second == next->first) {
      // 当前块的结尾地址等于后一个块的起始地址,可以合并
      it->second += next->second;
      free_blocks.erase(next);
    }
  }
}

void *Allocator::getPtr() {
  if (this->ptr == nullptr) {
    this->ptr = runtime->alloc(this->peak);
    printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
  }
  return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size) {
  return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() {
  std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak
            << std::endl;
}
} // namespace infini
