// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_guard.hpp"

#include "ie_common.h"

namespace ov {

CacheGuardEntry::CacheGuardEntry(CacheGuard& cacheGuard,
                                 const std::string& hash,
                                 std::shared_ptr<std::mutex> m,
                                 std::atomic_int& refCount)
    : m_cacheGuard(cacheGuard),
      m_hash(hash),
      m_mutex(m),
      m_refCount(refCount) {
    // Don't lock mutex right here for exception-safe considerations
    m_refCount++;
}

CacheGuardEntry::~CacheGuardEntry() {
    m_refCount--;
    m_mutex->unlock();
    m_cacheGuard.check_for_remove(m_hash);
}

void CacheGuardEntry::perform_lock() {
    m_mutex->lock();
}

//////////////////////////////////////////////////////

std::unique_ptr<CacheGuardEntry> CacheGuard::get_hash_lock(const std::string& hash) {
    std::unique_lock<std::mutex> lock(m_tableMutex);
    auto& data = m_table[hash];
    std::unique_ptr<CacheGuardEntry> res;
        // TODO: use std::make_unique when migrated to C++14
        res =
            std::unique_ptr<CacheGuardEntry>(new CacheGuardEntry(*this, hash, data.m_mutexPtr, data.m_itemRefCounter));
    lock.unlock();        // can unlock table lock here, as refCounter is positive and nobody can remove entry
    res->perform_lock();  // in case of exception, 'res' will be destroyed and item will be cleaned up from table
    return res;
}

void CacheGuard::check_for_remove(const std::string& hash) {
    std::lock_guard<std::mutex> lock(m_tableMutex);
    if (m_table.count(hash)) {
        auto& data = m_table[hash];
        if (data.m_itemRefCounter == 0) {
            // Nobody is using this and nobody is waiting for it - can be removed
            m_table.erase(hash);
        }
    }
}

}  // namespace ov
