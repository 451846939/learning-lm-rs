use std::{usize, vec};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::tensor::Tensor;
pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    length: usize, // length of the current sequence
}

impl<T: Default + Copy> KVCache<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len: max_seq_len,
            dim: dim,
            length: init_len,
        }
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn increment(&mut self, seq_len: usize){
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }
}



/// **多用户 KVCache 管理**
pub struct KVCacheManager<T> {
    cache_map: RwLock<HashMap<String, Arc<RwLock<KVCache<T>>>>>,  // ✅ 允许可变访问
    n_layers: usize,
    max_seq_len: usize,
    dim: usize,
}

impl<T: Default + Copy + Send + Sync + 'static> KVCacheManager<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize) -> Arc<Self> {
        Arc::new(KVCacheManager {
            cache_map: RwLock::new(HashMap::new()),
            n_layers,
            max_seq_len,
            dim,
        })
    }

    /// **获取用户的 KVCache**
    pub async fn get_cache_for_user(&self, user_id: &str) -> Arc<RwLock<KVCache<T>>> {
        let mut cache = self.cache_map.write().await;
        cache
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(RwLock::new(KVCache::new(self.n_layers, self.max_seq_len, self.dim, 0))))
            .clone()  // ✅ 现在 `.clone()` 没问题
    }

    /// **存储用户 KVCache**（可以直接更新 `RwLock<KVCache<T>>`，不需要存回 `HashMap`）
    pub async fn store_cache_for_user(&self, _user_id: &str, _cache: Arc<RwLock<KVCache<T>>>) {
        // 这里不需要额外存储，因为 `Arc<RwLock<KVCache<T>>>` 内部已经更新
    }
}