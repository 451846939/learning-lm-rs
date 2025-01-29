use core::f32;
use num_traits::{Float, FromPrimitive, One, Zero};
use safetensors::SafeTensors;
use std::fs::File;
use std::iter::Sum;
use std::path::Path;
use std::vec;
use num_traits::real::Real;
use crate::config::{LlamaConfigJson, TorchDType};
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::{FromLeBytes, LLamaParams};
use crate::tensor::Tensor;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    pub(crate) n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    pub(crate) n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    pub(crate) dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    pub(crate) max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    pub(crate) eos_token_id: u32,      // end token id
}

pub fn read_config(model_dir: impl AsRef<Path>) -> LlamaConfigJson {
    let config_file = File::open(model_dir.as_ref().join("config.json")).unwrap();
    serde_json::from_reader(config_file).unwrap()
}

impl<T: Default + Copy + Sum + Float + FromPrimitive + FromLeBytes > Llama<T>  {

    pub fn from_safetensors(model_dir: impl AsRef<Path>,config: LlamaConfigJson) -> Self {
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);


        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                T::from_f32(self.eps).unwrap(),
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(
                q,
                T::zero(),
                &hidden_states,
                &self.params.wq[layer],
                T::one(),
            );
            OP::matmul_transb(
                k,
                T::zero(),
                &hidden_states,
                &self.params.wk[layer],
                T::one(),
            );
            OP::matmul_transb(
                v,
                T::zero(),
                &hidden_states,
                &self.params.wv[layer],
                T::one(),
            );
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                T::from_f32(self.rope_theta).unwrap(),
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                T::from_f32(self.rope_theta).unwrap(),
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention_multihead(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            OP::matmul_transb(
                &mut residual,
                T::one(),
                &hidden_states,
                &self.params.wo[layer],
                T::one(),
            );

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                T::from_f32(self.eps).unwrap(),
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<T>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![ self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            T::from_f32(self.eps).unwrap(),
        );

        OP::matmul_transb(
            &mut logits,
            T::zero(),
            &hidden_states,
            &self.params.lm_head,
            T::one(),
        );

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: T,
        top_k: u32,
        temperature: T,
    ) -> Vec<u32> {
        let mut result_tokens = token_ids.to_vec();
        let mut kvcache = self.new_cache();
        let mut input_tensors =
            Tensor::<u32>::new(result_tokens.clone(), &vec![result_tokens.len()]);

        while result_tokens.len() < max_len {
            let logits = self.forward(&input_tensors, &mut kvcache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result_tokens.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            input_tensors = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }

        result_tokens
    }

    pub fn streaming_generate<'a>(
        &'a self,
        token_ids: &[u32],
        max_len: usize,
        top_p: T,
        top_k: u32,
        temperature: T,
        kvcache: &'a mut KVCache<T>,
    ) -> impl Iterator<Item = u32> + 'a {
        let mut result_tokens = token_ids.to_vec();
        let mut input_tensors =
            Tensor::<u32>::new(result_tokens.clone(), &vec![result_tokens.len()]);

        std::iter::from_fn(move || {
            if result_tokens.len() >= max_len {
                return None;
            }

            let logits = self.forward(&input_tensors, kvcache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result_tokens.push(next_token);
            input_tensors = Tensor::<u32>::new(vec![next_token], &vec![1]);

            if next_token == self.eos_token_id {
                None
            } else {
                Some(next_token)
            }
        })
    }
}

/// 计算自注意力 (Q*K^T / sqrt(d) => softmax => x V)，指针式实现。
///
/// # 参数
/// - `hidden_states`: [seq_len, n_kv_h*n_groups*dqkv] (最终输出, 大小 = seq_len * n_q_h * dqkv)，其中 n_q_h = n_kv_h*n_groups
/// - `att_scores`:    [n_kv_h, n_groups, seq_len, total_seq_len] (4D，展平在 data 里，供中间结果存储)
/// - `q`:             [seq_len, n_q_h*dqkv]  (Q buffer)
/// - `k`:             [total_seq_len, n_kv_h*dqkv] (K buffer)
/// - `v`:             [total_seq_len, n_kv_h*dqkv] (V buffer)
/// - `n_kv_h`: Key/Value 头数
/// - `n_groups`:  Q 相较 KV 多出的倍数，所以总 Query 头数 n_q_h = n_kv_h * n_groups
/// - `seq_len`: 当前序列长度
/// - `total_seq_len`: 包含缓存在内的上下文 + 当前长度
/// - `dqkv`: 每个头的维度大小 d_head
///
/// 要求：n_q_h = n_kv_h * n_groups。
/// 使用前请先 matmul 计算 Q, K, V，再将本函数呼叫做注意力，输出到 hidden_states。
pub fn self_attention_multihead<T: Default + Copy + Sum + Float + FromPrimitive>(
    hidden_states: &mut crate::tensor::Tensor<T>,
    att_scores: &mut crate::tensor::Tensor<T>,
    q: &crate::tensor::Tensor<T>,
    k: &crate::tensor::Tensor<T>,
    v: &crate::tensor::Tensor<T>,
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // ========== Step 1: 先填 0 ==========
    {
        let att_data = unsafe { att_scores.data_mut() };
        att_data.fill(T::zero());
    }

    // ========== Step 2: 计算 Q×K^T => 填进 att_scores ==========
    // 这里再用 "att_data" 但记得别留到 masked_softmax 之后
    let q_data = q.data();
    let k_data = k.data();
    let inv_scale = (T::one()/ (T::from(dqkv).unwrap() ).sqrt());

    let num_key_value_heads = n_kv_h;
    let num_query_heads_per_kv_group = n_groups;
    let num_attention_heads = n_kv_h * n_groups;
    let d_head = dqkv;

    let total_d_q = num_attention_heads * d_head;
    let total_d_kv = num_key_value_heads * d_head;

    let total_d_atts_3 = num_query_heads_per_kv_group * seq_len * total_seq_len;
    let total_d_atts_2 = seq_len * total_seq_len;
    let total_d_atts_1 = total_seq_len;

    {
        // 因为还没调用 masked_softmax，这里可以暂时可变借用
        let att_data = unsafe { att_scores.data_mut() };

        for curr_k_head in 0..num_key_value_heads {
            let offset_k = curr_k_head * d_head;
            for curr_q_in_group in 0..num_query_heads_per_kv_group {
                let curr_att_head = curr_k_head*num_query_heads_per_kv_group + curr_q_in_group;
                let offset_q = curr_att_head*d_head;
                for i_seq in 0..seq_len {
                    let begin_vec_q = i_seq*total_d_q + offset_q;
                    for i_tseq in 0..total_seq_len {
                        let begin_vec_k = i_tseq*total_d_kv + offset_k;
                        let mut dot = T::zero();
                        for dd in 0..d_head {
                            dot = dot+ q_data[begin_vec_q+dd]*k_data[begin_vec_k+dd];
                        }
                        dot = dot * inv_scale;

                        let att_idx = curr_k_head*total_d_atts_3
                            + curr_q_in_group*total_d_atts_2
                            + i_seq*total_d_atts_1
                            + i_tseq;
                        att_data[att_idx] = dot;
                    }
                }
            }
        }
    }

    // ========== Step 3: 调 masked_softmax(att_scores) ==========
    crate::operators::masked_softmax(att_scores);
    // 这里需要 &mut att_scores

    // ========== Step 4: hidden_states = att_scores × V ==========
    // 这里重新用 read-only 的 att_data
    let att_data = att_scores.data(); // <-- 只读
    let v_data = v.data();
    {
        let hs_data = unsafe { hidden_states.data_mut() };
        hs_data.fill(T::zero());

        for curr_v_head in 0..num_key_value_heads {
            let offset_matrix_v_g = curr_v_head*d_head;
            for curr_q_in_group in 0..num_query_heads_per_kv_group {
                let offset_matrix_a_h = curr_q_in_group*total_d_atts_2
                    + curr_v_head*total_d_atts_3;
                for curr_idx_seq in 0..seq_len {
                    let begin_vec_a = offset_matrix_a_h + curr_idx_seq*total_d_atts_1;
                    for curr_idx_dhead in 0..d_head {
                        let begin_vec_v = curr_idx_dhead + offset_matrix_v_g;
                        let mut sum_ = T::zero();
                        for curr_idx_tseq in 0..total_seq_len {
                            let idx_a = begin_vec_a + curr_idx_tseq;
                            let idx_v = begin_vec_v + curr_idx_tseq*total_d_kv;
                            sum_ = sum_ + att_data[idx_a] * v_data[idx_v];
                        }

                        let curr_att_head = curr_v_head*num_query_heads_per_kv_group + curr_q_in_group;
                        let hs_offset = curr_idx_seq*(num_attention_heads*d_head)
                            + curr_att_head*d_head
                            + curr_idx_dhead;
                        hs_data[hs_offset] = sum_;
                    }
                }
            }
        }
    }
}

pub fn self_attention(
    hidden_states: &mut crate::tensor::Tensor<f32>,
    att_scores: &mut crate::tensor::Tensor<f32>,
    q: &crate::tensor::Tensor<f32>,
    k: &crate::tensor::Tensor<f32>,
    v: &crate::tensor::Tensor<f32>,
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // ========== Step 1: 先填 0 ==========
    {
        let att_data = unsafe { att_scores.data_mut() };
        att_data.fill(0.0);
    }

    // ========== Step 2: 计算 Q×K^T => 填进 att_scores ==========
    // 这里再用 "att_data" 但记得别留到 masked_softmax 之后
    let q_data = q.data();
    let k_data = k.data();
    let inv_scale = 1.0 / (dqkv as f32).sqrt();

    let num_key_value_heads = n_kv_h;
    let num_query_heads_per_kv_group = n_groups;
    let num_attention_heads = n_kv_h * n_groups;
    let d_head = dqkv;

    let total_d_q = num_attention_heads * d_head;
    let total_d_kv = num_key_value_heads * d_head;

    let total_d_atts_3 = num_query_heads_per_kv_group * seq_len * total_seq_len;
    let total_d_atts_2 = seq_len * total_seq_len;
    let total_d_atts_1 = total_seq_len;

    {
        // 因为还没调用 masked_softmax，这里可以暂时可变借用
        let att_data = unsafe { att_scores.data_mut() };

        for curr_k_head in 0..num_key_value_heads {
            let offset_k = curr_k_head * d_head;
            for curr_q_in_group in 0..num_query_heads_per_kv_group {
                let curr_att_head = curr_k_head*num_query_heads_per_kv_group + curr_q_in_group;
                let offset_q = curr_att_head*d_head;
                for i_seq in 0..seq_len {
                    let begin_vec_q = i_seq*total_d_q + offset_q;
                    for i_tseq in 0..total_seq_len {
                        let begin_vec_k = i_tseq*total_d_kv + offset_k;
                        let mut dot = 0.0;
                        for dd in 0..d_head {
                            dot += q_data[begin_vec_q+dd]*k_data[begin_vec_k+dd];
                        }
                        dot *= inv_scale;

                        let att_idx = curr_k_head*total_d_atts_3
                            + curr_q_in_group*total_d_atts_2
                            + i_seq*total_d_atts_1
                            + i_tseq;
                        att_data[att_idx] = dot;
                    }
                }
            }
        }
    }

    // ========== Step 3: 调 masked_softmax(att_scores) ==========
    crate::operators::masked_softmax(att_scores);
    // 这里需要 &mut att_scores

    // ========== Step 4: hidden_states = att_scores × V ==========
    // 这里重新用 read-only 的 att_data
    let att_data = att_scores.data(); // <-- 只读
    let v_data = v.data();
    {
        let hs_data = unsafe { hidden_states.data_mut() };
        hs_data.fill(0.0);

        for curr_v_head in 0..num_key_value_heads {
            let offset_matrix_v_g = curr_v_head*d_head;
            for curr_q_in_group in 0..num_query_heads_per_kv_group {
                let offset_matrix_a_h = curr_q_in_group*total_d_atts_2
                    + curr_v_head*total_d_atts_3;
                for curr_idx_seq in 0..seq_len {
                    let begin_vec_a = offset_matrix_a_h + curr_idx_seq*total_d_atts_1;
                    for curr_idx_dhead in 0..d_head {
                        let begin_vec_v = curr_idx_dhead + offset_matrix_v_g;
                        let mut sum_ = 0.0;
                        for curr_idx_tseq in 0..total_seq_len {
                            let idx_a = begin_vec_a + curr_idx_tseq;
                            let idx_v = begin_vec_v + curr_idx_tseq*total_d_kv;
                            sum_ += att_data[idx_a] * v_data[idx_v];
                        }

                        let curr_att_head = curr_v_head*num_query_heads_per_kv_group + curr_q_in_group;
                        let hs_offset = curr_idx_seq*(num_attention_heads*d_head)
                            + curr_att_head*d_head
                            + curr_idx_dhead;
                        hs_data[hs_offset] = sum_;
                    }
                }
            }
        }
    }
}

fn mlp<T: Default + Copy + Sum + Float + FromPrimitive>(
    residual: &mut Tensor<T>,
    hidden_states: &mut Tensor<T>,
    gate: &mut Tensor<T>,
    up: &mut Tensor<T>,
    w_up: &Tensor<T>,
    w_down: &Tensor<T>,
    w_gate: &Tensor<T>,
    rms_w: &Tensor<T>,
    eps: T,
) {
    OP::rms_norm(hidden_states, &residual, &rms_w, eps);
    OP::matmul_transb(gate, T::zero(), &hidden_states, &w_gate, T::one());
    OP::matmul_transb(up, T::zero(), &hidden_states, &w_up, T::one());
    OP::swiglu(up, &gate);
    OP::matmul_transb(residual, T::one(), &up, &w_down, T::one());
}

#[test]
pub fn test_self_attention() {
    let seq_len = 2;
    let total_seq_len = 4;
    let n_kv_h = 2;
    let n_groups = 1;
    let dqkv = 3;

    // Initialize simple test tensors for Q, K, and V
    let q_data = vec![
        0.1, 0.2, 0.3, // Q for seq_idx 0, head 0
        0.4, 0.5, 0.6, // Q for seq_idx 1, head 0
        0.7, 0.8, 0.9, // Q for seq_idx 0, head 1
        1.0, 1.1, 1.2, // Q for seq_idx 1, head 1
    ];
    let q = Tensor::<f32>::new(q_data, &vec![seq_len, n_kv_h * n_groups * dqkv]);

    let k_data = vec![
        0.1, 0.2, 0.3, // K for total_seq_idx 0, head 0
        0.4, 0.5, 0.6, // K for total_seq_idx 1, head 0
        0.7, 0.8, 0.9, // K for total_seq_idx 2, head 0
        1.0, 1.1, 1.2, // K for total_seq_idx 3, head 0
        1.3, 1.4, 1.5, // K for total_seq_idx 0, head 1
        1.6, 1.7, 1.8, // K for total_seq_idx 1, head 1
        1.9, 2.0, 2.1, // K for total_seq_idx 2, head 1
        2.2, 2.3, 2.4, // K for total_seq_idx 3, head 1
    ];
    let k = Tensor::<f32>::new(k_data, &vec![total_seq_len, n_kv_h * dqkv]);

    let v_data = vec![
        0.1, 0.2, 0.3, // V for total_seq_idx 0, head 0
        0.4, 0.5, 0.6, // V for total_seq_idx 1, head 0
        0.7, 0.8, 0.9, // V for total_seq_idx 2, head 0
        1.0, 1.1, 1.2, // V for total_seq_idx 3, head 0
        1.3, 1.4, 1.5, // V for total_seq_idx 0, head 1
        1.6, 1.7, 1.8, // V for total_seq_idx 1, head 1
        1.9, 2.0, 2.1, // V for total_seq_idx 2, head 1
        2.2, 2.3, 2.4, // V for total_seq_idx 3, head 1
    ];
    let v = Tensor::<f32>::new(v_data, &vec![total_seq_len, n_kv_h * dqkv]);

    // Initialize attention score tensor and hidden_states
    let mut att_scores = Tensor::<f32>::default(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, n_kv_h * n_groups * dqkv]);

    // Run self_attention
    self_attention(
        &mut hidden_states,
        &mut att_scores,
        &q,
        &k,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    // Check the results (example expected results, calculated manually for f32)
    let expected_hidden_states = Tensor::<f32>::new(
        vec![
            0.7825454, 0.8825454, 0.9825454, // Output for seq_idx 0, head 0
            1.1990090, 1.2990088, 1.3990089, // Output for seq_idx 0, head 1
            1.5267198, 1.6267197, 1.7267196, // Output for seq_idx 1, head 0
            1.9442390, 2.0442388, 2.1442390, // Output for seq_idx 1, head 1
        ],
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );

    // Use float_eq for comparison of floating-point values
    assert!(hidden_states.close_to(&expected_hidden_states, 1e-5));
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

// #[test]
// pub fn test_load_safetensors() {
//     use crate::tensor::float_eq;
//     use std::path::PathBuf;
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let model = Llama::from_safetensors(model_dir);
//     assert_eq!(model.vocab, 2048);
//     assert_eq!(model.n_layers, 2);
//     assert_eq!(model.n_q_h, 8);
//     assert_eq!(model.n_kv_h, 4);
//     assert_eq!(model.d, 128);
//     assert_eq!(model.dqkv, 16);
//     assert_eq!(model.di, 384);

//     assert!(float_eq(
//         &model.params.embedding_table.data()[50],
//         &0.14453125,
//         1e-6
//     ));
//     assert_eq!(
//         model.params.lm_head.data()[10],
//         model.params.embedding_table.data()[10]
//     );
//     assert!(float_eq(
//         &model.params.rms_att_w[0].data()[10],
//         &0.18652344,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.rms_ffn_w[1].data()[10],
//         &0.32421875,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.rms_out_w.data()[100],
//         &0.73046875,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.w_down[0].data()[100],
//         &-0.0625,
//         1e-6
//     ));
//     assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
//     assert!(float_eq(
//         &model.params.w_gate[1].data()[100],
//         &0.296875,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.wq[1].data()[100],
//         &0.032226563,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.wk[1].data()[100],
//         &-0.21386719,
//         1e-6
//     ));
//     assert!(float_eq(
//         &model.params.wv[0].data()[100],
//         &0.041015625,
//         1e-6
//     ));
//     assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
// }
