use core::f32;
use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
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
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let mut params = LLamaParams::from_safetensors(&safetensor, &config);

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

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let n_heads = self.n_q_h; // Q 头数 = K,V 头数
        let dqkv = self.dqkv;     // 每个头的维度大小 = hidden_size / n_heads

        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // 预分配临时张量，复用内存
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // 计算开始
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            // Q, K, V 的计算
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]);
            let k = &mut cache.k_cache(layer, past_seq_len);
            let v = &mut cache.v_cache(layer, past_seq_len);

            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            // println!(
            //     "Layer {}: hidden_states shape: {:?}, wk[layer] shape: {:?}",
            //     layer,
            //     hidden_states.shape(),
            //     self.params.wk[layer].shape()
            // );
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);

            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0);
            let full_v = &mut cache.v_cache(layer, 0);

            self_attention_multihead(
                &mut hidden_states, // 最终输出张量 [seq_len, n_heads * dqkv]
                &q,             // 输入 Q 张量 [seq_len, n_heads * dqkv]
                full_k,             // 输入 K 张量 [total_seq_len, n_kv_heads * dqkv]
                full_v,             // 输入 V 张量 [total_seq_len, n_kv_heads * dqkv]
                &mut att_scores,    // 用于存储 Q × K^T 结果 [n_heads, seq_len, total_seq_len]
                self.n_q_h,         // Q 的头数 (n_q_heads)
                self.n_kv_h,        // K 和 V 的头数 (n_kv_heads)
                dqkv,               // 每个头的维度大小
                seq_len,            // 当前序列长度
                total_seq_len,      // 包括缓存在内的总序列长度
            );

            // residual = residual.add(&hidden_states, |x, y| x + y);

            OP::matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
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
                self.eps,
            );
        }

        // 输出 logits
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        // 避免 mutable 和 immutable borrow 冲突
        // let hidden_states_last = hidden_states.clone();
        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }


    // pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
    //     let n_heads = self.n_q_h;  // = self.n_kv_h
    //     let dqkv = self.dqkv;      // = hidden_size / n_heads
    //
    //     let seq_len = input.size();
    //     let past_seq_len = cache.len();
    //     cache.increment(seq_len);
    //     let total_seq_len = past_seq_len + seq_len;
    //     let n_groups = self.n_q_h / self.n_kv_h;
    //
    //     // Some pre-allocated buffers that will be reused
    //     let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
    //     let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
    //     let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
    //     let mut att_scores =
    //         Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
    //     let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
    //     let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
    //
    //     // Computation Starts Here
    //     // Embedding lookup
    //     OP::gather(&mut residual, input, &self.params.embedding_table);
    //
    //     for layer in 0..self.n_layers {
    //         OP::rms_norm(
    //             &mut hidden_states,
    //             &residual,
    //             &self.params.rms_att_w[layer],
    //             self.eps,
    //         );
    //
    //         let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
    //         let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
    //         let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
    //         OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
    //         // 在调用前打印详细信息
    //         println!(
    //             "Layer {}: Shape of wk[layer]: {:?}",
    //             layer,
    //             self.params.wk[layer].shape()
    //         );
    //         // fixme !("matmul_transb"); assertion `left == right` failed: A 的列数应等于 B 转置的行数
    //         OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
    //         OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
    //         OP::rope(
    //             q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
    //             past_seq_len,
    //             self.rope_theta,
    //         );
    //         OP::rope(
    //             k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
    //             past_seq_len,
    //             self.rope_theta,
    //         );
    //
    //         let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
    //         let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
    //
    //         // todo!("self_attention(...)");
    //         // self_attention(
    //         //     &mut hidden_states,
    //         //     &mut att_scores,
    //         //     &q_buf,
    //         //     full_k,
    //         //     full_v,
    //         //     self.n_kv_h,
    //         //     n_groups,
    //         //     seq_len,
    //         //     total_seq_len,
    //         //     self.dqkv,
    //         // );
    //
    //         // 5) 调用多头Attention
    //         self_attention_multihead(
    //             &mut hidden_states, // out: [seq_len, n_heads*dqkv]
    //             &q_buf,            // [seq_len, n_heads, dqkv]
    //             full_k,            // [total_seq, n_heads*dqkv] => reshape => [total_seq, n_heads, dqkv]
    //             full_v,
    //             &mut att_scores,   // [n_heads, seq_len, total_seq_len], 复用
    //             n_heads,
    //             dqkv,
    //             seq_len,
    //             total_seq_len
    //         );
    //
    //         // (可选) down_proj => residual
    //         // let wo = &self.params.wo[layer_i]; // shape [hidden_size, hidden_size]
    //         // 这里没写，若需要 => matmul_transb(&mut hidden_states, 0.0, &hidden_states, wo, 1.0);
    //         // todo!("down_proj matmul and add residual");
    //         // Residual connection (residual = residual + hidden_states)
    //         residual = residual.add(&hidden_states, |x, y| x + y);
    //
    //         // RMS Normalization for FFN
    //         OP::rms_norm(
    //             &mut hidden_states,
    //             &residual,
    //             &self.params.rms_ffn_w[layer],
    //             self.eps,
    //         );
    //
    //         // todo!("mlp(...)");
    //         // Feedforward Network (MLP)
    //         mlp(
    //             &mut residual,
    //             &mut hidden_states,
    //             &mut gate_buf,
    //             &mut up_buf,
    //             &self.params.w_up[layer],
    //             &self.params.w_down[layer],
    //             &self.params.w_gate[layer],
    //             &self.params.rms_ffn_w[layer],
    //             self.eps,
    //         );
    //     }
    //
    //     // No matter what seq_len, the output is always a 1D vector of length vocab,
    //     // which contains the probabilities for the next token.
    //     let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
    //     let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
    //     let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);
    //
    //     OP::rms_norm(
    //         &mut hidden_states,
    //         &residual,
    //         &self.params.rms_out_w,
    //         self.eps,
    //     );
    //
    //     OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);
    //
    //     logits
    // }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result = token_ids.to_vec();
        //todo 实现文本生成
        let mut cache = self.new_cache();

        for _ in 0..max_len {
            // 只处理当前新增的 token，而不是整个序列
            let input_tensor = Tensor::<u32>::new(vec![*result.last().unwrap()], &vec![1]);

            // forward 方法利用 cache，只计算新增 token 的 logits
            let logits = self.forward(&input_tensor, &mut cache);

            // 使用随机采样策略生成下一个 token
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);

            // 如果遇到终止符号，提前结束生成
            if next_token == self.eos_token_id {
                break;
            }
            // 将生成的 token 添加到结果中
            result.push(next_token);

        }

        result
    }

    pub fn streaming_generate<'a>(
        &'a self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kvcache: &'a mut KVCache<f32>,
    ) -> impl Iterator<Item = u32> + 'a {
        let mut result_tokens = token_ids.to_vec(); // 保存生成的 token
        let mut input_tensors = Tensor::<u32>::new(result_tokens.clone(), &vec![result_tokens.len()]);

        // 返回一个迭代器
        std::iter::from_fn(move || {
            // 如果达到最大长度，停止生成
            if result_tokens.len() >= max_len {
                return None;
            }

            // 前向传播获取 logits
            let logits = self.forward(&input_tensors, kvcache);

            // 根据 top_p、top_k 和 temperature 采样下一个 token
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);

            // 将生成的 token 添加到结果中
            result_tokens.push(next_token);

            // 更新输入张量为新生成的 token
            input_tensors = Tensor::<u32>::new(vec![next_token], &vec![1]);

            // 如果生成的 token 是 EOS，停止生成
            if next_token == self.eos_token_id {
                None
            } else {
                Some(next_token) // 返回生成的 token
            }
        })
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // self_attention

    // 计算注意力分数
    // 计算注意力分数 (q @ k^T)
    OP::matmul_transb(att_scores, 0.0, q, k, 1.0 / (dqkv as f32).sqrt());

    // 应用 masked softmax
    OP::masked_softmax(att_scores);

    // 计算注意力输出 (y = att_scores @ v)
    OP::matmul_transb(hidden_states, 0.0, att_scores, v, 1.0);

    // n_q_h = n_kv_h * n_groups
    let n_q_h = n_kv_h * n_groups;

    // 1) 将 q, k, v 的 data() 取出来
    // Q: shape [seq_len, n_q_h*dqkv]
    let q_data = q.data();
    // K,V: shape [total_seq_len, n_kv_h*dqkv]
    let k_data = k.data();
    let v_data = v.data();

    // 2) 拿 att_scores 的 data_mut
    //   shape=[n_kv_h, n_groups, seq_len, total_seq_len]
    //   => index = i_kv * n_groups * seq_len * total_seq_len
    //             + i_group * seq_len * total_seq_len
    //             + i_seq * total_seq_len
    //             + i_tseq
    let att_score_data = unsafe { att_scores.data_mut() };

    // 3) 先将 att_scores清0
    att_score_data.fill(0.0);

    // 4) 计算 Q x K^T / sqrt(dqkv)
    //   对每个 kv_head in [0..n_kv_h]
    //   对每个 group in [0..n_groups]
    //   对每个 seq in [0..seq_len]
    //   对每个 tseq in [0..total_seq_len]
    //   => sum_{d in [0..dqkv]} Q[seq, q_head* dqkv + d]* K[tseq, kv_head*dqkv + d]

    let inv_scale = 1.0 / (dqkv as f32).sqrt();

    for i_kv in 0..n_kv_h {
        for g in 0..n_groups {
            let i_q = i_kv * n_groups + g; // q_head index
            // i_q in [0..n_q_h], i_kv in [0..n_kv_h], ratio = n_groups

            for s in 0..seq_len {
                for ts in 0..total_seq_len {
                    let mut sum = 0.0;
                    // dot product for dqkv
                    for d in 0..dqkv {
                        let q_idx = s * (n_q_h * dqkv) + i_q * dqkv + d;
                        let k_idx = ts * (n_kv_h * dqkv) + i_kv * dqkv + d;
                        sum += q_data[q_idx] * k_data[k_idx];
                    }
                    // multiply by inv_scale
                    sum *= inv_scale;

                    let att_idx = i_kv * n_groups * seq_len * total_seq_len
                        + g * seq_len * total_seq_len
                        + s * total_seq_len
                        + ts;
                    att_score_data[att_idx] = sum;
                }
            }
        }
    }

    // 5) masked softmax
    OP::masked_softmax(att_scores);

    // 6) 计算 attn_V = att_scores @ V
    //   hidden_states = [seq_len, n_kv_h*n_groups*dqkv]
    //   => We'll do shape [n_kv_h, n_groups, seq_len, total_seq_len]
    //   x   v [total_seq_len, n_kv_h*dqkv]
    //   依然需要分头, i_kv * dqkv
    let out_data = unsafe { hidden_states.data_mut() };
    out_data.fill(0.0);

    let att_data = att_scores.data(); // read-only

    // hidden_states shape: [seq_len, n_kv_h*n_groups*dqkv]
    // index = s*(n_kv_h*n_groups*dqkv) + i_kv*n_groups*dqkv + g*dqkv + d

    for i_kv in 0..n_kv_h {
        for g in 0..n_groups {
            let i_q = i_kv * n_groups + g; // same logic as above
            for s in 0..seq_len {
                for d in 0..dqkv {
                    let mut sum = 0.0;
                    // \sum_{ts} att_scores * V
                    for ts in 0..total_seq_len {
                        let att_idx = i_kv * n_groups * seq_len * total_seq_len
                            + g * seq_len * total_seq_len
                            + s * total_seq_len
                            + ts;
                        let v_idx = ts*(n_kv_h*dqkv) + i_kv*dqkv + d;
                        sum += att_data[att_idx]*v_data[v_idx];
                    }
                    // put into hidden_states
                    let out_idx = s*(n_kv_h*n_groups*dqkv) + i_q*dqkv + d;
                    out_data[out_idx] = sum;
                }
            }
        }
    }
}

pub fn self_attention_multihead(
    hidden_states: &mut Tensor<f32>, // 输出: [seq_len, n_q_h * dqkv]
    q: &Tensor<f32>,                 // 输入 Q: [seq_len, n_q_h * dqkv]
    k: &Tensor<f32>,                 // 输入 K: [total_seq_len, n_kv_h * dqkv]
    v: &Tensor<f32>,                 // 输入 V: [total_seq_len, n_kv_h * dqkv]
    att_scores: &mut Tensor<f32>,    // 中间结果: [n_kv_h, n_groups, seq_len, total_seq_len]
    n_q_h: usize,                    // Q 的头数
    n_kv_h: usize,                   // K 和 V 的头数
    dqkv: usize,                     // 每个头的维度大小
    seq_len: usize,                  // 当前序列长度
    total_seq_len: usize,            // 总序列长度 (包括缓存)
) {
    let n_groups = n_q_h / n_kv_h; // Q 的头数是 KV 的整数倍
    let scale = (dqkv as f32).sqrt();

    // =========== Step 1: Q × K^T ===========
    for kv_head in 0..n_kv_h {
        for group in 0..n_groups {
            let q_head = kv_head * n_groups + group;

            // 提取对应的 Q 和 K 块
            let q_sub = q.slice(q_head * dqkv, &vec![seq_len, dqkv]);
            let k_sub = k.slice(kv_head * dqkv, &vec![total_seq_len, dqkv]);

            // 提取 att_scores 的当前块
            let mut att_sub = att_scores.slice(
                (kv_head * n_groups + group) * seq_len * total_seq_len,
                &vec![seq_len, total_seq_len],
            );

            // 计算 att_scores: Q × K^T
            OP::matmul_transb(&mut att_sub, 0.0, &q_sub, &k_sub, 1.0 / scale);
        }
    }

    // =========== Step 2: Apply softmax ===========
    OP::masked_softmax(att_scores);

    // =========== Step 3: att_scores × V ===========
    for kv_head in 0..n_kv_h {
        for group in 0..n_groups {
            let q_head = kv_head * n_groups + group;

            // 提取 att_scores 和 V 的对应块
            let att_sub = att_scores.slice(
                (kv_head * n_groups + group) * seq_len * total_seq_len,
                &vec![seq_len, total_seq_len],
            );
            let v_sub = v.slice(kv_head * dqkv, &vec![total_seq_len, dqkv]);

            // 提取 hidden_states 的对应块
            let mut out_sub = hidden_states.slice(q_head * dqkv, &vec![seq_len, dqkv]);

            // 计算 hidden_states: att_scores × V
            OP::matmul(&mut out_sub, 0.0, &att_sub, &v_sub, 1.0);
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // Implement mlp

    // 确保 rms_w 的形状与 hidden_states 的最后一个维度一致
    assert_eq!(
        hidden_states.shape().last(),
        Some(&rms_w.size()),
        "rms_w size must match hidden_states last dimension"
    );

    // Step 1: RMS Normalization
    OP::rms_norm(hidden_states, residual, rms_w, eps);

    // Step 2: Compute Gate and Up Projections
    OP::matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);
    OP::matmul_transb(up, 0.0, hidden_states, w_up, 1.0);

    // Step 3: Apply SiLU activation
    OP::swiglu(gate, up);

    // Step 4: Compute residual
    OP::matmul_transb(residual, 1.0, gate, w_down, 1.0);
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

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
