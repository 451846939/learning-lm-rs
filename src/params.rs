use core::f32;
use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 打印 safetensors 中所有可用的张量名称
        println!("Available tensors: {:?}", safetensor.names());

        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor_view = safetensor
                .tensor(name)
                .unwrap_or_else(|_| panic!("Tensor `{}` not found in safetensors", name));
            let tensor_dtype = tensor_view.dtype();
            let element_size = tensor_dtype.size();
            let data = tensor_view
                .data()
                .chunks_exact(element_size)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<f32>>();
            let shape = tensor_view.shape().to_vec();
            Tensor::<f32>::new(data, &shape)
        };

        let n_layers = config.num_hidden_layers;

        let embedding_table=if config.tie_word_embeddings {
            get_tensor("lm_head.weight")
        } else {
            get_tensor("model.embed_tokens.weight")
        };

        LLamaParams {
            embedding_table: embedding_table,
            rms_att_w: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.input_layernorm.weight")))
                .collect(),
            wq: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.o_proj.weight")))
                .collect(),
            rms_ffn_w: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..n_layers)
                .map(|f| get_tensor(&format!("model.layers.{f}.mlp.down_proj.weight")))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}