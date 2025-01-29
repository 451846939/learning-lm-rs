use serde;
use serde::Serialize;

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct LlamaConfigJson {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: TorchDType,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}

#[inline(always)]
fn default_torch_dtype() -> TorchDType {
    TorchDType::default()
}

#[derive(Serialize,Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TorchDType {
    Float32,
    Float16,
    BFloat16,
}


#[inline(always)]
const fn default_tie_word_embeddings() -> bool {
    false
}

impl Default for TorchDType {
    fn default() -> Self {
        TorchDType::Float32
    }
}

impl<'de> serde::Deserialize<'de> for TorchDType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s: String = serde::Deserialize::deserialize(deserializer)?;
        match s.as_str() {
            "float32" => Ok(TorchDType::Float32),
            "float16" => Ok(TorchDType::Float16),
            "bfloat16" => Ok(TorchDType::BFloat16),
            _ => {
                Err(serde::de::Error::invalid_value(
                    serde::de::Unexpected::Str(&s),
                    &"valid torch dtype",
                ))
            }
        }
    }
}