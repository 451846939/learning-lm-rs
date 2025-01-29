use std::collections::HashMap;
use std::io::{self, Write};
use std::iter::Sum;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use crate::kvcache::KVCache;
use crate::model;
use crate::config::{LlamaConfigJson, TorchDType};
use half::{bf16, f16};
use num_traits::{Float, FromPrimitive};
use crate::params::FromLeBytes;

#[derive(Clone)]
pub struct Message {
    pub(crate) role: String,
    pub(crate) content: String,
}

impl Message {
    pub(crate) fn format(&self) -> String {
        format!("<|im_start|>{}\n{}<|im_end|>\n", self.role, self.content)
    }
}

/// **ChatAI 结构体（支持泛型）**
pub struct ChatAI<T> {
    pub(crate) llama: model::Llama<T>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) kvcache: KVCache<T>,
    pub(crate) messages: Vec<Message>,
}

impl<T> ChatAI<T>
where
    T: Float + Default + Copy + Sum + FromPrimitive + FromLeBytes,
{
    pub fn new(model_dir: PathBuf, config: LlamaConfigJson) -> Self {
        let llama = model::Llama::<T>::from_safetensors(&model_dir, config);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let kvcache = llama.new_cache();

        ChatAI {
            llama,
            tokenizer,
            kvcache,
            messages: Vec::new(),
        }
    }

    /// 清理用户输入
    fn clean_input(&self, input: &str) -> String {
        input
            .split_whitespace()
            .filter(|word| word.chars().all(char::is_alphanumeric))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// **构造 Prompt**
    pub(crate) fn build_prompt(&self, add_generation_prompt: bool) -> String {
        let mut prompt = String::new();
        self.messages.first().map(|msg| {
            prompt.push_str(&msg.format());
        });
        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
        }
        prompt
    }

    /// **聊天逻辑**
    pub fn chat(&mut self, user_input: &str) -> String {
        self.messages.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        let conversation_input = self.build_prompt(true);
        let encoded = self.tokenizer.encode(conversation_input, true).unwrap();
        let input_ids = encoded.get_ids().to_vec();

        let response_tokens = self.llama.streaming_generate(
            &input_ids,
            128,
            T::from_f32(0.9).unwrap(),
            4,
            T::from_f32(1.0).unwrap(),
            &mut self.kvcache,
        );

        let mut response_text = String::new();
        print!("\rAssistant: ");

        for token in response_tokens {
            let token_str = self.tokenizer.decode(&[token], true).unwrap() + " ";
            response_text.push_str(&token_str);
            print!("{}", token_str);
            io::stdout().flush().unwrap();

            if token == self.llama.eos_token_id {
                break;
            }
        }
        println!();

        self.messages.push(Message {
            role: "assistant".to_string(),
            content: response_text.clone(),
        });

        response_text
    }
}