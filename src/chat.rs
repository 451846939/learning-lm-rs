use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use crate::kvcache::KVCache;
use crate::model;

struct Message {
    role: String,
    content: String,
}

impl Message {
    fn format(&self) -> String {
        format!("<|im_start|>{}\n{}<|im_end|>\n", self.role, self.content)
    }
}


pub struct ChatAI {
    llama: model::Llama<f32>,
    tokenizer: Tokenizer,
    kvcache: KVCache<f32>,
    messages: Vec<Message>,
}

impl ChatAI {
    pub fn new(model_dir: PathBuf) -> Self {
        let llama = model::Llama::<f32>::from_safetensors(&model_dir);
        // 加载分词器
        // let tokenizer_config_file = model_dir.join("tokenizer_config.json");
        // let tokenizer = if tokenizer_config_file.exists() {
        //     // 优先使用 tokenizer_config.json 配置文件
        //     let tokenizer_file = model_dir.join("tokenizer.model");
        //     Tokenizer::from_file(tokenizer_file).unwrap()
        // } else {
        //     // 回退到 tokenizer.json
        //     Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap()
        // };
        // 加载 tokenizer
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        // 从 Llama 实例中获取模型配置
        let n_layers = llama.n_layers; // 模型的层数
        let max_seq_len = llama.max_seq_len; // 最大序列长度
        let dim = llama.dqkv * llama.n_kv_h; // 每层键值对的维度（注意力头数 * 每个向量的长度）
        // 初始化 KVCache
        let kvcache = KVCache::new(n_layers, max_seq_len, dim, 0);
        ChatAI {
            llama,
            tokenizer,
            kvcache: kvcache,
            messages: Vec::new(),
        }
    }

    /// 清理用户输入
    fn clean_input(&self, input: &str) -> String {
        input
            .split_whitespace()
            .filter(|word| word.chars().all(char::is_alphanumeric)) // 保留字母和数字
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn chat(&mut self, user_input: &str) -> String {
        if user_input.trim().is_empty() {
            return "抱歉，我无法理解你的输入。可以再试一次吗？".to_string();
        }

        self.messages.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // Format the input using Jinja-like template for the model
        let conversation_input: String =
            self.messages.iter().map(|msg| msg.format()).collect::<String>() + "<|im_start|>assistant";

        let encoded = self.tokenizer.encode(conversation_input, true).unwrap();
        let input_ids = encoded.get_ids();

        print!("Assistant: ");
        io::stdout().flush().unwrap();

        let mut generated_tokens = vec![];
        let mut response_text = String::new();

        let response_tokens = self.llama.streaming_generate(
            input_ids,
            50,  // max_length
            0.8,  // temperature
            30,   // top_k
            0.9,  // top_p
            &mut self.kvcache,
        );

        for token in response_tokens {
            generated_tokens.push(token);

            // Decode the latest token only
            let token_text = self.tokenizer.decode(&[token], true).unwrap();
            response_text.push_str(&token_text);

            // Print the latest decoded text incrementally
            print!("{}", token_text);
            io::stdout().flush().unwrap();
        }

        println!();

        self.messages.push(Message {
            role: "assistant".to_string(),
            content: response_text.clone(),
        });

        response_text
    }
}