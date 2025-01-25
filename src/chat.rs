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
        let kvcache = llama.new_cache();
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

    // pub fn chat(&mut self, user_input: &str) -> String {
    //     if user_input.trim().is_empty() {
    //         return "抱歉，我无法理解你的输入。可以再试一次吗？".to_string();
    //     }
    //
    //     self.messages.push(Message {
    //         role: "user".to_string(),
    //         content: user_input.to_string(),
    //     });
    //
    //     // Format the input using Jinja-like template for the model
    //     let conversation_input: String =
    //         self.messages.iter().map(|msg| msg.format()).collect::<String>() + "<|im_start|>assistant\n";
    //
    //     let encoded = self.tokenizer.encode(conversation_input, true).unwrap();
    //     let input_ids = encoded.get_ids();
    //
    //     print!("Assistant: ");
    //     io::stdout().flush().unwrap();
    //
    //     // let mut generated_tokens = vec![];
    //     let mut response_text = String::new();
    //
    //     let response_tokens = self.llama.streaming_generate(
    //         input_ids,
    //         30,  // max_length
    //         0.9,  // temperature
    //         30,   // top_k
    //         0.9,  // top_p
    //         &mut self.kvcache,
    //     );
    //
    //     for token in response_tokens {
    //         // 解码“单个token”
    //         let token_str = self.tokenizer.decode(&[token], true).unwrap();
    //
    //         // 追加到完整回复里
    //         response_text.push_str(&token_str);
    //
    //         // 只打印新出现的这部分
    //         print!("{}", token_str);
    //         io::stdout().flush().unwrap();
    //     }
    //
    //     println!();
    //
    //     self.messages.push(Message {
    //         role: "assistant".to_string(),
    //         content: response_text.clone(),
    //     });
    //
    //     response_text
    // }

    // 根据 chat_template 拼接
    fn build_prompt(&self, add_generation_prompt: bool) -> String {
        let mut prompt = String::new();
        self.messages.first().map(|msg| {
            prompt.push_str(&msg.format());
        });
        // for msg in &self.messages {
        //     // "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
        //     prompt.push_str("<|im_start|>");
        //     prompt.push_str(&msg.role);
        //     prompt.push('\n');
        //     prompt.push_str(&msg.content);
        //     prompt.push_str("<|im_end|>\n");
        // }
        // 若需要追加 "assistant" 的启动标记，让模型来生成回答
        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
        }
        prompt
    }

    pub fn chat(&mut self, user_input: &str) -> String {
        // 1) 存入历史对话
        self.messages.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // 2) 构造 Prompt（不再手写 <|im_start|>user…<|im_end|>；直接用 build_prompt）
        let conversation_input = self.build_prompt(true);

        // 3) 编码
        let encoded = self.tokenizer.encode(conversation_input, /*add_special_tokens=*/true).unwrap();
        let input_ids = encoded.get_ids().to_vec();

        // 4) 流式生成
        let response_tokens = self.llama.streaming_generate(
            &input_ids,
            128,  // max_len, 可以视需要调整
            0.7,  // top_p
            30,   // top_k
            1.0,  // temperature, 可再调低些
            &mut self.kvcache,
        );

        // 5) 实时解码 & 打印
        let mut generated_tokens = vec![];
        let mut response_text = String::new();

        print!("\rAssistant: \n", );
        for token in response_tokens {
            generated_tokens.push(token);

            let token_str = self.tokenizer.decode(&[token], true).unwrap()+ " ";

            response_text.push_str(&token_str);
            // Decode the generated tokens so far
            // let partial_response = self.tokenizer
            //     .decode(&generated_tokens, true)
            //     .unwrap()
            //     .trim()
            //     .to_string();
            // 流式打印
            // print!("\rAssistant: {}", partial_response);
            // let word = tokenizer.decode(&[token], true).unwrap() + " ";
            print!("{}", token_str);
            std::io::stdout().flush().unwrap();

            // 如果是 EOS，就 break
            if token == self.llama.eos_token_id {
                break;
            }
        }
        println!();

        // 6) 作为一个新的 Message 存起来（role="assistant"）
        self.messages.push(Message {
            role: "assistant".to_string(),
            content: response_text.clone(),
        });

        response_text
    }
}